"""Modèle LQ et optimisation illustrative de doses par fraction."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from morphort.growth import GrowthPoint, fit_growth, predict_volume


def linear_quadratic_survival(dose_gy: float, alpha: float, beta: float) -> float:
    """Calcule la survie LQ : S = exp(-(alpha*d + beta*d²))."""
    _validate_positive(alpha, "alpha")
    _validate_positive(beta, "beta")
    if not np.isfinite(dose_gy) or dose_gy < 0:
        raise ValueError("La dose doit être positive ou nulle.")
    return float(np.exp(-(alpha * dose_gy + beta * dose_gy**2)))


def biologically_effective_dose(
    doses_gy: Sequence[float], alpha_beta_gy: float
) -> float:
    """Calcule la BED : somme d_i (1 + d_i / (alpha/beta))."""
    _validate_positive(alpha_beta_gy, "alpha/beta")
    doses = np.asarray(doses_gy, dtype=np.float64)
    if np.any(~np.isfinite(doses)) or np.any(doses < 0):
        raise ValueError("Les doses doivent être positives ou nulles.")
    return float(np.sum(doses * (1.0 + doses / alpha_beta_gy)))


def optimize_fraction_doses(
    initial_volume_ml: float,
    rate: float,
    capacity_ml: float,
    model: str,
    alpha: float,
    beta: float,
    healthy_alpha_beta_gy: float,
    healthy_bed_limit_gy: float,
    fractions: int,
    *,
    maximum_dose_gy: float = 5.0,
) -> NDArray[np.float64]:
    """Optimise des fractions sous contrainte BED du tissu sain.

    Formules : log(V_final)=log(V0)+croissance-somme(alpha*d+beta*d²) ;
    BED_sain=somme d(1+d/(alpha/beta)_sain) <= BED_max.
    Ce modèle est pédagogique, non calibré pour la planification d'un patient.
    """
    for value, name in (
        (initial_volume_ml, "volume initial"),
        (capacity_ml, "capacité K"),
        (alpha, "alpha"),
        (beta, "beta"),
        (healthy_alpha_beta_gy, "alpha/beta sain"),
        (maximum_dose_gy, "dose maximale"),
    ):
        _validate_positive(value, name)
    if not np.isfinite(healthy_bed_limit_gy) or healthy_bed_limit_gy < 0:
        raise ValueError("La BED maximale doit être positive ou nulle.")
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Le taux de croissance r doit être positif ou nul.")
    if fractions < 1:
        raise ValueError("Le nombre de fractions doit être supérieur à zéro.")
    if model not in ("Gompertz", "Logistique"):
        raise ValueError("Modèle de croissance inconnu.")

    maximum_possible_bed = (
        fractions * maximum_dose_gy * (1.0 + maximum_dose_gy / healthy_alpha_beta_gy)
    )
    if healthy_bed_limit_gy > maximum_possible_bed:
        healthy_bed_limit_gy = maximum_possible_bed
    if healthy_bed_limit_gy == 0:
        return np.zeros(fractions, dtype=np.float64)
    bed_per_fraction = healthy_bed_limit_gy / fractions
    initial_dose = min(
        maximum_dose_gy,
        healthy_alpha_beta_gy
        * (np.sqrt(1.0 + 4.0 * bed_per_fraction / healthy_alpha_beta_gy) - 1.0)
        / 2.0,
    )
    start = np.full(fractions, initial_dose, dtype=np.float64)

    def objective(doses: NDArray[np.float64]) -> float:
        volume = initial_volume_ml
        for dose in doses:
            volume = predict_volume(volume, rate, capacity_ml, 1.0, model)
            volume *= linear_quadratic_survival(dose, alpha, beta)
        return float(np.log(max(volume, np.finfo(np.float64).tiny)))

    def remaining_bed(doses: NDArray[np.float64]) -> float:
        return healthy_bed_limit_gy - biologically_effective_dose(
            doses, healthy_alpha_beta_gy
        )

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, maximum_dose_gy)] * fractions,
        constraints={"type": "ineq", "fun": remaining_bed},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"L'optimisation SLSQP a échoué : {result.message}")
    if remaining_bed(result.x) < -1e-5:
        raise RuntimeError("L'optimiseur a dépassé la limite BED du tissu sain.")
    return np.asarray(result.x, dtype=np.float64)


def simulate_plans(
    observations: list[GrowthPoint],
    model: str,
    alpha: float,
    beta: float,
    healthy_alpha_beta_gy: float,
    healthy_bed_limit_gy: float,
    fractions: int,
) -> dict[str, object]:
    """Compare un plan fixe et un plan adaptatif lors d'examens successifs.

    Le plan fixe est optimisé une fois. Le plan adaptatif réajuste r, K et les
    doses restantes à chaque examen hypothétique, uniquement après trois mesures.
    """
    observations = sorted(observations, key=lambda point: point[0])
    if len(observations) < 3:
        raise ValueError("La simulation nécessite au moins trois examens.")
    baseline_points = observations[:3]
    baseline = fit_growth(baseline_points, model)
    rate = float(baseline["r"])
    capacity = float(baseline["K"])
    initial_volume = float(baseline["volumes"][0])
    fixed_doses = optimize_fraction_doses(
        initial_volume,
        rate,
        capacity,
        model,
        alpha,
        beta,
        healthy_alpha_beta_gy,
        healthy_bed_limit_gy,
        fractions,
    )
    fixed = _run_schedule(
        initial_volume, fixed_doses, rate, capacity, model, alpha, beta
    )

    adaptive_doses = fixed_doses.copy()
    adaptive_points = baseline_points.copy()
    adaptive_volume = initial_volume
    adaptive_trajectory: list[dict[str, float | int]] = []
    checkpoint_indices = _checkpoint_indices(len(observations), fractions)
    fitted_at_checkpoint = dict(baseline)
    for fraction_index in range(fractions):
        if fraction_index in checkpoint_indices:
            observation_index = checkpoint_indices[fraction_index]
            adaptive_points = observations[: observation_index + 1]
            fitted_at_checkpoint = fit_growth(adaptive_points, model)
            adaptive_volume = float(adaptive_points[-1][1])
            remaining = fractions - fraction_index
            if remaining:
                adaptive_doses[fraction_index:] = optimize_fraction_doses(
                    adaptive_volume,
                    float(fitted_at_checkpoint["r"]),
                    float(fitted_at_checkpoint["K"]),
                    model,
                    alpha,
                    beta,
                    healthy_alpha_beta_gy,
                    healthy_bed_limit_gy
                    - biologically_effective_dose(
                        adaptive_doses[:fraction_index],
                        healthy_alpha_beta_gy,
                    ),
                    remaining,
                )
        dose = float(adaptive_doses[fraction_index])
        adaptive_volume = predict_volume(
            adaptive_volume,
            float(fitted_at_checkpoint["r"]),
            float(fitted_at_checkpoint["K"]),
            1.0,
            model,
        )
        adaptive_volume *= linear_quadratic_survival(dose, alpha, beta)
        adaptive_trajectory.append(
            {
                "fraction": fraction_index + 1,
                "volume_ml": adaptive_volume,
                "dose_gy": dose,
                "bed_gy": biologically_effective_dose(
                    adaptive_doses[: fraction_index + 1],
                    healthy_alpha_beta_gy,
                ),
            }
        )

    return {
        "fixed": fixed,
        "adaptive": adaptive_trajectory,
        "fixed_doses": fixed_doses.tolist(),
        "adaptive_doses": adaptive_doses.tolist(),
        "fixed_bed_gy": biologically_effective_dose(fixed_doses, healthy_alpha_beta_gy),
        "adaptive_bed_gy": biologically_effective_dose(
            adaptive_doses, healthy_alpha_beta_gy
        ),
        "baseline_fit": baseline,
    }


def _checkpoint_indices(observation_count: int, fractions: int) -> dict[int, int]:
    """Associe les examens postérieurs à des fractions de réévaluation."""
    checkpoint_map: dict[int, int] = {}
    for observation_index in range(3, observation_count):
        fraction_index = round(
            (observation_index - 2) * fractions / (observation_count - 2)
        )
        fraction_index = min(fractions - 1, max(0, fraction_index))
        checkpoint_map[fraction_index] = observation_index
    return checkpoint_map


def _run_schedule(
    initial_volume: float,
    doses: NDArray[np.float64],
    rate: float,
    capacity: float,
    model: str,
    alpha: float,
    beta: float,
) -> list[dict[str, float | int]]:
    """Calcule le volume simulé après chaque fraction du plan."""
    volume = initial_volume
    trajectory: list[dict[str, float | int]] = []
    for index, dose in enumerate(doses):
        volume = predict_volume(volume, rate, capacity, 1.0, model)
        volume *= linear_quadratic_survival(float(dose), alpha, beta)
        trajectory.append(
            {
                "fraction": index + 1,
                "volume_ml": volume,
                "dose_gy": float(dose),
            }
        )
    return trajectory


def _validate_positive(value: float, name: str) -> None:
    """Refuse les paramètres numériques nuls, négatifs ou non finis."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} doit être strictement positif.")
