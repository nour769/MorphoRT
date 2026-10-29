"""Ajustement des modèles mathématiques de croissance tumorale."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

GrowthModel = str
GrowthPoint = tuple[float, float]
GrowthFunction = Callable[
    [NDArray[np.float64], float, float, float], NDArray[np.float64]
]


def gompertz_volume(
    day: NDArray[np.float64], rate: float, capacity: float, initial: float
) -> NDArray[np.float64]:
    """Évalue Gompertz : V(t)=K exp(ln(V₀/K) exp(-r t))."""
    return capacity * np.exp(np.log(initial / capacity) * np.exp(-rate * day))


def simulate_gompertz_followup(
    initial_volume: float,
    rate: float,
    capacity: float,
    days: list[float],
    relative_noise: float,
    *,
    seed: int = 42,
) -> list[GrowthPoint]:
    """Simule V(t)=K exp(ln(V₀/K) exp(-rt)) avec bruit relatif reproductible."""
    if (
        not np.isfinite(initial_volume)
        or initial_volume <= 0
        or not np.isfinite(rate)
        or rate <= 0
        or not np.isfinite(capacity)
        or capacity <= initial_volume
        or not np.isfinite(relative_noise)
        or not 0 <= relative_noise <= 0.5
    ):
        raise ValueError("Les paramètres du suivi simulé sont invalides.")
    times = np.asarray(days, dtype=np.float64)
    if (
        times.size < 3
        or not np.all(np.isfinite(times))
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Le suivi simulé exige au moins trois jours croissants à partir de zéro."
        )

    volumes = gompertz_volume(times, rate, capacity, initial_volume)
    random = np.random.default_rng(seed)
    if relative_noise:
        volumes[1:] *= random.normal(1.0, relative_noise, times.size - 1)
        volumes[1:] = np.maximum(volumes[1:], initial_volume * 0.05)
    volumes[0] = initial_volume
    return list(zip(times.tolist(), volumes.tolist(), strict=True))


def logistic_volume(
    day: NDArray[np.float64], rate: float, capacity: float, initial: float
) -> NDArray[np.float64]:
    """Évalue la logistique : V(t)=K/(1+((K-V₀)/V₀) exp(-r t))."""
    return capacity / (1.0 + ((capacity - initial) / initial) * np.exp(-rate * day))


MODEL_FUNCTIONS: dict[str, GrowthFunction] = {
    "Gompertz": gompertz_volume,
    "Logistique": logistic_volume,
}


def fit_growth(
    observations: list[GrowthPoint], model: GrowthModel
) -> dict[str, object]:
    """Ajuste r et K sur au moins trois examens distincts.

    Formules : Gompertz et logistique décrivent une croissance initiale rapide
    qui ralentit à l'approche de la capacité limite K.
    """
    if model not in MODEL_FUNCTIONS:
        raise ValueError("Choisissez le modèle de Gompertz ou le modèle logistique.")
    if len(observations) < 3:
        raise ValueError("L'ajustement nécessite au moins trois examens.")

    days = np.asarray([point[0] for point in observations], dtype=np.float64)
    volumes = np.asarray([point[1] for point in observations], dtype=np.float64)
    if not np.all(np.isfinite(days)) or not np.all(np.isfinite(volumes)):
        raise ValueError("Les jours et volumes doivent être des nombres finis.")
    if np.any(days < 0) or np.any(volumes <= 0):
        raise ValueError(
            "Les jours doivent être positifs et les volumes strictement positifs."
        )
    if len(np.unique(days)) != len(days):
        raise ValueError("Chaque examen doit avoir un jour de suivi distinct.")

    order = np.argsort(days)
    days = days[order]
    volumes = volumes[order]
    relative_days = days - days[0]
    initial = float(volumes[0])
    maximum = float(volumes.max())
    function = MODEL_FUNCTIONS[model]
    fitted, covariance = curve_fit(
        lambda time, rate, capacity: function(time, rate, capacity, initial),
        relative_days,
        volumes,
        p0=(0.03, maximum * 2.0),
        bounds=([1e-8, maximum * 1.001], [10.0, maximum * 1e5]),
        maxfev=20_000,
    )
    rate, capacity = (float(value) for value in fitted)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    confidence_rate = (
        max(0.0, rate - 1.96 * float(standard_errors[0])),
        rate + 1.96 * float(standard_errors[0]),
    )
    confidence_capacity = (
        max(maximum, capacity - 1.96 * float(standard_errors[1])),
        capacity + 1.96 * float(standard_errors[1]),
    )
    curve_days = np.linspace(days[0], days[-1], 100)
    curve_volumes = function(curve_days - days[0], rate, capacity, initial)

    return {
        "model": model,
        "r": rate,
        "K": capacity,
        "ci_r": confidence_rate,
        "ci_K": confidence_capacity,
        "days": days.tolist(),
        "volumes": volumes.tolist(),
        "curve_days": curve_days.tolist(),
        "curve_volumes": curve_volumes.tolist(),
    }


def predict_volume(
    volume: float, rate: float, capacity: float, elapsed_days: float, model: str
) -> float:
    """Projette un volume sur une durée avec le modèle choisi."""
    if volume <= 0 or capacity <= 0 or rate < 0 or elapsed_days < 0:
        raise ValueError("Les paramètres de projection doivent être positifs.")
    if model not in MODEL_FUNCTIONS:
        raise ValueError("Modèle de croissance inconnu.")
    predicted = MODEL_FUNCTIONS[model](
        np.asarray([elapsed_days], dtype=np.float64),
        rate,
        capacity,
        volume,
    )
    return float(predicted[0])
