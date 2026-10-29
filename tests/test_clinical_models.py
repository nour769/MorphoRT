"""Vérifie segmentation, mesures, croissance et modèle de traitement simulé."""

import numpy as np
import pytest

from morphort.growth import (
    fit_growth,
    gompertz_volume,
    logistic_volume,
    simulate_gompertz_followup,
)
from morphort.measurements import (
    dice_coefficient,
    hausdorff_distance_95_mm,
    hausdorff_distance_mm,
    jaccard_index,
    measure_mask,
)
from morphort.radiotherapy import (
    biologically_effective_dose,
    linear_quadratic_survival,
    optimize_fraction_doses,
    simulate_plans,
)
from morphort.segmentation import otsu_threshold, segment_tumor
from morphort.synthetic import create_synthetic_mri


@pytest.mark.parametrize("shape", [(32, 32, 24), (28, 30, 32)])
def test_synthetic_segmentation_retains_tumor(shape: tuple[int, int, int]) -> None:
    """Le pipeline explicable retrouve la tumeur connue de la démo."""
    modalities, expert = create_synthetic_mri(shape, seed=17)
    spacing = (1.0, 1.0, 1.5)

    predicted, steps, threshold = segment_tumor(modalities["FLAIR"], spacing)

    assert len(steps) == 6
    assert np.isfinite(threshold)
    assert all(step.image.shape == shape for step in steps)
    assert dice_coefficient(predicted, expert) >= 0.75


def test_otsu_separates_two_intensity_classes() -> None:
    """Le seuil fait-main sépare deux groupes clairement distincts."""
    values = np.concatenate([np.full(200, 1.0), np.full(200, 8.0)])
    threshold = otsu_threshold(values)
    assert 1.0 < threshold < 8.0


def test_physical_measurements_and_dice() -> None:
    """Le volume d'un cube suit le produit des dimensions de ses voxels."""
    cube = np.zeros((8, 8, 8), dtype=bool)
    cube[2:4, 2:4, 2:4] = True
    result = measure_mask(cube, (1.0, 2.0, 3.0))

    assert result["volume_ml"] == pytest.approx(0.048)
    assert result["surface_cm2"] == pytest.approx(0.88)
    assert 0 < result["sphericity"] <= 1
    assert result["compactness"] >= 1
    assert dice_coefficient(cube, cube.copy()) == 1.0
    assert dice_coefficient(cube, np.zeros_like(cube)) == 0.0
    assert jaccard_index(cube, cube.copy()) == 1.0
    assert jaccard_index(cube, np.zeros_like(cube)) == 0.0
    assert hausdorff_distance_mm(cube, cube.copy(), (1.0, 2.0, 3.0)) == 0.0
    assert hausdorff_distance_mm(
        cube, np.zeros_like(cube), (1.0, 2.0, 3.0)
    ) == float("inf")
    assert hausdorff_distance_95_mm(
        cube, cube.copy(), (1.0, 2.0, 3.0)
    ) == 0.0
    assert hausdorff_distance_95_mm(
        cube, np.zeros_like(cube), (1.0, 2.0, 3.0)
    ) == float("inf")


def test_hausdorff95_is_less_affected_by_one_distant_outlier() -> None:
    """Un voxel isolé augmente le Hausdorff maximal mais pas le percentile 95."""
    expert = np.zeros((20, 20, 20), dtype=bool)
    expert[5:13, 5:13, 5:13] = True
    predicted = expert.copy()
    predicted[1, 1, 1] = True

    maximum = hausdorff_distance_mm(predicted, expert, (1.0, 1.0, 1.0))
    percentile = hausdorff_distance_95_mm(predicted, expert, (1.0, 1.0, 1.0))

    assert maximum > 5.0
    assert percentile < maximum


def test_growth_models_fit_data_and_refuse_too_few_exams() -> None:
    """Les ajustements retrouvent les deux courbes et refusent deux examens."""
    days = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    initial = 0.2
    gompertz_volumes = gompertz_volume(days, 0.045, 2.5, initial)
    logistic_volumes = logistic_volume(days, 0.035, 2.2, initial)
    gompertz_points = list(zip(days.tolist(), gompertz_volumes.tolist(), strict=True))
    logistic_points = list(zip(days.tolist(), logistic_volumes.tolist(), strict=True))

    gompertz_fit = fit_growth(gompertz_points, "Gompertz")
    logistic_fit = fit_growth(logistic_points, "Logistique")
    assert gompertz_fit["r"] == pytest.approx(0.045, rel=0.01)
    assert gompertz_fit["K"] == pytest.approx(2.5, rel=0.01)
    assert logistic_fit["r"] == pytest.approx(0.035, rel=0.01)
    assert logistic_fit["K"] == pytest.approx(2.2, rel=0.01)

    with pytest.raises(ValueError, match="trois examens"):
        fit_growth(gompertz_points[:2], "Gompertz")


def test_simulated_gompertz_followup_starts_at_real_volume() -> None:
    """Le suivi simulé conserve V0 et permet de retrouver r et K sans bruit."""
    initial = 0.4
    points = simulate_gompertz_followup(
        initial, 0.03, 1.8, [0.0, 30.0, 60.0, 90.0, 120.0], 0.0
    )
    fitted = fit_growth(points, "Gompertz")

    assert points[0] == (0.0, initial)
    assert fitted["r"] == pytest.approx(0.03, rel=0.01)
    assert fitted["K"] == pytest.approx(1.8, rel=0.01)

    with pytest.raises(ValueError, match="trois jours"):
        simulate_gompertz_followup(initial, 0.03, 1.8, [0.0, 30.0], 0.0)


def test_lq_and_bed_formulas() -> None:
    """Les calculs de survie et BED reproduisent directement leurs formules."""
    assert linear_quadratic_survival(2.0, 0.1, 0.03) == pytest.approx(
        np.exp(-(0.1 * 2.0 + 0.03 * 4.0))
    )
    assert biologically_effective_dose([2.0, 2.0], 3.0) == pytest.approx(
        2 * 2.0 * (1.0 + 2.0 / 3.0)
    )


def test_optimized_dose_satisfies_healthy_bed_constraint() -> None:
    """SLSQP garde les doses dans leurs bornes et sous la BED autorisée."""
    doses = optimize_fraction_doses(0.2, 0.04, 2.0, "Gompertz", 0.1, 0.03, 3.0, 45.0, 8)

    assert len(doses) == 8
    assert np.all((doses >= 0.0) & (doses <= 5.0))
    assert biologically_effective_dose(doses, 3.0) <= 45.0 + 1e-5


def test_fixed_and_adaptive_plans_are_computed_from_observations() -> None:
    """Les plans ont le même nombre de fractions et respectent la BED."""
    days = [0.0, 10.0, 20.0, 30.0, 40.0]
    volumes = [0.2, 0.31, 0.48, 0.68, 0.91]
    observations = list(zip(days, volumes, strict=True))

    results = simulate_plans(
        observations,
        "Gompertz",
        0.1,
        0.03,
        3.0,
        45.0,
        8,
    )

    assert len(results["fixed"]) == 8
    assert len(results["adaptive"]) == 8
    assert results["fixed_bed_gy"] <= 45.0 + 1e-5
    assert results["adaptive_bed_gy"] <= 45.0 + 1e-5
