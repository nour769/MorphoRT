"""Vérifie le découpage morphologique des sous-régions sur volumes factices."""

import numpy as np
import pytest

from morphort.measurements import dice_coefficient
from morphort.subregions import EDEMA, ENHANCING, NECROTIC, segment_tumor_subregions


def _synthetic_modalities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crée un cerveau bruité avec une tumeur et trois régions connues."""
    random = np.random.default_rng(42)
    shape = (32, 32, 32)
    t1ce = np.zeros(shape, dtype=np.float64)
    flair = np.zeros(shape, dtype=np.float64)
    brain = np.zeros(shape, dtype=bool)
    brain[3:29, 3:29, 3:29] = True
    t1ce[brain] = random.normal(10.0, 1.0, int(brain.sum()))
    flair[brain] = random.normal(12.0, 1.5, int(brain.sum()))

    whole = np.zeros(shape, dtype=bool)
    whole[8:24, 8:24, 8:24] = True
    enhancing = np.zeros(shape, dtype=bool)
    enhancing[10:22, 10:22, 10:22] = True
    necrotic = np.zeros(shape, dtype=bool)
    necrotic[14:18, 14:18, 14:18] = True

    t1ce[enhancing] = 30.0
    t1ce[necrotic] = 2.0
    flair[whole] = 35.0
    return t1ce, flair, whole


def test_subregion_pipeline_returns_disjoint_masks_and_explanatory_steps() -> None:
    """Les trois sous-régions sont confinées à la tumeur et non chevauchantes."""
    t1ce, flair, whole = _synthetic_modalities()

    predicted, steps = segment_tumor_subregions(
        t1ce,
        flair,
        whole,
        (1.0, 1.0, 1.0),
        opening_radius_mm=1.0,
        closing_radius_mm=1.0,
    )

    assert set(predicted) == {ENHANCING, NECROTIC, EDEMA}
    for mask in predicted.values():
        assert mask.dtype == np.bool_
        assert mask.shape == whole.shape
        assert np.all(mask <= whole)
    assert not np.any(predicted[ENHANCING] & predicted[NECROTIC])
    assert not np.any(predicted[ENHANCING] & predicted[EDEMA])
    assert not np.any(predicted[NECROTIC] & predicted[EDEMA])
    assert predicted[ENHANCING].any()
    assert predicted[NECROTIC].any()
    assert predicted[EDEMA].any()
    assert len(steps[ENHANCING]) == 4
    assert len(steps[NECROTIC]) == 3
    assert len(steps[EDEMA]) == 3
    assert "Otsu" in steps[ENHANCING][1].explanation
    assert "T1ce" in steps[NECROTIC][2].explanation
    assert "reconstruction" in steps[EDEMA][-1].name.lower()


def test_subregion_intensity_masks_match_known_synthetic_regions() -> None:
    """Un cas simple permet de mesurer les recouvrements sans modèle appris."""
    t1ce, flair, whole = _synthetic_modalities()
    expected_enhancing = np.zeros_like(whole)
    expected_enhancing[10:22, 10:22, 10:22] = True
    expected_necrotic = np.zeros_like(whole)
    expected_necrotic[14:18, 14:18, 14:18] = True

    predicted, _steps = segment_tumor_subregions(
        t1ce, flair, whole, (1.0, 1.0, 1.0)
    )

    assert dice_coefficient(predicted[ENHANCING], expected_enhancing) > 0.8
    assert dice_coefficient(predicted[NECROTIC], expected_necrotic) > 0.7


def test_subregion_pipeline_rejects_empty_or_incompatible_inputs() -> None:
    """Les erreurs de forme et les paramètres physiques invalides sont visibles."""
    volume = np.ones((8, 8, 8), dtype=np.float64)
    empty_mask = np.zeros(volume.shape, dtype=bool)
    with pytest.raises(ValueError, match="vide"):
        segment_tumor_subregions(volume, volume, empty_mask, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="même forme"):
        segment_tumor_subregions(
            volume, volume[:, :, :-1], ~empty_mask, (1.0, 1.0, 1.0)
        )
    with pytest.raises(ValueError, match="positif"):
        segment_tumor_subregions(
            volume,
            volume,
            np.ones(volume.shape, dtype=bool),
            (1.0, 1.0, 1.0),
            opening_radius_mm=0.0,
        )
