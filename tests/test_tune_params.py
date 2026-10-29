"""Vérifie la séparation des patients et le choix des paramètres."""

from types import SimpleNamespace

import numpy as np
import pytest

from morphort.segmentation import robust_normalize, segment_tumor
from scripts import tune_params
from scripts.tune_params import SegmentationParameters


def test_segment_tumor_supports_percentile_threshold_and_separate_radii() -> None:
    """Le percentile 90 et le rayon de fermeture sont rapportés explicitement."""
    volume = np.zeros((24, 24, 20), dtype=np.float64)
    volume[3:21, 3:21, 2:18] = np.linspace(
        1.0, 100.0, 18 * 18 * 16
    ).reshape((18, 18, 16))
    normalized, brain = robust_normalize(volume)
    expected_threshold = float(np.percentile(normalized[brain], 90))

    mask, steps, selected_threshold = segment_tumor(
        volume,
        (1.0, 1.0, 1.0),
        opening_radius_mm=0.5,
        closing_radius_mm=1.5,
        threshold_method="percentile_90",
    )

    assert selected_threshold == pytest.approx(expected_threshold)
    assert steps[1].name == "Seuil percentile 90"
    assert "1.50 mm" in steps[3].explanation
    assert "P_{90}" in steps[1].formula
    fast_mask, fast_steps, fast_threshold = segment_tumor(
        volume,
        (1.0, 1.0, 1.0),
        opening_radius_mm=0.5,
        closing_radius_mm=1.5,
        threshold_method="percentile_90",
        include_diagnostic_steps=False,
    )
    assert np.array_equal(mask, fast_mask)
    assert fast_steps == []
    assert fast_threshold == selected_threshold


def test_tuning_selects_parameters_only_on_ten_alternating_cases(
    tmp_path, monkeypatch, capsys
) -> None:
    """Les dix patients impairs sont réservés au test après sélection."""
    case_ids = [f"BraTS20_Training_{index:03d}" for index in range(1, 21)]
    expected = np.zeros((5, 5, 5), dtype=bool)
    expected[2, 2, 2] = True
    cases = {
        case_id: SimpleNamespace(
            volumes={"FLAIR": np.full((5, 5, 5), index, dtype=np.float64)},
            spacing_mm=(1.0, 1.0, 1.0),
            expert_masks={"Tumeur entière": expected},
        )
        for index, case_id in enumerate(case_ids, start=1)
    }
    configurations = (
        SegmentationParameters(0.5, 0.5, "otsu"),
        SegmentationParameters(0.5, 0.5, "percentile_90"),
        SegmentationParameters(1.0, 1.0, "otsu"),
        SegmentationParameters(1.0, 1.0, "percentile_90"),
    )
    evaluated_patients: list[tuple[str, float, str]] = []

    monkeypatch.setattr(tune_params, "list_brats_cases", lambda _path: case_ids)
    monkeypatch.setattr(tune_params, "_build_grid", lambda: configurations)
    monkeypatch.setattr(
        tune_params,
        "load_brats_case",
        lambda case_id, _path: cases[case_id],
    )

    def fake_segment(_flair, _spacing, **parameters):
        radius = parameters["opening_radius_mm"]
        method = parameters["threshold_method"]
        # Le meilleur candidat est connu, afin de vérifier le choix réel.
        predicted = (
            expected.copy()
            if radius == 1.0 and method == "otsu"
            else np.zeros_like(expected)
        )
        return predicted, [], 0.0

    monkeypatch.setattr(tune_params, "segment_tumor", fake_segment)

    def record_evaluation(case_id, parameters, data_directory, split):
        evaluated_patients.append((case_id, parameters.opening_radius_mm, split))
        return original_evaluate_case(case_id, parameters, data_directory, split)

    original_evaluate_case = tune_params._evaluate_case
    monkeypatch.setattr(tune_params, "_evaluate_case", record_evaluation)
    grid, patient_results = tune_params.tune_brats(tmp_path, tmp_path / "results")

    assert grid.iloc[0]["opening_radius_mm"] == 1.0
    assert grid.iloc[0]["threshold_method"] == "otsu"
    assert patient_results["split"].value_counts().to_dict() == {
        "réglage": 10,
        "test_indépendant": 10,
    }
    tuning_ids, held_out_ids = tune_params._split_cases(case_ids)
    assert set(tuning_ids).isdisjoint(held_out_ids)
    assert set(
        patient_results.loc[
            patient_results["split"] == "réglage", "patient"
        ]
    ) == set(tuning_ids)
    assert set(
        patient_results.loc[
            patient_results["split"] == "test_indépendant", "patient"
        ]
    ) == set(held_out_ids)
    test_calls = [row for row in evaluated_patients if row[2] == "test_indépendant"]
    assert len(test_calls) == 10
    assert all(row[1] == 1.0 for row in test_calls)
    assert (tmp_path / "results" / "brats_tuning_grid.csv").is_file()
    assert (tmp_path / "results" / "brats_tuning_patients.csv").is_file()
    assert "test indépendant" in capsys.readouterr().out.lower()


def test_tuning_refuses_non_twenty_case_splits() -> None:
    """Le script ne présente pas un découpage incomplet comme un test 10/10."""
    with pytest.raises(ValueError, match="exactement 20"):
        tune_params._split_cases(["BraTS20_Training_001"] * 2)
