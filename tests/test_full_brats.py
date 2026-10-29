"""Vérifie le split fixe et le bootstrap de l'évaluation complète."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_full_brats import (
    REGION_COLUMNS,
    _bootstrap_mean_interval,
    _create_or_validate_split,
    _summarize_test_results,
)


def test_split_is_deterministic_persisted_and_disjoint(tmp_path) -> None:
    """La graine persistée fixe une séparation répétable des patients."""
    cases = [f"BraTS20_Training_{index:03d}" for index in range(1, 21)]
    split_file = tmp_path / "split.json"

    split = _create_or_validate_split(cases, split_file, seed=2026, tuning_count=6)
    loaded = _create_or_validate_split(
        list(reversed(cases)), split_file, seed=2026, tuning_count=6
    )

    assert split == loaded
    assert len(split["tuning_cases"]) == 6
    assert len(split["test_cases"]) == 14
    assert set(split["tuning_cases"]).isdisjoint(split["test_cases"])
    with pytest.raises(ValueError, match="ne correspond plus"):
        _create_or_validate_split(cases, split_file, seed=42, tuning_count=6)


def test_bootstrap_mean_interval_is_reproducible_and_contains_mean() -> None:
    """L'intervalle est déterministe pour une graine fixée."""
    values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    first = _bootstrap_mean_interval(values, seed=2026, replicates=1000)
    second = _bootstrap_mean_interval(values, seed=2026, replicates=1000)

    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_test_summary_includes_all_tumor_regions_and_low_scores() -> None:
    """Le résumé conserve la région entière et chaque sous-région."""
    frame = pd.DataFrame(
        {
            column: [0.0, 0.5, 1.0]
            for column in REGION_COLUMNS.values()
        }
    )

    summary = _summarize_test_results(frame, seed=7, replicates=500)

    assert summary["region"].tolist() == list(REGION_COLUMNS)
    assert summary["patient_count"].tolist() == [3, 3, 3, 3]
    assert (summary["mean"] == 0.5).all()
    assert (summary["min"] == 0.0).all()
    assert (summary["max"] == 1.0).all()
    assert (
        summary["bootstrap_mean_ci95_low"]
        <= summary["bootstrap_mean_ci95_high"]
    ).all()
