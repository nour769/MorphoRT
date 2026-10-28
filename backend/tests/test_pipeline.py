import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from starlette.requests import Request

import app.main as api
from app.pipeline.metrics import dice_coefficient, linear_quadratic_survival
from app.pipeline.mock import MockBackend


def test_dice_coefficient_for_identical_masks() -> None:
    mask = np.array([False, True, True], dtype=bool)
    assert dice_coefficient(mask, mask) == 1.0


def test_dice_coefficient_for_empty_masks() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    assert dice_coefficient(mask, mask) == 1.0


def test_dice_coefficient_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dice_coefficient(np.zeros((2, 2), dtype=bool), np.zeros((2, 3), dtype=bool))


def test_gompertz_fit_on_noise_free_data() -> None:
    days = np.linspace(0, 60, 8)
    expected_r = 0.08
    expected_capacity = 12.0
    initial_volume = 0.7
    volumes = expected_capacity * np.exp(
        np.log(initial_volume / expected_capacity) * np.exp(-expected_r * days)
    )
    result = MockBackend().fit_growth(
        [
            {"day": float(day), "volume_cm3": float(volume)}
            for day, volume in zip(days, volumes, strict=True)
        ],
        "gompertz",
    )
    assert result["r"] == pytest.approx(expected_r, rel=1e-4)
    assert result["K"] == pytest.approx(expected_capacity, rel=1e-4)
    assert len(result["curve"]) == 100
    assert result["ci_r"][0] <= result["r"] <= result["ci_r"][1]


def test_linear_quadratic_survival_matches_formula() -> None:
    assert linear_quadratic_survival(0.2, 0.03, 2.0) == pytest.approx(
        np.exp(-(0.2 * 2.0 + 0.03 * 2.0**2))
    )


def test_linear_quadratic_survival_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        linear_quadratic_survival(0.2, 0.03, -1.0)


def test_measure_returns_physical_units() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    result = MockBackend().measure(mask, (1.0, 1.0, 1.0))
    assert result["volume_cm3"] == pytest.approx(0.001)
    assert result["surface_cm2"] == pytest.approx(0.06)
    assert 0.0 < result["sphericity"] <= 1.0


def test_adaptive_simulation_differs_without_exceeding_bed_limit() -> None:
    result = MockBackend().simulate_rt(
        {"volume_cm3": 1.0, "r": 0.05, "K": 10.0},
        alpha=0.2,
        beta=0.03,
        bed_max=60.0,
        n_fractions=5,
    )
    assert result["fixed"][-1]["volume_cm3"] != pytest.approx(
        result["adaptive"][-1]["volume_cm3"]
    )
    assert result["fixed"][-1]["bed_gy"] == pytest.approx(60.0)
    assert result["adaptive"][-1]["bed_gy"] <= 60.0


def test_missing_validation_file_is_reported_without_fabricated_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "RESULTS_FILE", tmp_path / "missing-results.csv")
    result = api.validation()
    assert result["data_source"] == api.DATA_SOURCE
    assert result["available"] is False
    assert result["count"] == 0
    assert result["dice"] == []
    assert result["mean"] is None


def test_unexpected_api_error_includes_data_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("unexpected failure")
    response = asyncio.run(
        api.handle_unexpected_exception(Request({"type": "http"}), error)
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "data_source": api.DATA_SOURCE,
        "detail": "Une erreur interne est survenue.",
    }
    assert "Unhandled API exception" in caplog.text
