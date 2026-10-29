"""Vérifie le calcul et l'export des résultats BraTS sans données réelles."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from morphort.config import DEFAULT_SEGMENTATION
from pages.resultats_globaux import _summary_frame
from scripts import evaluate_brats


def test_evaluate_brats_writes_one_transparent_row_per_case(
    tmp_path, monkeypatch, capsys
) -> None:
    """Les scores sont conservés sans arrondi dans le CSV et le résumé."""
    expected = np.zeros((8, 8, 8), dtype=bool)
    expected[2:5, 2:5, 2:5] = True
    predicted = expected.copy()
    predicted[5, 4, 4] = True
    cases = {
        case_id: SimpleNamespace(
            volumes={"FLAIR": np.zeros((8, 8, 8))},
            spacing_mm=(1.0, 1.0, 1.0),
            expert_masks={"Tumeur entière": expected},
        )
        for case_id in ("BraTS20_Training_001", "BraTS20_Training_002")
    }

    monkeypatch.setattr(
        evaluate_brats,
        "list_brats_cases",
        lambda _directory: list(cases),
    )
    monkeypatch.setattr(
        evaluate_brats,
        "load_brats_case",
        lambda case_id, _directory: cases[case_id],
    )
    monkeypatch.setattr(
        evaluate_brats,
        "segment_tumor",
        lambda *_args, **_kwargs: (predicted, [], 0.0),
    )
    output_file = tmp_path / "results" / "brats_eval.csv"

    results = evaluate_brats.evaluate_brats(tmp_path, output_file)

    saved = pd.read_csv(output_file)
    assert list(saved["patient"]) == list(cases)
    assert list(saved.columns) == [
        "patient",
        "opening_radius_mm",
        "closing_radius_mm",
        "threshold_method",
        "dice",
        "jaccard",
        "hausdorff95_mm",
    ]
    assert saved["opening_radius_mm"].unique().tolist() == [
        DEFAULT_SEGMENTATION.opening_radius_mm
    ]
    assert saved["closing_radius_mm"].unique().tolist() == [
        DEFAULT_SEGMENTATION.closing_radius_mm
    ]
    assert saved["threshold_method"].unique().tolist() == [
        DEFAULT_SEGMENTATION.threshold_method
    ]
    assert saved["dice"].tolist() == results["dice"].tolist()
    assert "Patients évalués : 2" in capsys.readouterr().out


def test_global_summary_keeps_infinite_hausdorff_explicit() -> None:
    """Un Hausdorff infini est visible dans les statistiques, pas transformé en NaN."""
    summary = _summary_frame(
        pd.DataFrame(
            {
                "dice": [0.0, 0.5],
                "hausdorff95_mm": [float("inf"), 2.0],
            }
        )
    )

    assert summary.loc["Moyenne", "hausdorff95_mm"] == float("inf")
    assert summary.loc["Écart-type", "hausdorff95_mm"] == float("inf")
    assert summary.loc["Minimum", "hausdorff95_mm"] == 2.0
    assert summary.loc["Maximum", "hausdorff95_mm"] == float("inf")
