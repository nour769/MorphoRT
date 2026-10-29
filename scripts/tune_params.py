"""Règle les paramètres morphologiques sur 10 cas et teste sur 10 autres."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from morphort.datasets import list_brats_cases, load_brats_case
from morphort.measurements import (
    dice_coefficient,
    hausdorff_distance_95_mm,
    jaccard_index,
)
from morphort.segmentation import (
    EmptySegmentationError,
    ThresholdMethod,
    segment_tumor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results"
OPENING_RADII_MM = (0.5, 1.0)
CLOSING_RADII_MM = (0.5, 1.0)
THRESHOLD_METHODS: tuple[ThresholdMethod, ...] = ("otsu", "percentile_90")


@dataclass(frozen=True)
class SegmentationParameters:
    """Un triplet de paramètres morphologiques à comparer."""

    opening_radius_mm: float
    closing_radius_mm: float
    threshold_method: ThresholdMethod


def _build_grid() -> tuple[SegmentationParameters, ...]:
    """Construit une grille cartésienne de 8 combinaisons prédéfinies."""
    return tuple(
        SegmentationParameters(opening_radius, closing_radius, threshold_method)
        for opening_radius in OPENING_RADII_MM
        for closing_radius in CLOSING_RADII_MM
        for threshold_method in THRESHOLD_METHODS
    )


def _split_cases(case_ids: list[str]) -> tuple[list[str], list[str]]:
    """Sépare les 20 cas triés en deux groupes alternés de 10 patients."""
    ordered = sorted(case_ids)
    if len(ordered) != 20 or len(set(ordered)) != 20:
        raise ValueError(
            "Le réglage défendable nécessite exactement 20 patients complets "
            "et distincts (10 réglage + 10 test indépendant) ; "
            f"{len(ordered)} entrées trouvées."
        )
    tuning_cases = ordered[::2]
    held_out_cases = ordered[1::2]
    if set(tuning_cases) & set(held_out_cases):
        raise RuntimeError("Les groupes de réglage et de test se chevauchent.")
    return tuning_cases, held_out_cases


def _evaluate_case(
    case_id: str,
    parameters: SegmentationParameters,
    data_directory: Path,
    split: str,
) -> dict[str, str | float]:
    """Calcule les métriques d'un patient et signale les segmentations vides."""
    case = load_brats_case(case_id, data_directory)
    expert = case.expert_masks["Tumeur entière"]
    status = "ok"
    try:
        predicted, _steps, _threshold = segment_tumor(
            case.volumes["FLAIR"],
            case.spacing_mm,
            opening_radius_mm=parameters.opening_radius_mm,
            closing_radius_mm=parameters.closing_radius_mm,
            threshold_method=parameters.threshold_method,
            include_diagnostic_steps=False,
        )
    except EmptySegmentationError as error:
        print(f"  Échec explicite pour {case_id} : {error}", flush=True)
        predicted = np.zeros(expert.shape, dtype=np.bool_)
        status = "masque vide après morphologie"

    return {
        "split": split,
        "patient": case_id,
        "opening_radius_mm": parameters.opening_radius_mm,
        "closing_radius_mm": parameters.closing_radius_mm,
        "threshold_method": parameters.threshold_method,
        "dice": dice_coefficient(predicted, expert),
        "jaccard": jaccard_index(predicted, expert),
        "hausdorff95_mm": hausdorff_distance_95_mm(
            predicted, expert, case.spacing_mm
        ),
        "status": status,
    }


def _score_summary(rows: list[dict[str, str | float]]) -> dict[str, float]:
    """Résume les trois métriques sans arrondir les observations individuelles."""
    scores = pd.DataFrame(rows)[["dice", "jaccard", "hausdorff95_mm"]]
    summary: dict[str, float] = {}
    for metric in scores.columns:
        values = scores[metric].to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError(f"La métrique {metric} contient une valeur NaN.")
        if np.isposinf(values).any():
            summary[f"{metric}_mean"] = float("inf")
            summary[f"{metric}_std"] = float("inf")
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float("inf")
            continue
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))
    return summary


def _print_summary(
    label: str, rows: list[dict[str, str | float]], parameters: SegmentationParameters
) -> None:
    """Affiche les résultats et paramètres d'un seul groupe."""
    print(f"\n{label} — {len(rows)} patients")
    print(
        "Paramètres : "
        f"ouverture={parameters.opening_radius_mm:g} mm, "
        f"fermeture={parameters.closing_radius_mm:g} mm, "
        f"seuil={parameters.threshold_method}"
    )
    summary = _score_summary(rows)
    for metric, label in (
        ("dice", "Dice"),
        ("jaccard", "Jaccard"),
        ("hausdorff95_mm", "Hausdorff 95 (mm)"),
    ):
        print(
            f"{label} : moyenne={summary[f'{metric}_mean']:.4f}, "
            f"écart-type={summary[f'{metric}_std']:.4f}, "
            f"minimum={summary[f'{metric}_min']:.4f}, "
            f"maximum={summary[f'{metric}_max']:.4f}"
        )


def tune_brats(
    data_directory: Path,
    output_directory: Path = DEFAULT_RESULTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choisit la grille sur 10 cas, puis évalue une fois sur 10 cas tenus à part."""
    tuning_cases, held_out_cases = _split_cases(list_brats_cases(data_directory))
    grid = _build_grid()
    if not grid:
        raise ValueError("La grille de paramètres est vide.")

    tuning_by_configuration: dict[
        SegmentationParameters, list[dict[str, str | float]]
    ] = {}
    grid_rows: list[dict[str, str | float]] = []
    for config_index, parameters in enumerate(grid, start=1):
        print(
            f"\nGrille {config_index}/{len(grid)} : "
            f"ouverture={parameters.opening_radius_mm:g} mm, "
            f"fermeture={parameters.closing_radius_mm:g} mm, "
            f"seuil={parameters.threshold_method}",
            flush=True,
        )
        patient_rows = []
        for patient_index, case_id in enumerate(tuning_cases, start=1):
            print(
                f"  Réglage {patient_index}/10 : {case_id}",
                flush=True,
            )
            patient_rows.append(
                _evaluate_case(case_id, parameters, data_directory, "réglage")
            )
        tuning_by_configuration[parameters] = patient_rows
        grid_rows.append(
            {
                "opening_radius_mm": parameters.opening_radius_mm,
                "closing_radius_mm": parameters.closing_radius_mm,
                "threshold_method": parameters.threshold_method,
                **_score_summary(patient_rows),
            }
        )

    # Le tri secondaire rend les égalités déterministes et auditables.
    ranked_grid = sorted(
        zip(grid, grid_rows, strict=True),
        key=lambda item: (
            -float(item[1]["dice_mean"]),
            item[0].opening_radius_mm,
            item[0].closing_radius_mm,
            item[0].threshold_method,
        ),
    )
    selected_parameters, _best_grid_row = ranked_grid[0]
    selected_tuning_rows = tuning_by_configuration[selected_parameters]
    held_out_rows = []
    for patient_index, case_id in enumerate(held_out_cases, start=1):
        print(
            f"  Test indépendant {patient_index}/10 : {case_id}",
            flush=True,
        )
        held_out_rows.append(
            _evaluate_case(
                case_id, selected_parameters, data_directory, "test_indépendant"
            )
        )

    grid_results = pd.DataFrame(grid_rows).sort_values(
        ["dice_mean", "opening_radius_mm", "closing_radius_mm", "threshold_method"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    patient_results = pd.DataFrame([*selected_tuning_rows, *held_out_rows])
    output_directory.mkdir(parents=True, exist_ok=True)
    grid_path = output_directory / "brats_tuning_grid.csv"
    patient_path = output_directory / "brats_tuning_patients.csv"
    grid_results.to_csv(grid_path, index=False)
    patient_results.to_csv(patient_path, index=False)

    _print_summary(
        "Groupe de réglage — résultats après sélection",
        selected_tuning_rows,
        selected_parameters,
    )
    _print_summary(
        "Test indépendant — paramètres figés",
        held_out_rows,
        selected_parameters,
    )
    print(f"\nGrille complète : {grid_path}")
    print(f"Résultats patients : {patient_path}")
    print(
        "Les cas de test indépendant n'ont pas contribué à la sélection "
        "des paramètres."
    )
    return grid_results, patient_results


def main() -> None:
    """Lance la recherche de paramètres et l'évaluation indépendante."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "brats",
        help="Dossier contenant exactement 20 patients BraTS complets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Dossier des CSV de grille et des scores séparés.",
    )
    arguments = parser.parse_args()
    tune_brats(arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
