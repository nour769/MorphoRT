"""Règle sur BraTS2020 entraînement et évalue uniquement le groupe test."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from morphort.config import (
    BRATS_BOOTSTRAP_REPLICATES,
    BRATS_EXPECTED_CASE_DIRECTORIES,
    BRATS_EXPECTED_LABELED_CASES,
    BRATS_RAW_DIRECTORY,
    BRATS_SPLIT_SEED,
    BRATS_TUNING_CASES,
    CLOSING_RADII_MM,
    OPENING_RADII_MM,
    THRESHOLD_METHODS,
    SegmentationParameters,
)
from morphort.datasets import list_brats_case_inventory, load_brats_case
from morphort.measurements import (
    dice_coefficient,
    hausdorff_distance_95_mm,
)
from morphort.segmentation import EmptySegmentationError, segment_tumor
from morphort.subregions import (
    EDEMA,
    ENHANCING,
    NECROTIC,
    segment_tumor_subregions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "results"
SPLIT_FILE = DEFAULT_RESULTS_DIRECTORY / "split.json"
WHOLE_TUMOR = "Tumeur entière"
REGION_COLUMNS = {
    WHOLE_TUMOR: "dice_whole_tumor",
    ENHANCING: "dice_enhancing",
    NECROTIC: "dice_necrotic",
    EDEMA: "dice_edema",
}


def _build_grid() -> tuple[SegmentationParameters, ...]:
    """Construit les huit combinaisons morphologiques pré-définies."""
    return tuple(
        SegmentationParameters(opening, closing, method)
        for opening in OPENING_RADII_MM
        for closing in CLOSING_RADII_MM
        for method in THRESHOLD_METHODS
    )


def _create_or_validate_split(
    case_ids: list[str],
    split_file: Path,
    seed: int,
    tuning_count: int,
    excluded_cases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Écrit une partition fixe ou refuse toute réinterprétation silencieuse."""
    ordered = sorted(set(case_ids))
    if len(ordered) != len(case_ids):
        raise ValueError("La liste BraTS contient des identifiants dupliqués.")
    if not 0 < tuning_count < len(ordered):
        raise ValueError("Le groupe de réglage doit être plus petit que le dataset.")

    if split_file.is_file():
        existing = json.loads(split_file.read_text(encoding="utf-8"))
        saved_exclusions = existing.get("excluded_cases", {})
        if (
            existing.get("seed") != seed
            or existing.get("tuning_count") != tuning_count
            or existing.get("all_cases") != ordered
            or set(existing.get("tuning_cases", []))
            | set(existing.get("test_cases", []))
            != set(ordered)
            or set(existing.get("tuning_cases", []))
            & set(existing.get("test_cases", []))
            or saved_exclusions != (excluded_cases or {})
        ):
            raise ValueError(
                f"Le split existant {split_file} ne correspond plus au dataset, "
                "à la graine ou à la taille de réglage. Vérifiez-le explicitement."
            )
        return existing

    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    split = {
        "seed": seed,
        "case_count": len(ordered),
        "tuning_count": tuning_count,
        "all_cases": ordered,
        "excluded_cases": excluded_cases or {},
        "tuning_cases": sorted(shuffled[:tuning_count]),
        "test_cases": sorted(shuffled[tuning_count:]),
    }
    if set(split["tuning_cases"]) & set(split["test_cases"]):
        raise RuntimeError("Les groupes de réglage et de test se chevauchent.")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return split


def _segment(
    flair: np.ndarray,
    spacing_mm: tuple[float, float, float],
    parameters: SegmentationParameters,
) -> tuple[np.ndarray, str]:
    """Retourne la tumeur entière et documente les masques devenus vides."""
    try:
        prediction, _, _ = segment_tumor(
            flair,
            spacing_mm,
            opening_radius_mm=parameters.opening_radius_mm,
            closing_radius_mm=parameters.closing_radius_mm,
            threshold_method=parameters.threshold_method,
            include_diagnostic_steps=False,
        )
    except EmptySegmentationError:
        return np.zeros(flair.shape, dtype=np.bool_), "masque vide après morphologie"
    return prediction, "ok"


def _dice_for_case(
    case_id: str,
    data_directory: Path,
    parameters: SegmentationParameters,
) -> float:
    """Calcule le Dice entier pour sélectionner les paramètres de réglage."""
    case = load_brats_case(case_id, data_directory)
    prediction, _status = _segment(case.volumes["FLAIR"], case.spacing_mm, parameters)
    return dice_coefficient(prediction, case.expert_masks[WHOLE_TUMOR])


def _tune_on_cases(
    case_ids: list[str], data_directory: Path
) -> tuple[SegmentationParameters, pd.DataFrame]:
    """Compare la grille uniquement sur les patients explicitement fournis."""
    grid = _build_grid()
    scores_by_parameters: dict[SegmentationParameters, list[float]] = {
        parameters: [] for parameters in grid
    }
    for patient_index, case_id in enumerate(case_ids, start=1):
        print(
            f"Réglage {patient_index}/{len(case_ids)} : {case_id}",
            flush=True,
        )
        case = load_brats_case(case_id, data_directory)
        expert = case.expert_masks[WHOLE_TUMOR]
        for parameters in grid:
            prediction, _status = _segment(
                case.volumes["FLAIR"], case.spacing_mm, parameters
            )
            scores_by_parameters[parameters].append(
                dice_coefficient(prediction, expert)
            )

    rows = []
    for parameters, scores in scores_by_parameters.items():
        values = np.asarray(scores, dtype=np.float64)
        rows.append(
            {
                "opening_radius_mm": parameters.opening_radius_mm,
                "closing_radius_mm": parameters.closing_radius_mm,
                "threshold_method": parameters.threshold_method,
                "training_dice_mean": float(np.mean(values)),
                "training_dice_std": float(np.std(values, ddof=1)),
                "training_dice_median": float(np.median(values)),
                "training_dice_min": float(np.min(values)),
                "training_dice_max": float(np.max(values)),
            }
        )
    grid_frame = pd.DataFrame(rows).sort_values(
        [
            "training_dice_mean",
            "opening_radius_mm",
            "closing_radius_mm",
            "threshold_method",
        ],
        ascending=[False, True, True, True],
        kind="stable",
    )
    best = grid_frame.iloc[0]
    selected = next(
        parameters
        for parameters in grid
        if parameters.opening_radius_mm == float(best["opening_radius_mm"])
        and parameters.closing_radius_mm == float(best["closing_radius_mm"])
        and parameters.threshold_method == best["threshold_method"]
    )
    return selected, grid_frame


def _evaluate_test_case(
    case_id: str,
    data_directory: Path,
    parameters: SegmentationParameters,
) -> dict[str, str | float]:
    """Calcule les métriques du test, y compris les quatre Dice par région."""
    case = load_brats_case(case_id, data_directory)
    whole_prediction, status = _segment(
        case.volumes["FLAIR"], case.spacing_mm, parameters
    )
    row: dict[str, str | float] = {
        "patient": case_id,
        "split": "test",
        "opening_radius_mm": parameters.opening_radius_mm,
        "closing_radius_mm": parameters.closing_radius_mm,
        "threshold_method": parameters.threshold_method,
        "status": status,
        REGION_COLUMNS[WHOLE_TUMOR]: dice_coefficient(
            whole_prediction, case.expert_masks[WHOLE_TUMOR]
        ),
        "hausdorff95_whole_mm": hausdorff_distance_95_mm(
            whole_prediction,
            case.expert_masks[WHOLE_TUMOR],
            case.spacing_mm,
        ),
    }
    if whole_prediction.any():
        predicted_regions, _steps = segment_tumor_subregions(
            case.volumes["T1CE"],
            case.volumes["FLAIR"],
            whole_prediction,
            case.spacing_mm,
            opening_radius_mm=parameters.opening_radius_mm,
            closing_radius_mm=parameters.closing_radius_mm,
        )
    else:
        predicted_regions = {
            ENHANCING: np.zeros(whole_prediction.shape, dtype=np.bool_),
            NECROTIC: np.zeros(whole_prediction.shape, dtype=np.bool_),
            EDEMA: np.zeros(whole_prediction.shape, dtype=np.bool_),
        }
    for region in (ENHANCING, NECROTIC, EDEMA):
        row[REGION_COLUMNS[region]] = dice_coefficient(
            predicted_regions[region], case.expert_masks[region]
        )
    return row


def _bootstrap_mean_interval(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    """Calcule l'IC percentile à 95 % de la moyenne par bootstrap patients."""
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Le bootstrap exige des Dice finis et non vides.")
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    batch_size = max(1, min(256, replicates))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = generator.integers(
            0, values.size, size=(stop - start, values.size)
        )
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(lower), float(upper)


def _summarize_test_results(
    results: pd.DataFrame,
    seed: int,
    replicates: int,
) -> pd.DataFrame:
    """Résume les quatre distributions sans exclure les Dice faibles."""
    rows = []
    for index, (region, column) in enumerate(REGION_COLUMNS.items()):
        values = results[column].to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError(f"{column} contient des valeurs NaN.")
        lower, upper = _bootstrap_mean_interval(
            values, seed + index, replicates
        )
        rows.append(
            {
                "region": region,
                "patient_count": int(values.size),
                "mean": float(np.mean(values)),
                "std_sample": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "bootstrap_mean_ci95_low": lower,
                "bootstrap_mean_ci95_high": upper,
            }
        )
    return pd.DataFrame(rows)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Écrit un JSON lisible de façon atomique."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_full_evaluation(
    data_directory: Path = BRATS_RAW_DIRECTORY,
    results_directory: Path = DEFAULT_RESULTS_DIRECTORY,
    *,
    seed: int = BRATS_SPLIT_SEED,
    tuning_count: int = BRATS_TUNING_CASES,
    bootstrap_replicates: int = BRATS_BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Règle sur le split fixe, puis écrit les mesures des patients test."""
    cases, incomplete_cases = list_brats_case_inventory(data_directory)
    directory_count = len(cases) + len(incomplete_cases)
    if len(cases) != BRATS_EXPECTED_LABELED_CASES or (
        directory_count != BRATS_EXPECTED_CASE_DIRECTORIES
    ):
        raise ValueError(
            f"{BRATS_EXPECTED_CASE_DIRECTORIES} dossiers et "
            f"{BRATS_EXPECTED_LABELED_CASES} cas complets étiquetés étaient "
            f"attendus dans {data_directory} ; {directory_count} dossiers et "
            f"{len(cases)} cas complets trouvés."
        )
    if any("seg" not in missing for missing in incomplete_cases.values()):
        raise ValueError(
            "Des cas BraTS incomplets autres que les masques experts absents "
            f"ont été trouvés : {incomplete_cases}."
        )
    if incomplete_cases:
        print(
            "Patients exclus des métriques faute de masque expert : "
            f"{incomplete_cases}",
            flush=True,
        )
    split_file = results_directory / "split.json"
    split = _create_or_validate_split(
        cases,
        split_file,
        seed,
        tuning_count,
        excluded_cases=incomplete_cases,
    )
    tuning_cases = split["tuning_cases"]
    test_cases = split["test_cases"]
    if (
        len(tuning_cases) != tuning_count
        or len(test_cases) != len(cases) - tuning_count
    ):
        raise ValueError("Les tailles des groupes dans split.json sont invalides.")

    print(
        f"Split fixe : {len(tuning_cases)} réglage / {len(test_cases)} test ; "
        f"graine={seed}.",
        flush=True,
    )
    selected, tuning_grid = _tune_on_cases(tuning_cases, data_directory)
    results_directory.mkdir(parents=True, exist_ok=True)
    tuning_grid.to_csv(results_directory / "brats_full_tuning_grid.csv", index=False)
    split["selected_parameters"] = {
        "opening_radius_mm": selected.opening_radius_mm,
        "closing_radius_mm": selected.closing_radius_mm,
        "threshold_method": selected.threshold_method,
    }
    split["tuning_metric"] = "mean whole-tumor Dice on tuning_cases"
    _save_json(split_file, split)
    print(
        "Paramètres sélectionnés sur le groupe de réglage : "
        f"ouverture={selected.opening_radius_mm:g} mm, "
        f"fermeture={selected.closing_radius_mm:g} mm, "
        f"seuil={selected.threshold_method}",
        flush=True,
    )

    progress_file = results_directory / "brats_test_progress.csv"
    progress_rows: dict[str, dict[str, str | float]] = {}
    if progress_file.is_file():
        previous = pd.read_csv(progress_file)
        expected_params = (
            selected.opening_radius_mm,
            selected.closing_radius_mm,
            selected.threshold_method,
        )
        for row in previous.to_dict(orient="records"):
            actual_params = (
                float(row["opening_radius_mm"]),
                float(row["closing_radius_mm"]),
                str(row["threshold_method"]),
            )
            if row["patient"] not in test_cases or actual_params != expected_params:
                raise ValueError(
                    "Le fichier de progression ne correspond pas au split ou "
                    "aux paramètres sélectionnés."
                )
            progress_rows[str(row["patient"])] = row

    for index, case_id in enumerate(test_cases, start=1):
        if case_id in progress_rows:
            continue
        started = time.monotonic()
        print(f"Test {index}/{len(test_cases)} : {case_id}", flush=True)
        progress_rows[case_id] = _evaluate_test_case(
            case_id, data_directory, selected
        )
        if index % 5 == 0 or index == len(test_cases):
            pd.DataFrame(
                [progress_rows[case] for case in test_cases if case in progress_rows]
            ).to_csv(progress_file, index=False)
        print(f"  terminé en {time.monotonic() - started:.1f} s", flush=True)

    test_results = pd.DataFrame([progress_rows[case] for case in test_cases])
    summary = _summarize_test_results(test_results, seed, bootstrap_replicates)
    test_results.to_csv(results_directory / "brats_test.csv", index=False)
    summary.to_csv(results_directory / "brats_test_summary.csv", index=False)
    progress_file.unlink(missing_ok=True)
    print(f"\nRésultats test : {results_directory / 'brats_test.csv'}")
    print(
        f"Résumé Dice (moyenne, écart-type, médiane, min, max, IC95 bootstrap "
        f"de la moyenne ; {bootstrap_replicates} réplications) :"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return test_results, summary, tuning_grid


def main() -> None:
    """Lance le réglage et l'évaluation complète sur les chemins configurés."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BRATS_RAW_DIRECTORY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--seed", type=int, default=BRATS_SPLIT_SEED)
    parser.add_argument("--tuning-count", type=int, default=BRATS_TUNING_CASES)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BRATS_BOOTSTRAP_REPLICATES,
    )
    arguments = parser.parse_args()
    run_full_evaluation(
        arguments.data_dir,
        arguments.results_dir,
        seed=arguments.seed,
        tuning_count=arguments.tuning_count,
        bootstrap_replicates=arguments.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
