"""Analyse les cinq pires cas sans utiliser le test pour choisir une correction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morphort.config import DEFAULT_SEGMENTATION
from morphort.datasets import load_brats_case
from morphort.measurements import dice_coefficient, hausdorff_distance_95_mm
from morphort.segmentation import EmptySegmentationError, segment_tumor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
EVALUATION_FILE = RESULTS / "brats_eval.csv"
TUNING_PATIENTS_FILE = RESULTS / "brats_tuning_patients.csv"
OVERLAY_DIRECTORY = RESULTS / "failure_overlays"
DATA_DIRECTORY = PROJECT_ROOT / "data" / "brats"


def _segment(case, opening_method: str) -> np.ndarray:
    """Applique les paramètres verrouillés avec la variante d'ouverture choisie."""
    try:
        prediction, _, _ = segment_tumor(
            case.volumes["FLAIR"],
            case.spacing_mm,
            opening_radius_mm=DEFAULT_SEGMENTATION.opening_radius_mm,
            closing_radius_mm=DEFAULT_SEGMENTATION.closing_radius_mm,
            threshold_method=DEFAULT_SEGMENTATION.threshold_method,
            opening_method=opening_method,
            include_diagnostic_steps=False,
        )
    except EmptySegmentationError:
        return np.zeros(case.labels.shape, dtype=np.bool_)
    return prediction


def _volume_ml(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Convertit le nombre de voxels en millilitres."""
    return float(np.count_nonzero(mask) * np.prod(spacing_mm) / 1000.0)


def _center_distance_mm(
    first: np.ndarray, second: np.ndarray, spacing_mm: tuple[float, float, float]
) -> float:
    """Distance physique entre les centres de masse de deux masques."""
    if not first.any() or not second.any():
        return float("inf")
    first_center = np.asarray(np.argwhere(first).mean(axis=0)) * spacing_mm
    second_center = np.asarray(np.argwhere(second).mean(axis=0)) * spacing_mm
    return float(np.linalg.norm(first_center - second_center))


def _probable_cause(
    predicted: np.ndarray,
    expert: np.ndarray,
    spacing_mm: tuple[float, float, float],
    dice: float,
    hausdorff95_mm: float,
) -> str:
    """Attribue une hypothèse descriptive à partir des volumes et positions."""
    if not predicted.any():
        return "Tumeur perdue après seuillage/ouverture : masque prédit vide."
    ratio = np.count_nonzero(predicted) / max(np.count_nonzero(expert), 1)
    distance = _center_distance_mm(predicted, expert, spacing_mm)
    if distance > 20 and hausdorff95_mm > 20 and ratio < 0.75:
        return (
            "Sous-segmentation avec composante prédite probablement éloignée "
            "du foyer expert."
        )
    if distance > 20 and hausdorff95_mm > 20:
        return "Faux positif probablement distant du foyer expert."
    if ratio < 0.75:
        return "Sous-segmentation probable : le seuil ou l'ouverture retire du volume."
    if ratio > 1.5:
        return "Sur-segmentation probable : seuil trop permissif ou faux positifs."
    if dice < 0.3:
        return "Seuil probablement inadapté ou forme prédite mal localisée."
    return "Contour morphologique probablement différent du masque expert."


def _save_overlay(
    flair: np.ndarray,
    predicted: np.ndarray,
    expert: np.ndarray,
    case_id: str,
) -> Path:
    """Enregistre la coupe axiale ayant la plus grande surface experte."""
    areas = np.count_nonzero(expert, axis=(0, 1))
    slice_index = int(np.argmax(areas))
    image = flair[:, :, slice_index].T
    finite = image[np.isfinite(image)]
    low, high = (
        np.percentile(finite, (1, 99)) if finite.size else (0.0, 0.0)
    )
    gray = (
        np.clip((image - low) / (high - low), 0, 1)
        if high > low
        else np.zeros(image.shape, dtype=np.float64)
    )
    rgb = np.repeat(gray[..., None], 3, axis=2)
    pred_slice = predicted[:, :, slice_index].T
    expert_slice = expert[:, :, slice_index].T
    rgb[expert_slice & ~pred_slice] = (1.0, 0.15, 0.15)
    rgb[pred_slice & ~expert_slice] = (0.1, 0.85, 1.0)
    rgb[pred_slice & expert_slice] = (1.0, 0.85, 0.1)

    OVERLAY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = OVERLAY_DIRECTORY / f"{case_id}.png"
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(rgb, origin="lower")
    axis.set_title(f"{case_id} · coupe {slice_index}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def _select_correction(training_ids: list[str]) -> tuple[str, pd.DataFrame]:
    """Choisit entre deux ouvertures sur le groupe de réglage uniquement."""
    rows: list[dict[str, str | float]] = []
    for index, case_id in enumerate(training_ids, start=1):
        print(
            f"Réglage correction {index}/{len(training_ids)} : {case_id}",
            flush=True,
        )
        case = load_brats_case(case_id, DATA_DIRECTORY)
        expert = case.expert_masks["Tumeur entière"]
        for method in ("standard", "reconstruction"):
            predicted = _segment(case, method)
            rows.append(
                {
                    "patient": case_id,
                    "opening_method": method,
                    "dice": dice_coefficient(predicted, expert),
                }
            )
    scores = pd.DataFrame(rows)
    means = scores.groupby("opening_method")["dice"].mean()
    selected = (
        "reconstruction"
        if means["reconstruction"] > means["standard"]
        else "standard"
    )
    summary = (
        scores.groupby("opening_method", as_index=False)["dice"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(RESULTS / "brats_failure_correction_tuning.csv", index=False)
    print("\nComparaison limitée au groupe de réglage :")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Correction verrouillée : {selected}")
    return selected, scores


def analyze_failures() -> pd.DataFrame:
    """Produit métriques, causes probables et superpositions des cinq pires cas."""
    if not EVALUATION_FILE.is_file() or not TUNING_PATIENTS_FILE.is_file():
        raise FileNotFoundError(
            "Les CSV d'évaluation et de séparation sont requis. "
            "Relancez d'abord les scripts d'évaluation et de réglage."
        )
    evaluation = pd.read_csv(EVALUATION_FILE)
    tuning = pd.read_csv(TUNING_PATIENTS_FILE)
    if "threshold_method" in evaluation and (
        evaluation["threshold_method"].nunique() != 1
        or evaluation["threshold_method"].iloc[0]
        != DEFAULT_SEGMENTATION.threshold_method
    ):
        raise ValueError("Le CSV global ne correspond pas à la configuration active.")
    training_ids = sorted(
        tuning.loc[tuning["split"] == "réglage", "patient"].astype(str).unique()
    )
    if not training_ids:
        raise ValueError("Le CSV de réglage ne contient aucun patient de réglage.")

    selected_method, _training_scores = _select_correction(training_ids)
    worst_cases = evaluation.sort_values("dice", kind="stable").head(5)
    rows: list[dict[str, str | float]] = []
    for case_id in worst_cases["patient"].astype(str):
        case = load_brats_case(case_id, DATA_DIRECTORY)
        expert = case.expert_masks["Tumeur entière"]
        baseline = _segment(case, "standard")
        reconstruction_candidate = _segment(case, "reconstruction")
        dice = dice_coefficient(baseline, expert)
        hausdorff = hausdorff_distance_95_mm(
            baseline, expert, case.spacing_mm
        )
        overlay_path = _save_overlay(
            case.volumes["FLAIR"], baseline, expert, case_id
        )
        rows.append(
            {
                "patient": case_id,
                "split": (
                    "réglage"
                    if case_id in training_ids
                    else "test_indépendant"
                ),
                "dice": dice,
                "predicted_volume_ml": _volume_ml(baseline, case.spacing_mm),
                "expert_volume_ml": _volume_ml(expert, case.spacing_mm),
                "hausdorff95_mm": hausdorff,
                "center_distance_mm": _center_distance_mm(
                    baseline, expert, case.spacing_mm
                ),
                "probable_cause": _probable_cause(
                    baseline, expert, case.spacing_mm, dice, hausdorff
                ),
                "selected_opening_method": selected_method,
                "reconstruction_candidate_dice": dice_coefficient(
                    reconstruction_candidate, expert
                ),
                "reconstruction_candidate_delta": dice_coefficient(
                    reconstruction_candidate, expert
                )
                - dice,
                "overlay": str(overlay_path.relative_to(PROJECT_ROOT)),
            }
        )

    results = pd.DataFrame(rows)
    output_file = RESULTS / "brats_failure_analysis.csv"
    results.to_csv(output_file, index=False)
    print(
        "\nCinq cas les plus faibles "
        "(rouge expert, cyan prédit, jaune recouvrement) :"
    )
    print(
        results[
            [
                "patient",
                "split",
                "dice",
                "predicted_volume_ml",
                "expert_volume_ml",
                "hausdorff95_mm",
                "probable_cause",
                "reconstruction_candidate_dice",
                "reconstruction_candidate_delta",
                "overlay",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nDétails CSV : {output_file}")
    print(
        "La variante a été choisie uniquement par le Dice moyen du groupe "
        "de réglage ; aucun score du groupe test n'a servi au choix."
    )
    return results


def main() -> None:
    """Exécute l'analyse reproductible des échecs."""
    analyze_failures()


if __name__ == "__main__":
    main()
