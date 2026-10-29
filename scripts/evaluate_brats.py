"""Évalue la segmentation morphologique sur les cas BraTS préparés."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from morphort.datasets import list_brats_cases, load_brats_case
from morphort.measurements import (
    dice_coefficient,
    hausdorff_distance_95_mm,
    jaccard_index,
)
from morphort.segmentation import segment_tumor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "brats_eval.csv"


def evaluate_brats(
    data_directory: Path | None = None,
    output_file: Path = DEFAULT_RESULTS,
) -> pd.DataFrame:
    """Segmente chaque cas complet et écrit une ligne de métriques par patient."""
    cases = list_brats_cases(data_directory)
    if not cases:
        source = data_directory or PROJECT_ROOT / "data" / "brats"
        raise FileNotFoundError(f"Aucun cas BraTS complet trouvé dans {source}.")

    rows: list[dict[str, str | float]] = []
    for index, case_id in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Évaluation de {case_id}...", flush=True)
        case = load_brats_case(case_id, data_directory)
        predicted, _steps, _threshold = segment_tumor(
            case.volumes["FLAIR"], case.spacing_mm
        )
        expert = case.expert_masks["Tumeur entière"]
        rows.append(
            {
                "patient": case_id,
                "dice": dice_coefficient(predicted, expert),
                "jaccard": jaccard_index(predicted, expert),
                "hausdorff95_mm": hausdorff_distance_95_mm(
                    predicted, expert, case.spacing_mm
                ),
            }
        )

    results = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\nRésultats enregistrés dans : {output_file}")
    print(f"Patients évalués : {len(results)}")
    print("Statistiques (écart-type échantillonnal) :")
    print(
        results[["dice", "jaccard", "hausdorff95_mm"]]
        .agg(["mean", "std", "min", "max"])
        .T.rename_axis("Métrique")
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    return results


def main() -> None:
    """Lance l'évaluation locale avec des chemins configurables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "brats",
        help="Dossier contenant les patients BraTS préparés.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Chemin du CSV de sortie (défaut : results/brats_eval.csv).",
    )
    arguments = parser.parse_args()
    evaluate_brats(arguments.data_dir, arguments.output)


if __name__ == "__main__":
    main()
