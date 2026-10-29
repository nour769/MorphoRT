"""Copie un sous-ensemble uniforme des cas BraTS d'entraînement."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
TRAINING_CASE = re.compile(r"^BraTS20_Training_\d+$", re.IGNORECASE)


def _case_files(source: Path) -> dict[str, dict[str, Path]]:
    """Repère les cinq NIfTI des dossiers d'entraînement réellement présents."""
    cases: dict[str, dict[str, Path]] = {}
    for path in source.rglob("*"):
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        relative_parts = path.relative_to(source).parts
        case_id = path.parent.name
        if not TRAINING_CASE.fullmatch(case_id) or any(
            "validation" in part.lower() for part in relative_parts[:-1]
        ):
            continue
        stem = path.name.lower()
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        else:
            stem = stem[:-4]
        modality = next(
            (name for name in MODALITIES if stem.endswith(f"_{name}")), None
        )
        if modality is not None:
            cases.setdefault(case_id, {})[modality] = path
    return {
        case_id: files
        for case_id, files in sorted(cases.items())
        if all(modality in files for modality in MODALITIES)
    }


def _evenly_spaced(items: list[str], count: int) -> list[str]:
    """Choisit des indices répartis du premier au dernier cas disponible."""
    if count < 1:
        raise ValueError("N doit être supérieur ou égal à 1.")
    if count >= len(items):
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    ]
    return [items[index] for index in indices]


def prepare_dataset(source: Path, destination: Path, count: int) -> list[str]:
    """Copie seulement les cinq fichiers des N cas complets retenus."""
    cases = _case_files(source)
    if not cases:
        raise FileNotFoundError(
            f"Aucun cas BraTS20_Training complet trouvé sous {source}."
        )

    selected = _evenly_spaced(list(cases), count)
    for case_id in selected:
        target = destination / case_id
        target.mkdir(parents=True, exist_ok=True)
        for modality in MODALITIES:
            source_file = cases[case_id][modality]
            target_file = target / source_file.name
            if not target_file.exists():
                shutil.copy2(source_file, target_file)
    return selected


def main() -> None:
    """Expose la préparation comme une commande locale."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Downloads" / "brats_raw",
        help="Racine des fichiers BraTS bruts (défaut : ~/Downloads/brats_raw).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "data" / "brats",
        help="Dossier des cas copiés (défaut : data/brats).",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=20,
        help="Nombre de cas d'entraînement à copier (défaut : 20).",
    )
    arguments = parser.parse_args()
    selected = prepare_dataset(
        arguments.source, arguments.destination, arguments.n
    )
    print(f"{len(selected)} cas préparés dans {arguments.destination}:")
    for case_id in selected:
        print(f"  {case_id}")


if __name__ == "__main__":
    main()
