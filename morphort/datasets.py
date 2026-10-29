"""Accès local aux cas BraTS2020 préparés par patient."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from morphort.morphology import BinaryImage
from morphort.volumes import (
    FloatAffine,
    FloatVolume,
    VoxelSpacing,
    read_nifti_bytes,
    validate_modalities,
)

PROJECT_DATA = Path(__file__).resolve().parents[1] / "data" / "brats"
MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
MASK_NAMES = ("Tumeur entière", "Noyau nécrotique", "Œdème", "Tumeur rehaussée")
ExpertMasks = dict[str, BinaryImage]
IntegerVolume = NDArray[np.uint8]
TRAINING_CASE = re.compile(r"^BraTS20_Training_\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class BratsCase:
    """Volumes réels et étiquettes expertes d'un patient BraTS2020."""

    case_id: str
    volumes: dict[str, FloatVolume]
    labels: IntegerVolume
    expert_masks: ExpertMasks
    spacing_mm: VoxelSpacing


def _case_files(case_directory: Path) -> dict[str, Path]:
    """Associe les suffixes de fichier réellement utilisés à leurs modalités."""
    files: dict[str, Path] = {}
    for path in case_directory.iterdir():
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        stem = path.name.lower()
        stem = stem[:-7] if stem.endswith(".nii.gz") else stem[:-4]
        for modality in MODALITIES:
            if stem.endswith(f"_{modality}"):
                files[modality] = path
                break
    return files


def list_brats_cases(data_directory: Path | None = None) -> list[str]:
    """Liste les patients copiés qui possèdent les cinq fichiers NIfTI."""
    root = PROJECT_DATA if data_directory is None else Path(data_directory)
    if not root.is_dir():
        return []
    return sorted(
        directory.name
        for directory in root.iterdir()
        if directory.is_dir()
        and TRAINING_CASE.fullmatch(directory.name)
        and set(_case_files(directory)) == set(MODALITIES)
    )


def load_brats_case(
    case_id: str, data_directory: Path | None = None
) -> BratsCase:
    """Charge les quatre séquences, les labels 0/1/2/4 et les masques BraTS."""
    if not case_id or not TRAINING_CASE.fullmatch(case_id):
        raise ValueError("Identifiant de patient BraTS invalide.")
    root = PROJECT_DATA if data_directory is None else Path(data_directory)
    case_directory = root / case_id
    if not case_directory.is_dir():
        raise FileNotFoundError(f"Le patient BraTS {case_id} est introuvable.")

    files = _case_files(case_directory)
    if set(files) != set(MODALITIES):
        missing = sorted(set(MODALITIES) - set(files))
        raise FileNotFoundError(
            f"Le patient {case_id} n'a pas tous les fichiers requis : "
            f"{', '.join(missing)}."
        )

    loaded: dict[str, tuple[FloatVolume, VoxelSpacing, FloatAffine]] = {}
    for modality in MODALITIES:
        path = files[modality]
        loaded[modality] = read_nifti_bytes(path.read_bytes(), path.name)

    reference = loaded["t1"]
    for modality in ("t1ce", "t2", "flair", "seg"):
        validate_modalities(reference, loaded[modality])

    raw_labels = loaded["seg"][0]
    rounded = np.rint(raw_labels)
    if not np.allclose(raw_labels, rounded, atol=1e-5):
        raise ValueError(f"Le masque expert du patient {case_id} n'est pas entier.")
    labels = rounded.astype(np.uint8)
    unexpected = np.setdiff1d(np.unique(labels), np.array([0, 1, 2, 4], dtype=np.uint8))
    if unexpected.size:
        raise ValueError(
            f"Le masque expert BraTS contient des labels inattendus : {unexpected}."
        )

    expert_masks: ExpertMasks = {
        "Tumeur entière": labels > 0,
        "Noyau nécrotique": labels == 1,
        "Œdème": labels == 2,
        "Tumeur rehaussée": labels == 4,
    }
    return BratsCase(
        case_id=case_id,
        volumes={name.upper(): loaded[name][0] for name in MODALITIES[:-1]},
        labels=labels,
        expert_masks=expert_masks,
        spacing_mm=reference[1],
    )
