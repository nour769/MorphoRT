"""Lecture et vérification des volumes IRM NIfTI."""

from __future__ import annotations

import gzip
import math
import zlib

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

FloatVolume = NDArray[np.float32]
FloatAffine = NDArray[np.float64]
VoxelSpacing = tuple[float, float, float]


def read_nifti_bytes(
    content: bytes, filename: str
) -> tuple[FloatVolume, VoxelSpacing, FloatAffine]:
    """Lit un fichier NIfTI 3D `.nii` ou `.nii.gz` fourni en mémoire."""
    normalized_name = filename.lower()
    if not normalized_name.endswith((".nii", ".nii.gz")):
        raise ValueError("Choisissez un fichier NIfTI .nii ou .nii.gz.")
    if not content:
        raise ValueError("Le fichier sélectionné est vide.")

    try:
        if normalized_name.endswith(".nii.gz"):
            nifti_bytes = gzip.decompress(content)
        else:
            nifti_bytes = content
        image = nib.Nifti1Image.from_bytes(nifti_bytes)
        if len(image.shape) != 3:
            raise ValueError("Le volume IRM doit être une image 3D.")
        if any(size < 2 for size in image.shape):
            raise ValueError(
                "Le volume IRM doit contenir au moins deux voxels par axe."
            )

        volume = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        if not np.all(np.isfinite(volume)):
            raise ValueError("Le volume IRM contient des valeurs non finies.")

        spacing_values = tuple(float(value) for value in image.header.get_zooms()[:3])
        if len(spacing_values) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in spacing_values
        ):
            raise ValueError("Les dimensions des voxels doivent être positives.")

        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("Le repère spatial de l'IRM est invalide.")
        return volume, spacing_values, affine
    except (
        OSError,
        EOFError,
        zlib.error,
        nib.filebasedimages.ImageFileError,
        nib.wrapstruct.WrapStructError,
    ) as error:
        raise ValueError(
            "Le fichier ne peut pas être lu comme un volume NIfTI valide."
        ) from error


def validate_modalities(
    reference: tuple[FloatVolume, VoxelSpacing, FloatAffine],
    candidate: tuple[FloatVolume, VoxelSpacing, FloatAffine],
) -> None:
    """Vérifie que deux séquences sont acquises dans le même espace voxel."""
    reference_volume, reference_spacing, reference_affine = reference
    candidate_volume, candidate_spacing, candidate_affine = candidate
    if reference_volume.shape != candidate_volume.shape:
        raise ValueError("Toutes les séquences IRM doivent avoir la même dimension.")
    if not np.allclose(reference_spacing, candidate_spacing, rtol=1e-3, atol=1e-3):
        raise ValueError(
            "Toutes les séquences IRM doivent avoir le même espacement voxel."
        )
    if not np.allclose(reference_affine, candidate_affine, rtol=1e-3, atol=1e-3):
        raise ValueError("Les séquences IRM doivent partager le même repère spatial.")
