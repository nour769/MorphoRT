"""Vérifie les volumes synthétiques et la lecture des examens NIfTI."""

import gzip

import nibabel as nib
import numpy as np
import pytest

from morphort.synthetic import MODALITY_NAMES, create_synthetic_mri
from morphort.volumes import read_nifti_bytes, validate_modalities


@pytest.mark.parametrize("shape", [(32, 32, 24), (24, 28, 32)])
def test_synthetic_mri_has_four_aligned_modalities_and_known_tumor(
    shape: tuple[int, int, int],
) -> None:
    """Les volumes de démo sont reproductibles, finis et partagent un masque."""
    first_modalities, first_tumor = create_synthetic_mri(shape, seed=3)
    second_modalities, second_tumor = create_synthetic_mri(shape, seed=3)

    assert tuple(first_modalities) == MODALITY_NAMES
    assert first_tumor.shape == shape
    assert np.any(first_tumor)
    for name in MODALITY_NAMES:
        assert first_modalities[name].shape == shape
        assert np.all(np.isfinite(first_modalities[name]))
        np.testing.assert_array_equal(first_modalities[name], second_modalities[name])
    np.testing.assert_array_equal(first_tumor, second_tumor)


@pytest.mark.parametrize("compressed", [False, True])
def test_read_nifti_bytes_reads_three_dimensional_images(compressed: bool) -> None:
    """Les formats .nii et .nii.gz sont chargés en conservant zooms et affine."""
    source = np.arange(6 * 7 * 8, dtype=np.float32).reshape((6, 7, 8))
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    payload = nib.Nifti1Image(source, affine).to_bytes()
    filename = "examen.nii"
    if compressed:
        payload = gzip.compress(payload)
        filename = "examen.nii.gz"

    volume, spacing, loaded_affine = read_nifti_bytes(payload, filename)

    np.testing.assert_array_equal(volume, source)
    assert spacing == (1.0, 1.5, 2.0)
    np.testing.assert_allclose(loaded_affine, affine)


def test_read_nifti_bytes_rejects_invalid_files_and_dimensions() -> None:
    """Un fichier invalide ou non 3D produit une erreur compréhensible."""
    with pytest.raises(ValueError, match="volume NIfTI"):
        read_nifti_bytes(b"not a volume", "examen.nii")
    with pytest.raises(ValueError, match="NIfTI"):
        read_nifti_bytes(b"data", "examen.mgz")

    image_4d = nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="3D"):
        read_nifti_bytes(image_4d.to_bytes(), "examen.nii")


def test_validate_modalities_requires_same_space() -> None:
    """Les séquences sont refusées si forme, espacement ou repère diffèrent."""
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    affine = np.eye(4)
    reference = (volume, (1.0, 1.0, 1.0), affine)

    validate_modalities(reference, (volume.copy(), (1.0, 1.0, 1.0), affine.copy()))
    with pytest.raises(ValueError, match="dimension"):
        validate_modalities(
            reference,
            (np.zeros((7, 8, 8), dtype=np.float32), (1.0, 1.0, 1.0), affine),
        )
    with pytest.raises(ValueError, match="espacement"):
        validate_modalities(reference, (volume, (1.0, 1.0, 1.1), affine))
    shifted_affine = affine.copy()
    shifted_affine[0, 3] = 4.0
    with pytest.raises(ValueError, match="repère spatial"):
        validate_modalities(reference, (volume, (1.0, 1.0, 1.0), shifted_affine))
