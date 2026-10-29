"""Vérifie la préparation et le chargement BraTS sur de petits NIfTI factices."""

import nibabel as nib
import numpy as np
import pytest

from morphort.datasets import (
    list_brats_case_inventory,
    list_brats_cases,
    load_brats_case,
)
from scripts.prepare_brats import prepare_dataset

MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")


def _write_case(
    directory, case_id: str, *, include_seg: bool = True
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    for index, modality in enumerate(MODALITIES):
        if modality == "seg" and not include_seg:
            continue
        if modality == "seg":
            volume = np.zeros((8, 9, 10), dtype=np.uint8)
            volume[2:4, 2:5, 3:6] = 1
            volume[4:6, 2:5, 3:6] = 2
            volume[6, 2:5, 3:6] = 4
        else:
            volume = np.full((8, 9, 10), index + 1, dtype=np.float32)
        path = directory / f"{case_id}_{modality}.nii"
        nib.save(nib.Nifti1Image(volume, affine), path)


def test_prepare_dataset_selects_evenly_spaced_complete_training_cases(
    tmp_path,
) -> None:
    """Les validations sans masque sont omises et seuls trois cas sont copiés."""
    source = tmp_path / "brats20-dataset-training-validation"
    destination = tmp_path / "data" / "brats"
    case_ids = [f"BraTS20_Training_{index:03d}" for index in range(1, 10)]
    for case_id in case_ids:
        _write_case(source / "Training" / case_id, case_id)
    _write_case(source / "Validation" / "BraTS20_Training_010", "BraTS20_Training_010")
    (source / "Training" / case_ids[4] / f"{case_ids[4]}_seg.nii").unlink()

    selected = prepare_dataset(source, destination, 3)

    expected = [case_ids[0], case_ids[5], case_ids[8]]
    assert selected == expected
    assert list_brats_cases(destination) == expected
    assert all(
        len(list((destination / case_id).glob("*.nii"))) == len(MODALITIES)
        for case_id in expected
    )
    assert not (destination / case_ids[4]).exists()
    assert not (destination / "BraTS20_Training_010").exists()


def test_load_brats_case_returns_modalities_spacing_and_all_expert_regions(
    tmp_path,
) -> None:
    """Les labels BraTS 1/2/4 deviennent les masques experts attendus."""
    case_id = "BraTS20_Training_001"
    _write_case(tmp_path / case_id, case_id)

    case = load_brats_case(case_id, tmp_path)

    assert set(case.volumes) == {"T1", "T1CE", "T2", "FLAIR"}
    assert case.spacing_mm == pytest.approx((1.0, 1.5, 2.0))
    assert case.expert_masks["Noyau nécrotique"].sum() == 18
    assert case.expert_masks["Œdème"].sum() == 18
    assert case.expert_masks["Tumeur rehaussée"].sum() == 9
    assert case.expert_masks["Tumeur entière"].sum() == 45
    assert np.array_equal(case.expert_masks["Tumeur entière"], case.labels > 0)


def test_load_brats_case_rejects_non_training_path(tmp_path) -> None:
    """Un identifiant de chemin n'est jamais accepté comme patient."""
    with pytest.raises(ValueError, match="invalide"):
        load_brats_case("..", tmp_path)


def test_raw_nested_training_tree_is_discovered_and_loaded(tmp_path) -> None:
    """Les dossiers BraTS imbriqués du téléchargement brut sont pris en charge."""
    case_id = "BraTS20_Training_001"
    case_directory = (
        tmp_path
        / "BraTS2020_TrainingData"
        / "MICCAI_BraTS2020_TrainingData"
        / case_id
    )
    _write_case(case_directory, case_id)

    assert list_brats_cases(tmp_path) == [case_id]
    assert load_brats_case(case_id, tmp_path).case_id == case_id


def test_case_inventory_reports_missing_expert_mask(tmp_path) -> None:
    """Un patient sans seg est explicitement exclu des métriques supervisées."""
    case_id = "BraTS20_Training_355"
    case_directory = tmp_path / "Training" / case_id
    _write_case(case_directory, case_id, include_seg=False)

    complete, incomplete = list_brats_case_inventory(tmp_path)

    assert complete == []
    assert incomplete == {case_id: ["seg"]}
