"""Page 1 : chargement et aperçu orthogonal des séquences IRM."""

import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from morphort.datasets import list_brats_cases, load_brats_case
from morphort.synthetic import MODALITY_NAMES, create_synthetic_mri
from morphort.volumes import (
    FloatAffine,
    FloatVolume,
    read_nifti_bytes,
    validate_modalities,
)
from pages.common import show_data_origin

VolumeInfo = tuple[FloatVolume, tuple[float, float, float], FloatAffine]
PIPELINE_STATE_KEYS = (
    "morphort_volumes",
    "morphort_expert_labels",
    "morphort_expert_masks",
    "morphort_is_brats",
    "morphort_brats_case_id",
    "morphort_demo_mask",
    "morphort_is_demo",
    "morphort_growth_observations",
    "morphort_growth_results",
    "morphort_growth_model",
    "morphort_segmentation",
    "morphort_segmentation_steps",
    "morphort_otsu_threshold",
    "morphort_measurements",
    "morphort_dice",
    "morphort_radiotherapy_results",
    "morphort_radiotherapy_parameters",
    "growth_table",
)


def _clear_previous_exam() -> None:
    """Évite de mélanger les résultats de deux examens successifs."""
    for key in PIPELINE_STATE_KEYS:
        st.session_state.pop(key, None)


def _show_slice(volume: FloatVolume, plane: str, index: int) -> np.ndarray:
    """Extrait et normalise une coupe axiale, coronale ou sagittale."""
    if plane == "Axiale":
        image = volume[:, :, index].T
    elif plane == "Coronale":
        image = volume[:, index, :].T
    else:
        image = volume[index, :, :].T

    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _show_volume(volume: FloatVolume) -> None:
    """Affiche les trois plans orthogonaux avec un curseur par plan."""
    sizes = {
        "Axiale": volume.shape[2],
        "Coronale": volume.shape[1],
        "Sagittale": volume.shape[0],
    }
    columns = st.columns(3)
    for column, (plane, count) in zip(columns, sizes.items(), strict=True):
        with column:
            index = st.slider(
                f"Coupe {plane.lower()}",
                min_value=0,
                max_value=count - 1,
                value=count // 2,
                key=f"slice_{plane.lower()}",
            )
            st.image(
                _show_slice(volume, plane, index),
                caption=f"{plane} — coupe {index + 1} sur {count}",
                use_container_width=True,
            )


def _load_demo() -> None:
    """Crée les séquences de démo et leur masque connu."""
    _clear_previous_exam()
    modalities, tumor = create_synthetic_mri()
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    spacing = (1.0, 1.0, 1.5)
    st.session_state["morphort_volumes"] = {
        name: (volume, spacing, affine) for name, volume in modalities.items()
    }
    st.session_state["morphort_demo_mask"] = tumor
    st.session_state["morphort_is_demo"] = True
    st.session_state["morphort_is_brats"] = False
    synthetic_volume = float(np.count_nonzero(tumor)) * np.prod(spacing) / 1000
    st.session_state["morphort_growth_observations"] = [
        (0.0, synthetic_volume),
        (14.0, synthetic_volume * 1.12),
        (28.0, synthetic_volume * 1.28),
    ]


def _load_uploaded_files(
    files: dict[str, UploadedFile | None],
) -> None:
    """Lit les séquences choisies et confirme leur alignement spatial."""
    _clear_previous_exam()
    loaded: dict[str, VolumeInfo] = {}
    for name, uploaded_file in files.items():
        if uploaded_file is None:
            continue
        volume, spacing, affine = read_nifti_bytes(
            uploaded_file.getvalue(), uploaded_file.name
        )
        loaded[name] = (volume, spacing, affine)

    if not loaded:
        raise ValueError("Ajoutez au moins une séquence IRM avant de continuer.")

    reference = next(iter(loaded.values()))
    for candidate in list(loaded.values())[1:]:
        validate_modalities(reference, candidate)

    st.session_state["morphort_volumes"] = loaded
    st.session_state["morphort_is_demo"] = False
    st.session_state["morphort_is_brats"] = False


def _load_brats(case_id: str) -> None:
    """Charge un cas BraTS préparé depuis les cinq fichiers vérifiés."""
    _clear_previous_exam()
    case = load_brats_case(case_id)
    affine = np.eye(4, dtype=np.float64)
    st.session_state["morphort_volumes"] = {
        name: (volume, case.spacing_mm, affine)
        for name, volume in case.volumes.items()
    }
    st.session_state["morphort_expert_labels"] = case.labels
    st.session_state["morphort_expert_masks"] = case.expert_masks
    st.session_state["morphort_brats_case_id"] = case.case_id
    st.session_state["morphort_is_brats"] = True
    st.session_state["morphort_is_demo"] = False


def afficher() -> None:
    """Charge une IRM ou une démo et affiche ses trois plans."""
    st.title("Charger l'IRM")
    show_data_origin()
    st.write(
        "Choisissez la démonstration synthétique ou ajoutez vos fichiers NIfTI. "
        "Les fichiers sont lus en mémoire et ne sont pas enregistrés."
    )

    source = st.radio(
        "Source des images",
        ("BraTS2020", "Démo synthétique", "Mes fichiers NIfTI"),
        horizontal=True,
    )
    case_ids = list_brats_cases() if source == "BraTS2020" else []
    case_id = (
        st.selectbox("Patient BraTS2020", case_ids, key="brats_case_selector")
        if case_ids
        else None
    )
    files: dict[str, UploadedFile | None] = {}
    if source == "Mes fichiers NIfTI":
        st.caption("Ajoutez au moins une séquence. Formats acceptés : .nii et .nii.gz.")
        for modality in MODALITY_NAMES:
            files[modality] = st.file_uploader(
                modality,
                type=["nii", "gz"],
                key=f"upload_{modality}",
                max_upload_size=300,
            )

    if source == "BraTS2020" and not case_ids:
        st.info("Aucun cas BraTS prêt. Préparez d'abord des patients dans data/brats/.")

    if st.button("Charger l'examen", type="primary"):
        try:
            if source == "BraTS2020":
                if case_id is None:
                    raise ValueError("Aucun patient BraTS préparé n'est disponible.")
                _load_brats(case_id)
            elif source == "Démo synthétique":
                _load_demo()
            else:
                _load_uploaded_files(files)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))

    volumes = st.session_state.get("morphort_volumes")
    if not volumes:
        st.info("Chargez une image pour afficher les coupes.")
        return

    modality = st.selectbox("Séquence à afficher", list(volumes))
    volume, spacing, _affine = volumes[modality]
    source_label = "Volume réel BraTS2020" if st.session_state.get(
        "morphort_is_brats"
    ) else (
        "Données synthétiques"
        if st.session_state.get("morphort_is_demo")
        else "Volume chargé depuis vos fichiers"
    )
    st.caption(
        f"{source_label} · dimensions {volume.shape} · "
        f"taille voxel {spacing[0]:g} x {spacing[1]:g} x {spacing[2]:g} mm"
    )
    _show_volume(volume)

    if st.session_state.get("morphort_is_demo"):
        st.success(
            "Les images et le masque de démonstration sont synthétiques "
            "et ne représentent aucun patient."
        )
    elif st.session_state.get("morphort_is_brats"):
        st.success(
            "Masque expert fourni avec BraTS2020. Les résultats restent "
            "exploratoires et ne sont pas destinés à un usage clinique."
        )
