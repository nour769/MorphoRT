"""Page 1 : chargement et aperçu orthogonal des séquences IRM."""

import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from morphort.synthetic import MODALITY_NAMES, create_synthetic_mri
from morphort.volumes import (
    FloatAffine,
    FloatVolume,
    read_nifti_bytes,
    validate_modalities,
)

VolumeInfo = tuple[FloatVolume, tuple[float, float, float], FloatAffine]


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
    modalities, tumor = create_synthetic_mri()
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    spacing = (1.0, 1.0, 1.5)
    st.session_state["morphort_volumes"] = {
        name: (volume, spacing, affine) for name, volume in modalities.items()
    }
    st.session_state["morphort_demo_mask"] = tumor
    st.session_state["morphort_is_demo"] = True


def _load_uploaded_files(
    files: dict[str, UploadedFile | None],
) -> None:
    """Lit les séquences choisies et confirme leur alignement spatial."""
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
    st.session_state.pop("morphort_demo_mask", None)
    st.session_state["morphort_is_demo"] = False


def afficher() -> None:
    """Charge une IRM ou une démo et affiche ses trois plans."""
    st.title("Charger l'IRM")
    st.write(
        "Choisissez la démonstration synthétique ou ajoutez vos fichiers NIfTI. "
        "Les fichiers sont lus en mémoire et ne sont pas enregistrés."
    )

    source = st.radio(
        "Source des images",
        ("Démo synthétique", "Fichiers NIfTI"),
        horizontal=True,
    )
    files: dict[str, UploadedFile | None] = {}
    if source == "Fichiers NIfTI":
        st.caption("Ajoutez au moins une séquence. Formats acceptés : .nii et .nii.gz.")
        for modality in MODALITY_NAMES:
            files[modality] = st.file_uploader(
                modality,
                type=["nii", "gz"],
                key=f"upload_{modality}",
                max_upload_size=300,
            )

    if st.button("Charger l'IRM", type="primary"):
        try:
            if source == "Démo synthétique":
                _load_demo()
            else:
                _load_uploaded_files(files)
        except ValueError as error:
            st.error(str(error))

    volumes = st.session_state.get("morphort_volumes")
    if not volumes:
        st.info("Chargez une image pour afficher les coupes.")
        return

    modality = st.selectbox("Séquence à afficher", list(volumes))
    volume, spacing, _affine = volumes[modality]
    source_label = (
        "Démo entièrement synthétique"
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
