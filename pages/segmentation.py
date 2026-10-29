"""Page 2 : segmentation de la tumeur et explication des opérations."""

import numpy as np
import pandas as pd
import streamlit as st

from morphort.measurements import dice_coefficient, hausdorff_distance_mm, jaccard_index
from morphort.segmentation import (
    SegmentationStep,
    otsu_threshold,
    robust_normalize,
    segment_tumor,
)
from pages.common import show_data_origin


def _display_step(step: SegmentationStep, index: int, shape: tuple[int, ...]) -> None:
    """Affiche la formule et la coupe axiale d'une étape."""
    with st.expander(f"{index}. {step.name}", expanded=index == 6):
        st.latex(step.formula)
        st.write(step.explanation)
        if step.image.dtype == np.bool_:
            slice_image = step.image[:, :, shape[2] // 2].T.astype(np.uint8) * 255
        else:
            image = step.image[:, :, shape[2] // 2].T
            finite = image[np.isfinite(image)]
            low, high = np.percentile(finite, (1, 99)) if finite.size else (0.0, 0.0)
            if high > low:
                slice_image = (
                    np.clip((image - low) / (high - low), 0.0, 1.0) * 255
                ).astype(np.uint8)
            else:
                slice_image = np.zeros(image.shape, dtype=np.uint8)
        st.image(
            slice_image,
            caption=(
                f"Coupe axiale centrale · "
                f"{int(np.count_nonzero(step.image)):,} voxels"
            ),
            width=300,
        )


def afficher() -> None:
    """Segmente l'examen FLAIR choisi et détaille chaque étape."""
    st.title("Segmenter la tumeur")
    show_data_origin()
    st.write(
        "Le pipeline est une démonstration mathématique. Une segmentation "
        "automatique doit être vérifiée par un spécialiste."
    )
    volumes = st.session_state.get("morphort_volumes", {})
    if "FLAIR" not in volumes:
        st.info("Chargez une séquence FLAIR sur la première page.")
        return

    flair, spacing, _affine = volumes["FLAIR"]
    st.caption(f"FLAIR · {flair.shape[0]} x {flair.shape[1]} x {flair.shape[2]} voxels")
    radius = st.slider(
        "Rayon de l'élément structurant (mm)",
        min_value=0.5,
        max_value=4.0,
        value=1.0,
        step=0.5,
    )
    threshold_mode = st.radio(
        "Seuil appliqué",
        ("Automatique (Otsu)", "Régler manuellement"),
        horizontal=True,
    )
    threshold = None
    if threshold_mode == "Régler manuellement":
        normalized, brain = robust_normalize(flair)
        automatic_threshold = otsu_threshold(normalized[brain])
        threshold = st.number_input(
            "Seuil sur l'intensité normalisée",
            value=float(automatic_threshold),
            step=0.25,
            format="%.2f",
        )

    if st.button("Lancer la segmentation morphologique", type="primary"):
        try:
            mask, steps, applied_threshold = segment_tumor(
                flair,
                spacing,
                opening_radius_mm=radius,
                threshold=threshold,
            )
            st.session_state["morphort_segmentation"] = mask
            st.session_state["morphort_segmentation_steps"] = steps
            st.session_state["morphort_otsu_threshold"] = applied_threshold
            st.session_state["morphort_segmentation_parameters"] = {
                "opening_radius_mm": radius,
                "threshold": threshold,
            }
            st.session_state.pop("morphort_measurements", None)
            st.success(
                f"Segmentation terminée. Seuil appliqué : {applied_threshold:.3f}."
            )
        except ValueError as error:
            st.error(str(error))

    steps = st.session_state.get("morphort_segmentation_steps", [])
    for index, step in enumerate(steps, start=1):
        _display_step(step, index, flair.shape)

    if steps:
        mask = st.session_state["morphort_segmentation"]
        st.metric("Volume segmenté", f"{np.count_nonzero(mask):,} voxels")
        expert_masks = st.session_state.get("morphort_expert_masks")
        if (
            expert_masks is None
            and st.session_state.get("morphort_demo_mask") is not None
        ):
            expert_masks = {
                "Tumeur entière": st.session_state["morphort_demo_mask"]
            }
        if expert_masks:
            rows = []
            whole_tumor_mask = expert_masks.get("Tumeur entière")
            if whole_tumor_mask is not None:
                rows.append(
                    {
                        "Région": "Tumeur entière",
                        "Dice": dice_coefficient(mask, whole_tumor_mask),
                        "Jaccard": jaccard_index(mask, whole_tumor_mask),
                        "Hausdorff (mm)": hausdorff_distance_mm(
                            mask, whole_tumor_mask, spacing
                        ),
                        "Évaluation": "Comparaison effectuée",
                    }
                )
            for name in expert_masks:
                if name != "Tumeur entière":
                    rows.append(
                        {
                            "Région": name,
                            "Dice": None,
                            "Jaccard": None,
                            "Hausdorff (mm)": None,
                            "Évaluation": (
                                "Non évaluée : le pipeline ne sépare pas "
                                "les sous-régions"
                            ),
                        }
                    )
            st.subheader("Comparaison au masque expert")
            if any(row["Dice"] is None for row in rows):
                st.info(
                    "Le pipeline produit un masque de tumeur entière seulement. "
                    "Les sous-régions expertes sont listées, mais aucun score "
                    "n'est calculé pour elles sans prédictions séparées."
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            low_scores = [
                row["Région"]
                for row in rows
                if row["Dice"] is not None and row["Dice"] < 0.5
            ]
            if low_scores:
                st.warning(
                    "Dice inférieur à 0,50 pour "
                    f"{', '.join(low_scores)}. Essayez d'ajuster le rayon "
                    "morphologique ou le seuil, sans masquer le score obtenu."
                )
        else:
            st.info(
                "Aucun masque expert n'est disponible pour cet examen ; "
                "les métriques de recouvrement ne sont pas calculées."
            )
        st.caption(
            "Résultat exploratoire uniquement : ce masque n'est pas une "
            "segmentation clinique validée."
        )
