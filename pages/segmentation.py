"""Page 2 : segmentation de la tumeur et de ses sous-régions."""

import numpy as np
import pandas as pd
import streamlit as st

from morphort.config import DEFAULT_SEGMENTATION
from morphort.datasets import load_brats_case
from morphort.measurements import (
    dice_coefficient,
    hausdorff_distance_95_mm,
    jaccard_index,
)
from morphort.segmentation import (
    SegmentationStep,
    otsu_threshold,
    robust_normalize,
    segment_tumor,
)
from morphort.subregions import EDEMA, ENHANCING, NECROTIC, segment_tumor_subregions
from pages.common import show_data_origin

REGION_COLORS = {
    ENHANCING: (255, 75, 55),
    NECROTIC: (255, 210, 45),
    EDEMA: (40, 185, 255),
}


def _display_step(step: SegmentationStep, index: int, shape: tuple[int, ...]) -> None:
    """Affiche la formule et la coupe axiale d'une étape."""
    with st.expander(f"{index}. {step.name}", expanded=False):
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


def _colored_overlay(
    background: np.ndarray,
    masks: dict[str, np.ndarray],
    slice_index: int,
) -> np.ndarray:
    """Superpose les sous-régions sur une coupe FLAIR en conservant le fond."""
    image = background[:, :, slice_index].T
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, (1, 99)) if finite.size else (0.0, 0.0)
    if high > low:
        gray = np.clip((image - low) / (high - low), 0.0, 1.0)
    else:
        gray = np.zeros(image.shape, dtype=np.float64)
    rgb = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=2)
    for name, mask in masks.items():
        color = np.asarray(REGION_COLORS[name], dtype=np.float64)
        region = mask[:, :, slice_index].T
        rgb[region] = (
            0.35 * rgb[region].astype(np.float64) + 0.65 * color
        ).astype(np.uint8)
    return rgb


def _show_comparisons(
    flair: np.ndarray,
    predictions: dict[str, np.ndarray],
    expert_masks: dict[str, np.ndarray] | None,
    spacing_mm: tuple[float, float, float],
) -> None:
    """Affiche les cartes prédites, les références et les scores non filtrés."""
    slice_index = flair.shape[2] // 2
    regions = (ENHANCING, NECROTIC, EDEMA)
    predicted_regions = {name: predictions[name] for name in regions}
    expert_regions = (
        {
            name: expert_masks[name]
            for name in regions
            if name in expert_masks
        }
        if expert_masks
        else {}
    )
    columns = st.columns(2 if expert_regions else 1)
    with columns[0]:
        st.image(
            _colored_overlay(flair, predicted_regions, slice_index),
            caption="Prédictions morphologiques · coupe axiale centrale",
            width="stretch",
        )

    if expert_regions and len(columns) > 1:
        with columns[1]:
            st.image(
                _colored_overlay(flair, expert_regions, slice_index),
                caption="Masque expert BraTS · même coupe",
                width="stretch",
            )
    elif expert_masks:
        st.info(
            "Le masque de démonstration ne contient pas de références expertes "
            "séparées pour les sous-régions."
        )
    st.caption(
        "Légende : rouge = tumeur rehaussée · jaune = noyau nécrotique · "
        "bleu = œdème."
    )

    if not expert_masks:
        st.info("Aucun masque expert : les Dice par région ne sont pas calculés.")
        return

    all_predictions = {"Tumeur entière": predictions["Tumeur entière"]}
    all_predictions.update(predicted_regions)
    comparisons: list[dict[str, object]] = []
    for name, prediction in all_predictions.items():
        expert = expert_masks.get(name)
        if expert is None:
            continue
        comparisons.append(
            {
                "Région": name,
                "Dice": dice_coefficient(prediction, expert),
                "Jaccard": jaccard_index(prediction, expert),
                "Hausdorff 95 (mm)": hausdorff_distance_95_mm(
                    prediction, expert, spacing_mm
                ),
            }
        )

    st.subheader("Comparaison aux masques experts")
    st.dataframe(
        pd.DataFrame(comparisons),
        hide_index=True,
        width="stretch",
    )
    for result in comparisons:
        if result["Dice"] < 0.5:
            st.warning(
                f"{result['Région']} : Dice faible ({result['Dice']:.3f}). "
                "La méthode par seuils et morphologie ne suit pas bien le "
                "masque expert sur cette région."
            )


def _load_evaluation_case(case_id: str) -> None:
    """Charge un cas global faible et efface les résultats de l'ancien patient."""
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
    for key in (
        "morphort_segmentation",
        "morphort_segmentation_steps",
        "morphort_subregion_masks",
        "morphort_subregion_steps",
        "morphort_otsu_threshold",
        "morphort_measurements",
        "morphort_growth_observations",
        "morphort_growth_results",
        "morphort_growth_model",
        "morphort_growth_truth",
        "morphort_radiotherapy_results",
        "morphort_radiotherapy_parameters",
    ):
        st.session_state.pop(key, None)


def afficher() -> None:
    """Segmente l'examen FLAIR, sépare les régions et compare aux experts."""
    st.title("Segmenter la tumeur")
    show_data_origin()
    st.write(
        "Cette segmentation morphologique est exploratoire et doit être vérifiée. "
        "Les sous-régions sont estimées par seuils d'intensité et opérations binaires."
    )
    evaluation_case_ids = st.session_state.get("morphort_evaluation_case_ids", [])
    if evaluation_case_ids:
        selected_case = st.selectbox(
            "Patient de l'évaluation à inspecter (Dice croissant)",
            evaluation_case_ids,
            key="morphort_evaluation_case_selector",
        )
        if (
            not st.session_state.get("morphort_is_brats")
            or st.session_state.get("morphort_brats_case_id") != selected_case
            or "FLAIR" not in st.session_state.get("morphort_volumes", {})
        ):
            _load_evaluation_case(selected_case)

    volumes = st.session_state.get("morphort_volumes", {})
    if "FLAIR" not in volumes:
        st.info("Chargez une séquence FLAIR sur la première page.")
        return

    flair, spacing, _affine = volumes["FLAIR"]
    st.caption(f"FLAIR · {flair.shape[0]} x {flair.shape[1]} x {flair.shape[2]} voxels")
    opening_radius = st.slider(
        "Rayon de l'ouverture (mm)",
        min_value=0.5,
        max_value=4.0,
        value=DEFAULT_SEGMENTATION.opening_radius_mm,
        step=0.5,
    )
    closing_radius = st.slider(
        "Rayon de la fermeture (mm)",
        min_value=0.5,
        max_value=4.0,
        value=DEFAULT_SEGMENTATION.closing_radius_mm,
        step=0.5,
    )
    threshold_mode = st.radio(
        "Seuil FLAIR",
        ("Automatique (Otsu)", "Automatique (percentile 90)", "Régler manuellement"),
        index=(
            1
            if DEFAULT_SEGMENTATION.threshold_method == "percentile_90"
            else 0
        ),
        horizontal=True,
    )
    threshold = None
    threshold_method = (
        "percentile_90"
        if threshold_mode == "Automatique (percentile 90)"
        else "otsu"
    )
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
                opening_radius_mm=opening_radius,
                closing_radius_mm=closing_radius,
                threshold_method=threshold_method,
                threshold=threshold,
            )
            subregion_masks: dict[str, np.ndarray] = {}
            subregion_steps = {}
            t1ce_entry = next(
                (
                    volume_info
                    for name, volume_info in volumes.items()
                    if name.upper() == "T1CE"
                ),
                None,
            )
            if t1ce_entry is not None:
                subregion_masks, subregion_steps = segment_tumor_subregions(
                    t1ce_entry[0],
                    flair,
                    mask,
                    spacing,
                    opening_radius_mm=opening_radius,
                    closing_radius_mm=closing_radius,
                )
            st.session_state["morphort_segmentation"] = mask
            st.session_state["morphort_segmentation_steps"] = steps
            st.session_state["morphort_subregion_masks"] = subregion_masks
            st.session_state["morphort_subregion_steps"] = subregion_steps
            st.session_state["morphort_otsu_threshold"] = applied_threshold
            st.session_state["morphort_segmentation_parameters"] = {
                "opening_radius_mm": opening_radius,
                "closing_radius_mm": closing_radius,
                "threshold_method": threshold_method,
                "threshold": threshold,
            }
            st.session_state.pop("morphort_measurements", None)
            st.success(
                f"Segmentation terminée. Seuil appliqué : {applied_threshold:.3f}."
            )
        except ValueError as error:
            st.error(str(error))

    steps = st.session_state.get("morphort_segmentation_steps", [])
    if not steps:
        return
    mask = st.session_state["morphort_segmentation"]
    st.metric("Volume segmenté", f"{np.count_nonzero(mask):,} voxels")
    expert_masks = st.session_state.get("morphort_expert_masks")
    if expert_masks is None and st.session_state.get("morphort_demo_mask") is not None:
        expert_masks = {"Tumeur entière": st.session_state["morphort_demo_mask"]}
    subregion_masks = st.session_state.get("morphort_subregion_masks", {})
    predictions = {"Tumeur entière": mask, **subregion_masks}

    result_tab, steps_tab = st.tabs(("Superposition et scores", "Voir les étapes"))
    with result_tab:
        if subregion_masks:
            _show_comparisons(flair, predictions, expert_masks, spacing)
        else:
            st.info(
                "Les sous-régions nécessitent une séquence T1ce. "
                "La segmentation de la tumeur entière reste disponible."
            )
            if expert_masks and "Tumeur entière" in expert_masks:
                dice = dice_coefficient(mask, expert_masks["Tumeur entière"])
                st.metric("Dice · tumeur entière", f"{dice:.3f}")
                if dice < 0.5:
                    st.warning(
                        f"Dice faible ({dice:.3f}) : la segmentation "
                        "morphologique diffère fortement du masque expert."
                    )
            else:
                st.info("Aucun masque expert n'est disponible pour calculer le Dice.")

    with steps_tab:
        st.subheader("Tumeur entière")
        for index, step in enumerate(steps, start=1):
            _display_step(step, index, flair.shape)
        for name, region_steps in st.session_state.get(
            "morphort_subregion_steps", {}
        ).items():
            st.subheader(name)
            for index, step in enumerate(region_steps, start=1):
                _display_step(step, index, flair.shape)

    st.caption(
        "Résultat exploratoire uniquement : ce masque n'est pas une "
        "segmentation clinique validée."
    )


if __name__ == "__main__":
    afficher()
