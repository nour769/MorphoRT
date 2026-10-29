"""Page 3 : mesures physiques et comparaison à un masque expert."""

import streamlit as st

from morphort.measurements import dice_coefficient, measure_mask
from pages.common import show_data_origin


def afficher() -> None:
    """Affiche les mesures calculées pour le masque enregistré."""
    st.title("Mesurer la tumeur")
    show_data_origin()
    mask = st.session_state.get("morphort_segmentation")
    volumes = st.session_state.get("morphort_volumes", {})
    if mask is None:
        st.info("Lancez d'abord la segmentation sur la deuxième page.")
        return

    reference = next(iter(volumes.values()), None)
    if reference is None:
        st.error("Aucun volume IRM n'est disponible pour retrouver l'espacement.")
        return
    _, spacing, _affine = reference

    if st.button("Calculer les mesures", type="primary"):
        try:
            result = measure_mask(mask, spacing)
            st.session_state["morphort_measurements"] = result
            expert = st.session_state.get("morphort_demo_mask")
            if expert is not None:
                st.session_state["morphort_dice"] = dice_coefficient(mask, expert)
        except ValueError as error:
            st.error(str(error))

    result = st.session_state.get("morphort_measurements")
    if result is None:
        st.info("Lancez le calcul pour afficher les mesures.")
        return

    columns = st.columns(4)
    columns[0].metric("Volume", f"{result['volume_ml']:.2f} mL")
    columns[1].metric("Surface", f"{result['surface_cm2']:.2f} cm²")
    columns[2].metric("Sphéricité", f"{result['sphericity']:.3f}")
    columns[3].metric("Compacité", f"{result['compactness']:.3f}")

    dice = st.session_state.get("morphort_dice")
    if dice is not None:
        st.metric("Dice contre le masque synthétique connu", f"{dice:.3f}")
    else:
        st.info("Aucun masque expert n'est disponible : le Dice n'est pas calculé.")
    st.caption(
        "Les mesures décrivent le masque produit par cette démonstration ; "
        "elles ne constituent pas une interprétation médicale."
    )
