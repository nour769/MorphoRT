"""Point d'entrée de l'application locale MorphoRT."""

import streamlit as st

from pages import charger_irm, croissance, mesures, radiotherapie, segmentation

st.set_page_config(
    page_title="MorphoRT",
    page_icon="🧠",
    layout="wide",
)

st.warning("Simulation uniquement, pas d'usage clinique", icon="⚠️")

navigation = st.navigation(
    [
        st.Page(
            charger_irm.afficher,
            title="1. Charger l'IRM",
            url_path="charger-irm",
        ),
        st.Page(
            segmentation.afficher,
            title="2. Segmenter la tumeur",
            url_path="segmentation",
        ),
        st.Page(
            mesures.afficher,
            title="3. Mesurer la tumeur",
            url_path="mesures",
        ),
        st.Page(
            croissance.afficher,
            title="4. Étudier la croissance",
            url_path="croissance",
        ),
        st.Page(
            radiotherapie.afficher,
            title="5. Simuler la radiothérapie",
            url_path="radiotherapie",
        ),
    ],
    position="sidebar",
)
navigation.run()
