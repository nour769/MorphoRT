"""Pages Streamlit partagées entre la navigation et les liens internes."""

import streamlit as st

from pages import (
    charger_irm,
    croissance,
    mesures,
    radiotherapie,
)

CHARGER_PAGE = st.Page(
    charger_irm.afficher,
    title="1. Charger l'IRM",
    url_path="charger-irm",
)
SEGMENTATION_PAGE = st.Page(
    "pages/segmentation.py",
    title="2. Segmenter la tumeur",
    url_path="segmentation",
)
MESURES_PAGE = st.Page(
    mesures.afficher,
    title="3. Mesurer la tumeur",
    url_path="mesures",
)
CROISSANCE_PAGE = st.Page(
    croissance.afficher,
    title="4. Étudier la croissance",
    url_path="croissance",
)
RADIOTHERAPIE_PAGE = st.Page(
    radiotherapie.afficher,
    title="5. Simuler la radiothérapie",
    url_path="radiotherapie",
)
RESULTATS_PAGE = st.Page(
    "pages/resultats_globaux.py",
    title="Résultats globaux",
    url_path="resultats-globaux",
)

PAGES = (
    CHARGER_PAGE,
    SEGMENTATION_PAGE,
    MESURES_PAGE,
    CROISSANCE_PAGE,
    RADIOTHERAPIE_PAGE,
    RESULTATS_PAGE,
)
