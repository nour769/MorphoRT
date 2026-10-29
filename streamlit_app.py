"""Point d'entrée de l'application locale MorphoRT."""

from pathlib import Path

import streamlit as st

from pages import charger_irm, croissance, mesures, radiotherapie, segmentation

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="MorphoRT",
    page_icon="🧠",
    layout="wide",
)

with (ROOT / "assets" / "style.css").open(encoding="utf-8") as stylesheet:
    st.markdown(f"<style>{stylesheet.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="brand-symbol">M</div>
      <div>
        <div class="brand-name">MorphoRT</div>
        <div class="brand-caption">ATELIER DE MORPHOLOGIE</div>
      </div>
    </div>
    <div class="sidebar-description">
      De l'image IRM à la simulation, étape par étape.
    </div>
    <div class="sidebar-section-label">VOTRE PARCOURS</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-masthead">
      <div>
        <div class="masthead-eyebrow">ESPACE DE TRAVAIL · IMAGERIE CÉRÉBRALE</div>
        <div class="masthead-title">Exploration morphologique</div>
      </div>
      <div class="local-badge"><span></span> Traitement local</div>
    </div>
    """,
    unsafe_allow_html=True,
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
