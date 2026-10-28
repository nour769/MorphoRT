"""Page 1 : chargement et aperçu de l'IRM."""

import streamlit as st


def afficher() -> None:
    """Affiche le point d'entrée du parcours d'analyse."""
    st.title("Charger l'IRM")
    st.info("Le chargement des volumes sera ajouté à la phase 3.")
