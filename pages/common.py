"""Composants communs aux pages Streamlit."""

import streamlit as st


def show_data_origin() -> None:
    """Affiche clairement l'origine des données de l'examen courant."""
    if st.session_state.get("morphort_is_brats"):
        case_id = st.session_state.get("morphort_brats_case_id", "")
        st.caption(f"Origine des données : Volume réel BraTS2020 · {case_id}")
    elif st.session_state.get("morphort_is_demo"):
        st.caption("Origine des données : Données synthétiques de démonstration")
    elif st.session_state.get("morphort_volumes"):
        st.caption("Origine des données : Volume NIfTI importé localement")
    else:
        st.caption("Origine des données : aucune image chargée")
