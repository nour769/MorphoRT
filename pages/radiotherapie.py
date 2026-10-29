"""Page 5 : comparaison pédagogique de plans de radiothérapie simulés."""

import pandas as pd
import streamlit as st

from morphort.radiotherapy import (
    biologically_effective_dose,
    linear_quadratic_survival,
    simulate_plans,
)
from pages.common import show_data_origin


def afficher() -> None:
    """Affiche les modèles LQ et optimise deux scénarios non cliniques."""
    st.title("Simuler la radiothérapie")
    show_data_origin()
    st.error(
        "Simulation pédagogique uniquement. Les doses et courbes ne sont "
        "pas des recommandations ni des plans de traitement.",
        icon="⚠️",
    )
    observations = st.session_state.get("morphort_growth_observations", [])
    growth_model = st.session_state.get("morphort_growth_model")
    if len(observations) < 3 or growth_model is None:
        st.info(
            "Ajustez d'abord un modèle sur au moins trois examens dans "
            "l'étape Croissance."
        )
        return

    st.latex(r"S = \exp[-(\alpha d + \beta d^2)]")
    st.latex(
        r"\mathrm{BED}_{sain} = \sum_i d_i"
        r"\left(1+\frac{d_i}{(\alpha/\beta)_{sain}}\right)"
    )
    st.caption(
        "Le plan fixe est calculé une fois. Le scénario adaptatif réestime "
        "le modèle à des contrôles d'IRM hypothétiques."
    )
    columns = st.columns(2)
    with columns[0]:
        alpha = st.number_input("alpha (1/Gy)", min_value=0.001, value=0.1)
        beta = st.number_input("beta (1/Gy²)", min_value=0.001, value=0.03)
        healthy_alpha_beta = st.number_input(
            "alpha/beta sain (Gy)", min_value=0.1, value=3.0
        )
    with columns[1]:
        bed_limit = st.number_input("Limite BED saine (Gy)", min_value=1.0, value=60.0)
        fractions = st.number_input(
            "Nombre de fractions simulées",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )
        st.selectbox("Modèle de croissance utilisé", (growth_model,), disabled=True)

    if st.button("Simuler et comparer les scénarios", type="primary"):
        try:
            results = simulate_plans(
                observations,
                growth_model,
                alpha,
                beta,
                healthy_alpha_beta,
                bed_limit,
                int(fractions),
            )
            st.session_state["morphort_radiotherapy_results"] = results
            st.session_state["morphort_radiotherapy_parameters"] = {
                "alpha": alpha,
                "beta": beta,
                "healthy_alpha_beta": healthy_alpha_beta,
                "bed_limit": bed_limit,
                "fractions": int(fractions),
                "model": growth_model,
            }
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

    results = st.session_state.get("morphort_radiotherapy_results")
    if not results:
        st.info("Lancez la simulation pour comparer les deux scénarios.")
        return

    fixed_bed = float(results["fixed_bed_gy"])
    adaptive_bed = float(results["adaptive_bed_gy"])
    columns = st.columns(2)
    columns[0].metric("BED saine — plan fixe", f"{fixed_bed:.2f} Gy")
    columns[1].metric("BED saine — adaptatif", f"{adaptive_bed:.2f} Gy")
    st.caption(
        "Les deux BED doivent rester sous la limite indiquée. "
        "Le modèle LQ ne décrit pas l'ensemble de la biologie tumorale."
    )

    fixed = pd.DataFrame(results["fixed"]).set_index("fraction")
    adaptive = pd.DataFrame(results["adaptive"]).set_index("fraction")
    chart = pd.DataFrame(
        {
            "Fixe": fixed["volume_ml"],
            "Adaptatif": adaptive["volume_ml"],
        }
    )
    st.line_chart(chart, y_label="Volume tumoral simulé (mL)")

    dose_table = pd.DataFrame(
        {
            "Fraction": range(
                1,
                int(st.session_state["morphort_radiotherapy_parameters"]["fractions"])
                + 1,
            ),
            "Dose fixe (Gy)": results["fixed_doses"],
            "Dose adaptative (Gy)": results["adaptive_doses"],
        }
    )
    st.dataframe(dose_table, hide_index=True, use_container_width=True)
    verified_bed = biologically_effective_dose(
        results["fixed_doses"], healthy_alpha_beta
    )
    st.caption(
        f"Survie LQ d'une fraction de 2 Gy : "
        f"{linear_quadratic_survival(2.0, alpha, beta):.3f}. "
        f"BED fixe vérifiée : {verified_bed:.2f} Gy."
    )
