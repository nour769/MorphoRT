"""Page 4 : ajustement de Gompertz et du modèle logistique."""

import pandas as pd
import streamlit as st

from morphort.growth import fit_growth, simulate_gompertz_followup
from morphort.measurements import measure_mask
from pages.common import show_data_origin


def afficher() -> None:
    """Ajuste les modèles sur les examens saisis par l'utilisateur."""
    st.title("Étudier la croissance")
    show_data_origin()
    is_brats = st.session_state.get("morphort_is_brats", False)
    if is_brats:
        st.warning(
            "Suivi simulé à partir d'un volume réel BraTS, pas de données "
            "longitudinales",
            icon="⚠️",
        )
        case_id = st.session_state.get("morphort_brats_case_id", "BraTS")
        expert_masks = st.session_state.get("morphort_expert_masks", {})
        reference_mask = expert_masks.get("Tumeur entière")
        volumes = st.session_state.get("morphort_volumes", {})
        if reference_mask is None or not volumes:
            st.info("Chargez un cas BraTS avant de générer le suivi simulé.")
            return
        spacing = next(iter(volumes.values()))[1]
        initial_volume = measure_mask(reference_mask, spacing)["volume_ml"]
        if initial_volume <= 0:
            st.error(
                "Le masque expert BraTS est vide ; le suivi ne peut pas commencer."
            )
            return

        st.write(
            f"Patient {case_id} · volume initial du masque expert : "
            f"{initial_volume:.2f} mL."
        )
        parameter_columns = st.columns(3)
        with parameter_columns[0]:
            true_rate = st.number_input(
                "r simulé (par jour)",
                min_value=0.0001,
                max_value=1.0,
                value=0.025,
                step=0.005,
                format="%.4f",
            )
        with parameter_columns[1]:
            true_capacity = st.number_input(
                "K simulé (mL)",
                min_value=initial_volume * 1.001,
                value=max(initial_volume * 2.5, initial_volume + 0.01),
                step=max(initial_volume * 0.1, 0.01),
                format="%.3f",
            )
        with parameter_columns[2]:
            noise = st.number_input(
                "Bruit relatif simulé",
                min_value=0.0,
                max_value=0.5,
                value=0.03,
                step=0.01,
                format="%.2f",
            )
        if st.button("Générer le suivi et ajuster les modèles", type="primary"):
            days = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
            try:
                observations = simulate_gompertz_followup(
                    initial_volume,
                    true_rate,
                    true_capacity,
                    days,
                    noise,
                )
                results = {
                    name: fit_growth(observations, name)
                    for name in ("Gompertz", "Logistique")
                }
                st.session_state["morphort_growth_observations"] = observations
                st.session_state["morphort_growth_results"] = results
                st.session_state["morphort_growth_model"] = "Gompertz"
                st.session_state["morphort_growth_truth"] = {
                    "r": true_rate,
                    "K": true_capacity,
                    "noise": noise,
                    "v0": initial_volume,
                    "case_id": case_id,
                }
                st.success("Suivi simulé généré ; les deux modèles ont été ajustés.")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
    else:
        st.write(
            "Saisissez le délai depuis le premier examen et le volume segmenté en mL."
        )
        existing = st.session_state.get("morphort_growth_observations")
        if existing is None:
            initial = pd.DataFrame(
                [{"Jour": 0.0, "Volume (mL)": 0.0} for _ in range(3)]
            )
        else:
            initial = pd.DataFrame(
                [{"Jour": day, "Volume (mL)": volume} for day, volume in existing]
            )
        edited = st.data_editor(
            initial,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Jour": st.column_config.NumberColumn("Jour", min_value=0.0),
                "Volume (mL)": st.column_config.NumberColumn(
                    "Volume (mL)", min_value=0.0, format="%.4f"
                ),
            },
            key="growth_table",
        )
        model = st.selectbox("Modèle", ("Gompertz", "Logistique"))
        if st.button("Ajuster la croissance", type="primary"):
            try:
                observations = [
                    (float(row["Jour"]), float(row["Volume (mL)"]))
                    for _, row in edited.iterrows()
                    if pd.notna(row["Jour"]) and pd.notna(row["Volume (mL)"])
                ]
                if len(observations) < 3:
                    raise ValueError("Saisissez au moins trois examens complets.")
                if len(observations) != len(edited):
                    raise ValueError("Chaque ligne doit avoir un jour et un volume.")

                results = {
                    name: fit_growth(observations, name)
                    for name in ("Gompertz", "Logistique")
                }
                st.session_state["morphort_growth_observations"] = observations
                st.session_state["morphort_growth_results"] = results
                st.session_state["morphort_growth_model"] = model
                st.session_state.pop("morphort_growth_truth", None)
                st.success("Les deux modèles ont été ajustés.")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))

    results = st.session_state.get("morphort_growth_results")
    if not results:
        st.info("Au moins trois examens sont nécessaires pour ajuster les modèles.")
        return

    model_options = ("Gompertz", "Logistique")
    selected_model = st.selectbox(
        "Résultats affichés",
        model_options,
        index=model_options.index(
            st.session_state.get("morphort_growth_model", "Gompertz")
        ),
    )
    st.session_state["morphort_growth_model"] = selected_model
    selected = results[selected_model]
    columns = st.columns(2)
    columns[0].metric("Taux r (par jour)", f"{selected['r']:.5f}")
    columns[1].metric("Capacité limite K (mL)", f"{selected['K']:.3f}")
    st.caption(
        f"Intervalle indicatif de r : [{selected['ci_r'][0]:.5f}, "
        f"{selected['ci_r'][1]:.5f}] · "
        f"K : [{selected['ci_K'][0]:.3f}, {selected['ci_K'][1]:.3f}] mL"
    )

    observed = results["Gompertz"]
    chart = pd.DataFrame(
        {
            "Gompertz": observed["curve_volumes"],
            "Logistique": results["Logistique"]["curve_volumes"],
        },
        index=observed["curve_days"],
    )
    chart.index.name = "Jour"
    st.line_chart(chart, y_label="Volume simulé (mL)")
    st.caption(
        "Les courbes sont des ajustements mathématiques illustratifs, "
        "pas des prévisions médicales."
    )
    truth = st.session_state.get("morphort_growth_truth")
    if truth:
        comparison = pd.DataFrame(
            [
                {
                    "Modèle": name,
                    "r ajusté": results[name]["r"],
                    "K ajusté (mL)": results[name]["K"],
                    "Écart r (%)": 100
                    * (float(results[name]["r"]) - truth["r"])
                    / truth["r"],
                    "Écart K (%)": 100
                    * (float(results[name]["K"]) - truth["K"])
                    / truth["K"],
                }
                for name in model_options
            ]
        )
        st.subheader("Paramètres simulés et retrouvés")
        st.caption(
            f"Vérités de la simulation : r={truth['r']:.4f}/jour, "
            f"K={truth['K']:.3f} mL, bruit={truth['noise']:.0%}."
        )
        st.dataframe(comparison, hide_index=True, use_container_width=True)
