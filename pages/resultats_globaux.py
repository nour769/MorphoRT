"""Résultats de l'évaluation globale sur les patients BraTS préparés."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "brats_eval.csv"
METRIC_COLUMNS = ("dice", "jaccard", "hausdorff95_mm")
METRIC_LABELS = {
    "dice": "Dice",
    "jaccard": "Jaccard",
    "hausdorff95_mm": "Hausdorff 95 (mm)",
}
TUNING_GRID_FILE = RESULTS_FILE.parent / "brats_tuning_grid.csv"
TUNING_PATIENTS_FILE = RESULTS_FILE.parent / "brats_tuning_patients.csv"


def _summary_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques sans convertir une distance infinie en NaN."""
    summary: dict[str, dict[str, float]] = {}
    for metric in values.columns:
        scores = values[metric].to_numpy(dtype=np.float64)
        if np.isnan(scores).any():
            raise ValueError(f"La métrique {metric} contient une valeur NaN.")
        if np.isposinf(scores).any():
            summary[metric] = {
                "Moyenne": float("inf"),
                "Écart-type": float("inf"),
                "Minimum": float(np.min(scores)),
                "Maximum": float("inf"),
            }
        else:
            summary[metric] = {
                "Moyenne": float(np.mean(scores)),
                "Écart-type": float(np.std(scores, ddof=1)),
                "Minimum": float(np.min(scores)),
                "Maximum": float(np.max(scores)),
            }
    return pd.DataFrame(summary)


def afficher() -> None:
    """Affiche les scores bruts, les statistiques et les distributions."""
    st.title("Résultats globaux")
    st.caption("Origine des données évaluées : volumes et masques BraTS2020.")
    st.write(
        "Évaluation de la tumeur entière par la chaîne morphologique actuelle. "
        "Les cas et scores sont affichés tels quels, y compris les résultats faibles."
    )
    if not RESULTS_FILE.is_file():
        st.info(
            "Aucun CSV d'évaluation n'a encore été produit. Depuis la racine du "
            "projet, lancez `python -m scripts.evaluate_brats`."
        )
        return

    results = pd.read_csv(RESULTS_FILE)
    required_columns = {"patient", *METRIC_COLUMNS}
    if not required_columns.issubset(results.columns):
        st.error(
            "Le CSV ne contient pas les colonnes attendues : "
            f"{', '.join(sorted(required_columns))}."
        )
        return
    if results.empty:
        st.error("Le CSV d'évaluation ne contient aucun patient.")
        return

    st.caption(f"{len(results)} patients · résultats issus de {RESULTS_FILE.name}")
    try:
        summary = _summary_frame(results[list(METRIC_COLUMNS)])
    except ValueError as error:
        st.error(str(error))
        return
    st.subheader("Statistiques descriptives")
    st.dataframe(
        summary.rename(
            index={"Écart-type": "Écart-type (échantillonnal)"},
            columns=METRIC_LABELS,
        ).round(4),
        width="stretch",
    )

    st.subheader("Distribution des scores")
    long_results = results.melt(
        id_vars="patient",
        value_vars=list(METRIC_COLUMNS),
        var_name="Métrique",
        value_name="Valeur",
    )
    long_results["Métrique"] = long_results["Métrique"].map(METRIC_LABELS)
    non_finite_count = int((~np.isfinite(long_results["Valeur"])).sum())
    if non_finite_count:
        st.info(
            f"{non_finite_count} mesure(s) non finie(s) restent affichées dans "
            "le tableau et le CSV ; elles ne sont pas représentables dans un "
            "boxplot."
        )
        long_results = long_results[np.isfinite(long_results["Valeur"])]
    st.vega_lite_chart(
        long_results,
        {
            "facet": {
                "column": {"field": "Métrique", "type": "nominal", "title": None}
            },
            "spec": {
                "mark": {"type": "boxplot"},
                "encoding": {
                    "y": {
                        "field": "Valeur",
                        "type": "quantitative",
                        "scale": {"zero": False},
                    }
                },
            },
            "resolve": {"scale": {"y": "independent"}},
        },
        width="stretch",
    )

    st.subheader("Patients triés du Dice le plus faible au plus élevé")
    sorted_results = results.sort_values("dice", ascending=True, kind="stable")
    st.dataframe(
        sorted_results.rename(columns=METRIC_LABELS).round(4),
        hide_index=True,
        width="stretch",
    )
    st.session_state["morphort_evaluation_case_ids"] = sorted_results[
        "patient"
    ].tolist()
    from pages.navigation import SEGMENTATION_PAGE

    st.page_link(
        SEGMENTATION_PAGE,
        label="Inspecter les cas faibles sur la page 2 · Segmentation",
        help=(
            "La page 2 propose les patients évalués triés du Dice le plus faible "
            "au plus élevé."
        ),
    )
    if not TUNING_GRID_FILE.is_file() or not TUNING_PATIENTS_FILE.is_file():
        return

    st.divider()
    st.header("Réglage défendable des paramètres")
    st.write(
        "Les 20 patients sont séparés de façon reproductible après tri : indices "
        "pairs pour le réglage, indices impairs pour le test indépendant. Le Dice "
        "moyen des 10 cas de réglage choisit les paramètres ; les 10 autres ne "
        "servent qu'à mesurer leur performance."
    )
    tuning_grid = pd.read_csv(TUNING_GRID_FILE)
    tuning_patients = pd.read_csv(TUNING_PATIENTS_FILE)
    expected_patient_columns = {
        "split",
        "patient",
        "opening_radius_mm",
        "closing_radius_mm",
        "threshold_method",
        *METRIC_COLUMNS,
        "status",
    }
    if not expected_patient_columns.issubset(tuning_patients.columns):
        st.error(
            "Le CSV de réglage des patients ne contient pas les colonnes attendues."
        )
        return
    expected_grid_columns = {
        "opening_radius_mm",
        "closing_radius_mm",
        "threshold_method",
        "dice_mean",
        "dice_std",
    }
    if not expected_grid_columns.issubset(tuning_grid.columns):
        st.error("Le CSV de grille ne contient pas les colonnes attendues.")
        return

    best = tuning_grid.sort_values(
        ["dice_mean", "opening_radius_mm", "closing_radius_mm", "threshold_method"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]
    st.caption(
        "Paramètres retenus : ouverture "
        f"{best['opening_radius_mm']:g} mm · fermeture "
        f"{best['closing_radius_mm']:g} mm · seuil "
        f"{best['threshold_method']}"
    )
    st.subheader("Grille évaluée sur les 10 patients de réglage")
    st.dataframe(
        tuning_grid.rename(columns=METRIC_LABELS).round(4),
        hide_index=True,
        width="stretch",
    )
    for split_name, title in (
        ("réglage", "Résultats sur les 10 patients de réglage"),
        ("test_indépendant", "Résultats sur les 10 patients indépendants"),
    ):
        split_results = tuning_patients[tuning_patients["split"] == split_name]
        if split_results.empty:
            st.warning(f"Le CSV ne contient aucun résultat pour « {title} ».")
            continue
        st.subheader(title)
        try:
            score_summary = _summary_frame(
                split_results[list(METRIC_COLUMNS)]
            )
        except ValueError as error:
            st.error(str(error))
            return
        st.dataframe(
            score_summary.rename(columns=METRIC_LABELS).round(4),
            width="stretch",
        )
        st.dataframe(
            split_results.sort_values("dice", kind="stable")
            .rename(columns=METRIC_LABELS)
            .round(4),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    afficher()
