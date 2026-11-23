from __future__ import annotations

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import psycopg
import json
from src.update_neuron import to_sd
from src.ml_pipeline import (
    FEATURE_COLS,
    build_classifier,
    compute_literature_zscores,
    evaluate_cv,
    prepare_single_cell_data,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from identity import (
    get_canonical_identities,
    harmonize_identity_series,
    harmonize_identity_value,
)

# ==========================================
# DICTIONNAIRE DE TRADUCTION (FR / EN)
# ==========================================
LANGUAGES = {
    "Français": {
        "page_title": "Neuro-Electrophys Database",
        "sidebar_header": "PARAMÈTRES",
        "sidebar_desc": "Filtres d'exploration de la base locale.",
        "regions_label": "Région(s) cérébrale(s) :",
        "sources_label": "Source(s) des données :",
        "metric_selected": "Neurones sélectionnés",
        "main_title": "NEURO-ELECTROPHY DATABASE",
        "main_subtitle": "> **Database interactive, collaborative & comparative** — Tranches aiguës VS Cultures organotypiques VS Cultures cellulaires",
        "tabs": [
            "01 // DATABASE", 
            "02 // ACP", 
            "03 // RANDOM FOREST", 
            "04 // INSERTION",
            "05 // MODÉLISATION IN SILICO"
        ],
        # --- Onglet 1 ---
        "t1_sub": "Aperçu du jeu de données filtré",
        "t1_m1": "Total neurones",
        "t1_m2": "Préparations",
        "t1_m3": "Identités connues",
        "t1_dl": "Télécharger le jeu filtré (.csv)",
        # --- Onglet 2 ---
        "t2_title": "PROJECTION MULTIDIMENSIONNELLE",
        "t2_sub": "Analyse en Composantes Principales (PCA)",
        "t2_warn": "Sélectionnez au moins 5 neurones dans les filtres pour calculer l'ACP.",
        "t2_c1": "Colorer par :",
        "t2_c2": "Symbole par :",
        # --- Onglet 3 ---
        "t3_sub": "Simulateur de classification phénotypique",
        "t3_desc": "Estimation de l'identité neurochimique via Random Forest sur les paramètres électrophysiologiques.",
        "t3_err": "Pas assez de neurones étiquetés dans la base pour entraîner le modèle.",
        "t3_pred": "PRÉDICTION",
        "t3_prob": "PROBABILITÉ SÉROTONERGIQUE",
        "t3_exp": "Voir les sources bibliographiques et DOI des données d'entraînement du modèle",
        "t3_rmp": "RMP (mV)", "t3_tau": "Tau (ms)", "t3_cm": "Capacitance Cm (pF)",
        "t3_rheo": "Rhéobase (pA)", "t3_thr": "Seuil du PA (mV)", "t3_amp": "Amplitude du PA (mV)",
        "t3_hw": "Demi-largeur PA (ms)", "t3_ahp_amp": "Amplitude AHP (mV)", "t3_ahp_dur": "Durée AHP (ms)",
        "t3_conf": "Confiance",
        "t3_proba_chart": "Probabilités par classe prédite (%)",
        "t3_feat_imp": "Importance des variables dans le modèle",
        "t3_class_col": "Classe",
        "t3_proba_col": "Probabilité (%)",
        "t3_feat_col": "Paramètre",
        "t3_imp_col": "Importance",
        "t3_n_single": "n single-cell",
        "t3_f1_cv": "F1 macro (CV)",
        "t3_z_title": "Z-scores vs littérature (`group_summary`)",
        "t3_z_empty": "Aucun profil `group_summary` disponible dans la base pour le moment.",
        "t3_z_chart_title": "Écarts aux profils de littérature (seuils ±2 SD)",
        "t3_z_axis_x": "z-score (SD)",
        "t3_z_status": "Écart",
        "t3_z_out": "|z| > 2 SD",
        "t3_z_in": "Dans ±2 SD",
        "t3_z_dl": "Télécharger les z-scores (.csv)",
        "t3_col_id": "Identité",
        "t3_col_prob": "Probabilité",
        "t3_col_param": "Paramètre",
        "t3_col_profile": "Profil",
        "t3_col_pp": "Profil / Paramètre",
        # --- Onglet 4 ---
        "t4_sub": "Encoder un nouvel enregistrement",
        "t4_desc": "Injection directe d'une cellule dans la base de données SQLite locale.",
        "t4_id": "ID de la cellule (ex: Lab_N260)*",
        "t4_source": "Source",
        "t4_region": "Région cérébrale",
        "t4_prep": "Préparation",
        "t4_age": "Âge ou DIV",
        "t4_ident": "Identité connue",
        "t4_btn": "ENREGISTRER DANS LA BASE",
        "t4_err_id": "L'ID de la cellule est obligatoire !",
        "t4_success": "Le neurone **{}** a été ajouté avec succès !",
        "t4_rheo": "Rhéobase (pA)", "t4_thr": "Seuil PA (mV)", "t4_amp": "Amplitude PA (mV)",
        "t4_hw": "Demi-largeur PA (ms)", "t4_ahp_amp": "Amplitude AHP (mV)", "t4_ahp_dur": "Durée AHP (ms)",
        "t4_mode_label": "Action dans cet onglet",
        "t4_mode_add": "Ajouter un nouveau neurone",
        "t4_mode_update": "Compléter / mettre à jour un neurone existant",
        "t4_select_cell": "Sélectionner un neurone existant (cell_id)",
        "t4_no_cells": "Aucun neurone trouvé dans la base pour la mise à jour.",
        "t4_update_btn": "Enregistrer les modifications",
        "t4_update_success": "Neurone '{0}' mis à jour avec succès.",
        "t4_update_noop": "Aucune modification détectée pour '{0}'.",
        "t4_morpho_section": "Paramètres morphologiques (optionnels)",
        "t4_num_branches": "Nombre de branches (num_branches)",
        "t4_num_junctions": "Nombre de jonctions (num_junctions)",
        "t4_max_branch_len": "Longueur max. de branche (µm)",
        "t4_crit_radius": "Rayon critique (µm)",
        "t4_sholl": "Intersections de Sholl",

        # --- Onglet 5 (Modélisation) ---
        "t5_header": "Calibrage & Modélisation In Silico (LIF / Hodgkin-Huxley)",
        "t5_desc": "Extrayez les paramètres biophysiques validés par la littérature (neurones natifs ou systèmes d'expression WT/Mutants) pour construire vos modèles computationnels et réduire le recours à l'expérimentation animale.",
        "t5_reg": "Région / Système :",
        "t5_prep": "Préparation :",
        "t5_id": "Identité / Sous-type :",
        "t5_chan": "Canal ciblé :",
        "t5_geno": "Statut (WT / Mutant) :",
        "t5_mut": "Mutation spécifique :",
        "t5_count": "**{} enregistrements** retenus pour le calcul des paramètres biophysiques.",
        "t5_warn": "Aucune donnée ne correspond à cette combinaison de filtres.",
        "t5_sec1": "1. Propriétés passives & globales (Médiane ± SD)",
        "t5_sec2": "2. Conductances & Cinétiques des canaux ioniques",
        "t5_sec3": "3. Sources bibliographiques & DOI des données sélectionnées",
        "t5_orig": "**Origine des données (Sources) :** {}",
        "t5_dois": "**Liens DOI associés :**",
        "t5_nodoi": "Aucun DOI spécifique n'a été renseigné pour cette sélection de cellules.",
        "t5_expand": "Voir la liste des identifiants (cell_ids) inclus dans cette extraction",
        "t5_capt": "Pensez à citer ces sources et DOI d'origine si vous utilisez ces paramètres dans une publication.",
        "t5_sec4": "4. Script de simulation & Aperçu en direct",
        "t5_form": "Choisir le type de modèle :",
        "t5_btn_sim": "Simuler un aperçu en direct (LIF - Intégration d'Euler)",
        "t5_sim_ok": "Simulation numérique exécutée avec succès à partir des paramètres extraits !",
        "t5_dl_py": "Télécharger le script Python (.py)",
        "t5_dl_json": "Télécharger le profil JSON / NeuroML (.json)",
    },
    "English": {
        "page_title": "Neuro-Electrophys Database",
        "sidebar_header": "PARAMETERS",
        "sidebar_desc": "Local database exploration filters.",
        "regions_label": "Brain region(s):",
        "sources_label": "Data source(s):",
        "metric_selected": "Selected neurons",
        "main_title": "NEURO-ELECTROPHY DATABASE",
        "main_subtitle": "> **Comparative, interactive, and collaborative** — Acute slices VS Organotypic cultures VS Cell cultures",
        "tabs": [
            "01 // DATABASE", 
            "02 // PCA", 
            "03 // RANDOM FOREST", 
            "04 // INSERTION",
            "05 // IN SILICO MODELING"
        ],
        # --- Tab 1 ---
        "t1_sub": "Filtered Dataset Overview",
        "t1_m1": "Total Neurons",
        "t1_m2": "Preparations",
        "t1_m3": "Known Identities",
        "t1_dl": "Download Filtered Dataset (CSV)",
        # --- Tab 2 ---
        "t2_title": "MULTIDIMENSIONAL PROJECTION",
        "t2_sub": "Principal Component Analysis (PCA)",
        "t2_warn": "Select at least 5 neurons in the filters to compute PCA.",
        "t2_c1": "Color by:",
        "t2_c2": "Symbol by:",
        # --- Tab 3 ---
        "t3_sub": "Phenotypic Classification Simulator",
        "t3_desc": "Estimation of neurochemical identity via Random Forest on electrophysiological parameters.",
        "t3_err": "Not enough labeled neurons in the database to train the model.",
        "t3_pred": "PREDICTION",
        "t3_prob": "SEROTONERGIC PROBABILITY",
        "t3_exp": "View bibliographic sources and DOIs of the training data",
        "t3_rmp": "RMP (mV)", "t3_tau": "Tau (ms)", "t3_cm": "Capacitance Cm (pF)",
        "t3_rheo": "Rheobase (pA)", "t3_thr": "AP Threshold (mV)", "t3_amp": "AP Amplitude (mV)",
        "t3_hw": "AP Half-width (ms)", "t3_ahp_amp": "AHP Amplitude (mV)", "t3_ahp_dur": "AHP Duration (ms)",
        "t3_conf": "Confidence",
        "t3_proba_chart": "Predicted probability per class (%)",
        "t3_feat_imp": "Feature importance in the model",
        "t3_class_col": "Class",
        "t3_proba_col": "Probability (%)",
        "t3_feat_col": "Feature",
        "t3_imp_col": "Importance",
        "t3_n_single": "n single-cell",
        "t3_f1_cv": "Macro F1 (CV)",
        "t3_z_title": "Z-scores vs literature (`group_summary`)",
        "t3_z_empty": "No `group_summary` profile available in the database yet.",
        "t3_z_chart_title": "Deviations from literature profiles (±2 SD thresholds)",
        "t3_z_axis_x": "z-score (SD)",
        "t3_z_status": "Deviation",
        "t3_z_out": "|z| > 2 SD",
        "t3_z_in": "Within ±2 SD",
        "t3_z_dl": "Download z-scores (.csv)",
        "t3_col_id": "Identity",
        "t3_col_prob": "Probability",
        "t3_col_param": "Parameter",
        "t3_col_profile": "Profile",
        "t3_col_pp": "Profile / Parameter",
        # --- Tab 4 ---
        "t4_sub": "Encode a New Recording",
        "t4_desc": "Direct injection of a cell into the local SQLite database.",
        "t4_id": "Cell ID (e.g., Lab_N260)*",
        "t4_source": "Source",
        "t4_region": "Brain region",
        "t4_prep": "Preparation",
        "t4_age": "Age or DIV",
        "t4_ident": "Known identity",
        "t4_btn": "SAVE TO DATABASE",
        "t4_err_id": "Cell ID is mandatory!",
        "t4_success": "Neuron **{}** successfully added!",
        "t4_rheo": "Rheobase (pA)", "t4_thr": "AP Threshold (mV)", "t4_amp": "AP Amplitude (mV)",
        "t4_hw": "AP Half-width (ms)", "t4_ahp_amp": "AHP Amplitude (mV)", "t4_ahp_dur": "AHP Duration (ms)",
        "t4_mode_label": "Action in this tab",
        "t4_mode_add": "Add a new neuron",
        "t4_mode_update": "Enrich / update an existing neuron",
        "t4_select_cell": "Select an existing neuron (cell_id)",
        "t4_no_cells": "No neurons found in the database to update.",
        "t4_update_btn": "Save updates",
        "t4_update_success": "Neuron '{0}' successfully updated.",
        "t4_update_noop": "No changes detected for '{0}'.",
        "t4_morpho_section": "Morphological parameters (optional)",
        "t4_num_branches": "Number of branches (num_branches)",
        "t4_num_junctions": "Number of junctions (num_junctions)",
        "t4_max_branch_len": "Max branch length (µm)",
        "t4_crit_radius": "Critical radius (µm)",
        "t4_sholl": "Sholl intersections",
        # --- Tab 5 (Modeling) ---
        "t5_header": "In Silico Calibration & Modeling (LIF / Hodgkin-Huxley)",
        "t5_desc": "Extract biophysical parameters validated by literature (native neurons or WT/Mutant expression systems) to build computational models and reduce animal experimentation.",
        "t5_reg": "Region / System:",
        "t5_prep": "Preparation:",
        "t5_id": "Identity / Subtype:",
        "t5_chan": "Target Channel:",
        "t5_geno": "Status (WT / Mutant):",
        "t5_mut": "Specific Mutation:",
        "t5_count": "**{} records** selected for biophysical parameter calculation.",
        "t5_warn": "No data matches this combination of filters.",
        "t5_sec1": "1. Passive & Global Properties (Median ± SD)",
        "t5_sec2": "2. Ion Channel Conductances & Kinetics",
        "t5_sec3": "3. Bibliographic Sources & DOIs of Selected Data",
        "t5_orig": "**Data Origin (Sources):** {}",
        "t5_dois": "**Associated DOI Links:**",
        "t5_nodoi": "No specific DOI was provided for this cell selection.",
        "t5_expand": "View the list of cell_ids included in this extraction",
        "t5_capt": "Remember to cite these original sources and DOIs if you use these parameters in a publication.",
        "t5_sec4": "4. Simulation Script & Live Preview",
        "t5_form": "Choose model type:",
        "t5_btn_sim": "Simulate Live Preview (LIF - Euler Integration)",
        "t5_sim_ok": "Numerical simulation successfully executed from extracted parameters!",
        "t5_dl_py": "Download Python Script (.py)",
        "t5_dl_json": "Download JSON / NeuroML Profile (.json)", 
    }
}
RF_TRANSLATIONS = {
    "fr": {
        "rf_title": "Classification neuronale multi-classe (Random Forest)",
        "rf_subtitle": "Entraînement dynamique sur les neurones étiquetés avec estimation de confiance par classe.",
        "min_samples_warning": "Il faut au moins 2 classes distinctes et suffisamment d'exemples étiquetés pour entraîner le modèle multi-classe.",
        "n_estimators_label": "Nombre d'arbres (n_estimators)",
        "train_n_caption": "Neurones d'entraînement retenus après harmonisation : n = {n} ({classes})",
        "target_neuron_select": "Sélectionner un neurone à prédire",
        "predicted_class_label": "Identité prédite",
        "confidence_label": "Score de confiance (probabilité max)",
        "proba_chart_title": "Distribution des probabilités prédites par classe",
        "feature_imp_title": "Importance des variables électrophysiologiques / morphologiques",
        "feature_col": "Variable",
        "importance_col": "Importance relative",
        "proba_col": "Probabilité",
        "class_col": "Identité canonique",
        "input_identity_label": "Identité connue (normalisée automatiquement)",
        "input_custom_identity": "Ou saisir une autre étiquette (sera harmonisée)",
    },
    "en": {
        "rf_title": "Multi-Class Neuronal Classification (Random Forest)",
        "rf_subtitle": "Dynamic training on labeled neurons with class-level confidence estimation.",
        "min_samples_warning": "At least 2 distinct classes and enough labeled examples are required to train the multi-class model.",
        "n_estimators_label": "Number of trees (n_estimators)",
        "train_n_caption": "Training neurons kept after harmonization: n = {n} ({classes})",
        "target_neuron_select": "Select a neuron to predict",
        "predicted_class_label": "Predicted identity",
        "confidence_label": "Confidence score (max probability)",
        "proba_chart_title": "Predicted Probability Distribution by Class",
        "feature_imp_title": "Electrophysiological / Morphological Feature Importance",
        "feature_col": "Feature",
        "importance_col": "Relative importance",
        "proba_col": "Probability",
        "class_col": "Canonical identity",
        "input_identity_label": "Known identity (auto-harmonized)",
        "input_custom_identity": "Or enter another label (will be harmonized)",
    },
}


# Configuration de la page
st.set_page_config(
    page_title="Neuro-Electrophys Database",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# DESIGN SYSTEM : BRUTALISTE & BLEU KLEIN
# ==========================================
st.markdown("""
<style>
    /* Fond général gris clair */
    .stApp {
        background-color: #F4F4F4;
        color: #111111;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Barre latérale brute avec bordure marquée */
    [data-testid="stSidebar"] {
        background-color: #EBEBEB;
        border-right: 2px solid #111111;
    }
    
    /* Titres et en-têtes tranchés */
    h1, h2, h3 {
        color: #002FA7 !important;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Boutons style brutaliste (Blocs pleins, bordures nettes) */
    .stButton>button, .stFormSubmitButton>button {
        background-color: #002FA7 !important;
        color: #FFFFFF !important;
        border: 2px solid #111111 !important;
        border-radius: 0px !important;
        font-weight: 700;
        box-shadow: 3px 3px 0px #111111;
        transition: all 0.1s ease;
    }
    
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #111111 !important;
        box-shadow: 1px 1px 0px #002FA7;
        transform: translate(2px, 2px);
    }
    
    /* Bouton de téléchargement */
    .stDownloadButton>button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 2px solid #002FA7 !important;
        border-radius: 0px !important;
        font-weight: 700;
        box-shadow: 3px 3px 0px #002FA7;
    }
    
    /* Conteneurs de métriques et boîtes d'information */
    [data-testid="stMetric"], .stAlert {
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 0px;
        box-shadow: 4px 4px 0px #111111;
        padding: 15px;
    }
    
    /* Onglets de navigation bruts */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E0E0E0;
        border: 2px solid #111111;
        border-radius: 0px;
        color: #111111;
        font-weight: 700;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #002FA7 !important;
        color: #FFFFFF !important;
    }
    
    /* Tableaux et dataframes */
    [data-testid="stDataFrame"] {
        border: 2px solid #111111;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "data/electrophys_database.db"

FEATURE_COLS = [
    'rmp_mv', 'tau_ms', 'cm_pf', 
    'rheobase_pa', 'ap_threshold_mv', 'ap_amplitude_mv', 
    'ap_half_width_ms', 'ahp_amplitude_mv', 'ahp_duration_ms'
]

# Fonction de chargement et nettoyage des données
DB_PATH = "data/electrophys_database.db"
PUBLIC_DB_PATH = "data/public_electrophys.db"

FEATURE_COLS = [
    'rmp_mv', 'tau_ms', 'cm_pf',
    'rheobase_pa', 'ap_threshold_mv', 'ap_amplitude_mv',
    'ap_half_width_ms', 'ahp_amplitude_mv', 'ahp_duration_ms'
]


def _get_supabase_url() -> str | None:
    try:
        url = st.secrets.get("SUPABASE_DB_URL", "")
        return url.strip() if url else None
    except Exception:
        return None


# Fonction de chargement et nettoyage des données
@st.cache_data(ttl=60)
def load_data():
    db_url = _get_supabase_url()
    df = None

    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as conn:
                df = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
        except Exception as exc:
            st.warning(f"Lecture Supabase indisponible ({exc}), repli sur SQLite local.")

    if df is None:
        conn = sqlite3.connect(PUBLIC_DB_PATH)
        try:
            df = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
        finally:
            conn.close()

    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


df = load_data()

# --- Fonctions utilistaires, helpers ---
@st.cache_resource(show_spinner=False)
def _fit_cached_classifier(
    df_fingerprint: tuple[tuple[str, object], ...],
    X_records: list[dict[str, float | None]],
    y_values: list[str],
):
    X_df = pd.DataFrame(X_records)
    y_ser = pd.Series(y_values)
    model = build_classifier()
    model.fit(X_df, y_ser)
    return model


def get_trained_model(df):
    X, y = prepare_single_cell_data(df)
    class_counts = y.value_counts()
    if len(y) < 4 or len(class_counts) < 2:
        return None, None, X, y

    fingerprint = tuple(
        (col, float(X[col].sum(skipna=True))) for col in X.columns
    ) + (("n_rows", len(y)),)
    model = _fit_cached_classifier(
        fingerprint,
        X.to_dict(orient="records"),
        y.tolist(),
    )
    return model, class_counts, X, y

# --- BARRE LATÉRALE (FILTRES GLOBAUX) ---
lang_choice = st.sidebar.selectbox("Langue / Language", ["Français", "English"])
t = LANGUAGES[lang_choice]  # 't' contient tous les textes de la langue choisie

st.sidebar.title(t["sidebar_header"])
st.sidebar.markdown("---")
st.sidebar.markdown(t["sidebar_desc"])

regions_disponibles = df['brain_region'].dropna().unique().tolist()
selected_regions = st.sidebar.multiselect(
    t["regions_label"],
    options=regions_disponibles,
    default=regions_disponibles
)

sources_disponibles = df['source'].dropna().unique().tolist()
selected_sources = st.sidebar.multiselect(
    t["sources_label"],
    options=sources_disponibles,
    default=sources_disponibles
)

df_filtered = df[
    (df['brain_region'].isin(selected_regions)) & 
    (df['source'].isin(selected_sources))
].copy()

st.sidebar.markdown("---")
st.sidebar.metric(t["metric_selected"], len(df_filtered))

# --- EN-TÊTE PRINCIPAL ---
st.title("NEURO-ELECTROPHY INTERACTIVE DATABASE")
st.markdown(t["main_subtitle"])
st.markdown("---")

# ==========================================
# FONCTION : VOLET MODÉLISATION IN SILICO
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_modeling_tab(df, t):
    st.header(t["t5_header"])
    st.markdown(t["t5_desc"])
    st.caption("DEBUG: onglet modélisation in silico uniquement")

    # --- 1. FILTRES BIOLOGIQUES & BIOPHYSIQUES ---
    col1, col2, col3 = st.columns(3)
    with col1:
        regions = ["Toutes" if lang_choice == "Français" else "All"] + sorted(df["brain_region"].dropna().unique().tolist())
        sel_region = st.selectbox(t["t5_reg"], regions, key="mod_reg")
    
    sub_df = df.copy()
    if sel_region not in ["Toutes", "All"]:
        sub_df = sub_df[sub_df["brain_region"] == sel_region]

    with col2:
        preps = ["Toutes" if lang_choice == "Français" else "All"] + sorted(sub_df["preparation"].dropna().unique().tolist())
        sel_prep = st.selectbox(t["t5_prep"], preps, key="mod_prep")
    if sel_prep not in ["Toutes", "All"]:
        sub_df = sub_df[sub_df["preparation"] == sel_prep]

    with col3:
        identities = ["Toutes" if lang_choice == "Français" else "All"] + sorted(sub_df["known_identity"].dropna().unique().tolist())
        sel_id = st.selectbox(t["t5_id"], identities, key="mod_id")
    if sel_id not in ["Toutes", "All"]:
        sub_df = sub_df[sub_df["known_identity"] == sel_id]

    # Filtres spécifiques canaux ioniques (WT vs Mutants) si les colonnes existent
    if "genotype_status" in sub_df.columns and "channel_target" in sub_df.columns:
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            channels = ["Tous" if lang_choice == "Français" else "All"] + sorted(sub_df["channel_target"].dropna().unique().tolist())
            sel_channel = st.selectbox(t["t5_chan"], channels, key="mod_chan")
        if sel_channel not in ["Tous", "All"]:
            sub_df = sub_df[sub_df["channel_target"] == sel_channel]

        with c_col2:
            genotypes = ["Tous" if lang_choice == "Français" else "All"] + sorted(sub_df["genotype_status"].dropna().unique().tolist())
            sel_geno = st.selectbox(t["t5_geno"], genotypes, key="mod_geno")
        if sel_geno not in ["Tous", "All"]:
            sub_df = sub_df[sub_df["genotype_status"] == sel_geno]

        with c_col3:
            mutants = ["Tous" if lang_choice == "Français" else "All"] + sorted(sub_df["mutation_details"].dropna().unique().tolist())
            sel_mut = st.selectbox(t["t5_mut"], mutants, key="mod_mut")
        if sel_mut not in ["Tous", "All"]:
            sub_df = sub_df[sub_df["mutation_details"] == sel_mut]

    st.info(t["t5_count"].format(len(sub_df)))

    if len(sub_df) == 0:
        st.warning(t["t5_warn"])
        return

    # --- 2. EXTRACTION INTELLIGENTE DES PARAMÈTRES ---
    def get_median_std(col):
        if col in sub_df.columns and not sub_df[col].dropna().empty:
            vals = pd.to_numeric(sub_df[col], errors="coerce").dropna()
            if not vals.empty:
                med = round(float(vals.median()), 2)
                sd = round(float(vals.std()), 2) if len(vals) > 1 else None
                if sd is not None and np.isnan(sd):
                    sd = None
                return med, sd
        return None, None

    rmp, rmp_sd = get_median_std("rmp_mv")
    rin, rin_sd = get_median_std("input_resistance_Mohm")
    tau, tau_sd = get_median_std("tau_ms")
    v_th, v_th_sd = get_median_std("ap_threshold_mv")
    ahp_amp, _ = get_median_std("ahp_amplitude_mv")
    ap_hw, _ = get_median_std("ap_half_width_ms")
    soma_area, _ = get_median_std("soma_area_um2")

    g_max, g_max_sd = get_median_std("g_max_ns")
    curr_dens, _ = get_median_std("current_density_pa_pf")
    gamma_sc, _ = get_median_std("single_channel_conductance_ps")
    v_half, v_half_sd = get_median_std("v_half_act_mv")
    slope_k, _ = get_median_std("slope_factor_k_mv")
    ec50_ca, _ = get_median_std("ec50_ca_um")

    cm, cm_sd = get_median_std("cm_pf")
    if cm is None:
        cm, cm_sd = get_median_std("capacitance_pF")

    rheo_pa, rheo_sd = get_median_std("rheobase_pa")
    if rheo_pa is None:
        rheo_na, rheo_na_sd = get_median_std("rheobase_nA")
        if rheo_na is not None:
            rheo_pa = round(rheo_na * 1000.0, 2)
            rheo_sd = round(rheo_na_sd * 1000.0, 2)

    E_L = rmp if rmp is not None else -65.0
    R_in = rin if rin is not None else 250.0
    tau_m = tau if tau is not None else 20.0
    V_th = v_th if v_th is not None else -40.0
    V_reset = round(V_th - (ahp_amp if ahp_amp is not None else 15.0), 2)
    I_rheo = rheo_pa if rheo_pa is not None else 50.0
    C_m = cm if cm is not None else round((tau_m / R_in) * 1000.0, 1)
    g_L = round(1000.0 / R_in, 3) if R_in > 0 else 4.0

    # --- 3. AFFICHAGE DES MÉTRIQUES ---
    st.subheader(t["t5_sec1"])
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("E_L (RMP)", f"{E_L} mV", f"± {rmp_sd}" if rmp_sd else "Default" if lang_choice == "English" else "Défaut")
    m2.metric("R_in", f"{R_in} MΩ", f"± {rin_sd}" if rin_sd else "Default" if lang_choice == "English" else "Défaut")
    m3.metric("τ_m", f"{tau_m} ms", f"± {tau_sd}" if tau_sd else "Default" if lang_choice == "English" else "Défaut")
    m4.metric("C_m", f"{C_m} pF", f"± {cm_sd}" if cm_sd else "Estimated" if lang_choice == "English" else "Estimé")
    m5.metric("V_th", f"{V_th} mV", f"± {v_th_sd}" if v_th_sd else "Default" if lang_choice == "English" else "Défaut")

    if g_max is not None or v_half is not None or curr_dens is not None:
        st.subheader(t["t5_sec2"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("g_max", f"{g_max} nS" if g_max is not None else "N/A", f"± {g_max_sd}" if g_max_sd else "")
        c2.metric("Current density", f"{curr_dens} pA/pF" if curr_dens is not None else "N/A")
        c3.metric("V_1/2", f"{v_half} mV" if v_half is not None else "N/A", f"± {v_half_sd}" if v_half_sd else "")
        c4.metric("Slope (k)", f"{slope_k} mV" if slope_k is not None else "N/A")

    # --- SOURCES BIBLIOGRAPHIQUES & DOI ---
    st.subheader(t["t5_sec3"])
    sources_list = sub_df["source"].dropna().unique().tolist() if "source" in sub_df.columns else []
    dois_list = sub_df["doi"].dropna().unique().tolist() if "doi" in sub_df.columns else []
    cell_ids_list = sub_df["cell_id"].dropna().unique().tolist() if "cell_id" in sub_df.columns else []

    st.markdown(t["t5_orig"].format(', '.join(map(str, sources_list))))
    
    if dois_list:
        st.markdown(t["t5_dois"])
        for d in dois_list:
            doi_url = d if d.startswith("http") else f"https://doi.org/{d}"
            st.markdown(f"- [{d}]({doi_url})")
    else:
        st.info(t["t5_nodoi"])
        
    with st.expander(t["t5_expand"]):
        st.write(cell_ids_list)
    st.caption(t["t5_capt"])

    # --- SCRIPT DE SIMULATION ---
    st.subheader(t["t5_sec4"])
    formalism = st.radio(
        t["t5_form"],
        ["Leaky Integrate-and-Fire (Brian2)", "Hodgkin-Huxley / Courant ionique (Python / NEURON)"],
        horizontal=True,
        key="mod_formalism"
    )

    if formalism == "Leaky Integrate-and-Fire (Brian2)":
        refrac = round(ap_hw * 2.0, 2) if ap_hw else 2.0
        code_out = f'''from brian2 import *

# Paramètres extraits de Electrophysiology-database (n = {len(sub_df)})
# Région : {sel_region} | Préparation : {sel_prep} | Identité : {sel_id}

E_L = {E_L} * mV            # Potentiel de repos / fuite
V_th = {V_th} * mV          # Seuil de déclenchement du PA
V_reset = {V_reset} * mV    # Potentiel post-spike (V_th - AHP)
R_in = {R_in} * Mohm        # Résistance d'entrée
tau_m = {tau_m} * ms        # Constante de temps membranaire
I_inj = {I_rheo} * pA       # Courant d'injection (Rhéobase médiane)

eqs = """
dV/dt = (-(V - E_L) + R_in * I_inj) / tau_m : volt (unless refractory)
"""

neuron = NeuronGroup(1, eqs, threshold='V > V_th', reset='V = V_reset', refractory={refrac}*ms, method='exact')
neuron.V = E_L

mon = StateMonitor(neuron, 'V', record=True)
run(500 * ms)
'''
    else:
        area_str = f"{soma_area} um^2" if soma_area else "Non renseignée"
        g_max_val = f"{g_max} * nS" if g_max is not None else "None"
        v_half_val = f"{v_half} * mV" if v_half is not None else "None"
        slope_val = f"{slope_k} * mV" if slope_k is not None else "None"
        gamma_val = f"{gamma_sc} * pS" if gamma_sc is not None else "None"
        ec50_val = f"{ec50_ca} * uM" if ec50_ca is not None else "None"

        code_out = f'''# Paramètres biophysiques & cinétiques pour Hodgkin-Huxley / Courant ionique
# Source : Electrophysiology-database (n = {len(sub_df)})

hh_channel_config = {{
    "C_m_pF": {C_m},
    "g_L_nS": {g_L},
    "E_L_mV": {E_L},
    "g_max_nS": {g_max_val},
    "single_channel_conductance": {gamma_val},
    "V_half_activation": {v_half_val},
    "slope_factor_k": {slope_val},
    "EC50_calcium": {ec50_val}
}}
'''

    st.code(code_out, language="python")

    # --- BOUTON « SIMULER UN APERÇU EN DIRECT » (EULER + PLOTLY) ---
    if st.button(t["t5_btn_sim"]):
        T_total = 200.0
        dt = 0.1
        steps = int(T_total / dt)
        time_arr = np.linspace(0, T_total, steps)
        v_trace = np.zeros(steps)
        
        v_current = E_L
        v_trace[0] = v_current
        i_test = I_rheo * 1.2
        
        for i in range(1, steps):
            dv = (-(v_current - E_L) + R_in * i_test) / tau_m * dt
            v_current += dv
            if v_current >= V_th:
                v_current = 20.0
                v_trace[i] = v_current
                if i + 1 < steps:
                    v_trace[i+1] = V_reset
                v_current = V_reset
            else:
                v_trace[i] = v_current

        sim_df = pd.DataFrame({
            "Temps (ms)" if lang_choice == "Français" else "Time (ms)": time_arr,
            "Potentiel de membrane (mV)" if lang_choice == "Français else" else "Membrane Potential (mV)": v_trace
        })

        fig_sim = px.line(
            sim_df, x=sim_df.columns[0], y=sim_df.columns[1],
            title=f"LIF Simulation Preview ({sel_region} | n={len(sub_df)})",
            height=400
        )
        fig_sim.update_layout(plot_bgcolor='#F4F4F4', paper_bgcolor='#F4F4F4', font=dict(color='#111111'))
        fig_sim.update_traces(line=dict(color='#002FA7', width=2))
        st.plotly_chart(fig_sim, use_container_width=True)
        st.success(t["t5_sim_ok"])

# --- PRÉPARATION DES LISTES POUR L'EXPORT JSON ---
    sources_list = sub_df["source"].dropna().unique().tolist() if "source" in sub_df.columns else []
    dois_list = sub_df["doi"].dropna().unique().tolist() if "doi" in sub_df.columns else []


    # --- EXPORT AU FORMAT JSON / NEUROML-LIKE ---
    export_dict = {
        "metadata": {
            "n_neurons": len(sub_df),
            "brain_region": sel_region,
            "preparation": sel_prep,
            "known_identity": sel_id,
            "sources": sources_list if "source" in sub_df.columns else [],
            "dois": dois_list if "doi" in sub_df.columns else []
        },
        "passive_parameters": {
            "resting_membrane_potential_mv": E_L,
            "input_resistance_Mohm": R_in,
            "membrane_time_constant_ms": tau_m,
            "capacitance_pf": C_m,
            "leak_conductance_ns": g_L
        },
        "action_potential_parameters": {
            "ap_threshold_mv": V_th,
            "ap_reset_mv": V_reset,
            "rheobase_pa": I_rheo,
            "ahp_amplitude_mv": ahp_amp
        },
        "channel_kinetics": {
            "g_max_ns": g_max,
            "v_half_mv": v_half,
            "slope_factor_k_mv": slope_k,
            "single_channel_conductance_ps": gamma_sc
        }
    }
    json_str = json.dumps(export_dict, indent=4)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(label=t["t5_dl_py"], data=code_out, file_name="in_silico_model_params.py", mime="text/x-python", key="mod_py")
    with col_dl2:
        st.download_button(label=t["t5_dl_json"], data=json_str, file_name="in_silico_model_profile.json", mime="application/json", key="mod_json")


# --- ONGLETS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(t["tabs"])

# ==========================================
# ONGLET 1 : EXPLORATION DES DONNÉES
# ==========================================
with tab1:
    st.subheader(t["t1_sub"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric(t["t1_m1"], len(df_filtered))
    col2.metric(t["t1_m2"], df_filtered['preparation'].nunique())
    col3.metric(t["t1_m3"], len(df_filtered[df_filtered['known_identity'] != 'Unknown']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=t["t1_dl"],
        data=csv,
        file_name='electrophys_filtered_export.csv',
        mime='text/csv',
    )

# ==========================================
# ONGLET 2 : ACP INTERACTIVE
# ==========================================
with tab2:
    st.subheader(t["t2_sub"])
    
    if len(df_filtered) < 5:
        st.warning(t["t2_warn"])
    else:
        col_opt1, col_opt2 = st.columns(2)
        color_by = col_opt1.selectbox(t["t2_c1"], ['preparation', 'known_identity', 'brain_region', 'source'])
        symbol_by = col_opt2.selectbox(t["t2_c2"], ['age_or_div', 'preparation', 'known_identity'])
        
        X = df_filtered[FEATURE_COLS]
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        df_filtered['PC1'] = coords[:, 0]
        df_filtered['PC2'] = coords[:, 1]
        
        var_exp = pca.explained_variance_ratio_ * 100
        
        fig = px.scatter(
            df_filtered, x='PC1', y='PC2',
            color=color_by, symbol=symbol_by,
            hover_data=['cell_id', 'brain_region', 'age_or_div', 'known_identity'],
            labels={
                'PC1': f"PC1 ({var_exp[0]:.1f}% variance)",
                'PC2': f"PC2 ({var_exp[1]:.1f}% variance)"
            },
            height=600
        )
        fig.update_layout(
            plot_bgcolor='#F4F4F4',
            paper_bgcolor='#F4F4F4',
            font=dict(color='#111111', family="sans-serif"),
            title=dict(text=t["t2_title"], font=dict(color='#002FA7', size=16, family="sans-serif"))
        )
        fig.update_traces(marker=dict(size=11, opacity=0.9, line=dict(width=1.5, color='#111111')))
        
        config = {
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'acp_electrophys_drn',
                'height': 700,
                'width': 1000,
                'scale': 3
            }
        }
        st.plotly_chart(fig, use_container_width=True, config=config)

# ==========================================
# ONGLET 3 : PRÉDICTEUR D'IDENTITÉ (RANDOM FOREST)
# ==========================================
with tab3:
    st.subheader(t["t3_sub"])
    st.markdown(t["t3_desc"])

    model, class_counts, X_train_df, y_train_ser = get_trained_model(df)
    active_cols = [c for c in FEATURE_COLS if c in df.columns]

    if model is None:
        st.error(t["t3_err"])
    else:
        # 1. Choix de l'enregistrement à évaluer (unique source pour la prédiction et les z-scores)
        eval_mode = st.radio(
            "Enregistrement à évaluer" if lang_choice == "Français" else "Record to evaluate",
            options=(
                ["Cellule existante dans la base", "Saisie manuelle"]
                if lang_choice == "Français"
                else ["Existing cell in database", "Manual entry"]
            ),
            horizontal=True,
            key="rf_eval_mode",
        )

        defaults = {
            "rmp_mv": -62.0,
            "tau_ms": 18.0,
            "cm_pf": 40.0,
            "rheobase_pa": 45.0,
            "ap_threshold_mv": -39.0,
            "ap_amplitude_mv": 75.0,
            "ap_half_width_ms": 1.4,
            "ahp_amplitude_mv": 11.0,
            "ahp_duration_ms": 45.0,
        }

        if eval_mode in ["Cellule existante dans la base", "Existing cell in database"] and not df.empty:
            cell_options = df["cell_id"].dropna().astype(str).unique().tolist()
            chosen_id = st.selectbox("cell_id", options=cell_options, key="rf_selected_cell_id")
            target_row = df.loc[df["cell_id"].astype(str) == chosen_id].iloc[0]
        else:
            c1, c2, c3 = st.columns(3)
            cols_grid = [c1, c2, c3]
            manual_vals: dict[str, float] = {}
            for idx, col in enumerate(active_cols):
                with cols_grid[idx % 3]:
                    default_val = defaults.get(col, 0.0)
                    manual_vals[col] = st.number_input(
                        col,
                        value=float(default_val),
                        key=f"rf_num_{col}",
                    )
            target_row = pd.Series(manual_vals)

        # 2. Prédiction unique
        x_one = pd.DataFrame(
            [{col: pd.to_numeric(target_row.get(col), errors="coerce") for col in X_train_df.columns}]
        )
        proba = model.predict_proba(x_one)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(proba))
        pred_label = classes[best_idx]
        conf_pct = float(proba[best_idx]) * 100.0

        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.success(f"### {t['t3_pred']} : **{pred_label}**")
        m_col2.info(f"### {t['t3_conf']} : **{conf_pct:.1f} %**")
        if class_counts is not None and class_counts.min() >= 2:
            try:
                cv_res = evaluate_cv(df)
                m_col3.metric(t["t3_f1_cv"], f"{cv_res.f1_macro:.2f}")
            except ValueError:
                m_col3.metric(t["t3_n_single"], len(y_train_ser))
        else:
            m_col3.metric(t["t3_n_single"], len(y_train_ser))

        # 3. Graphique probabilités + Tableau z-scores côte à côte
        g_col, z_col = st.columns([1, 1.2])

        with g_col:
            proba_df = pd.DataFrame({
                t["t3_class_col"]: classes,
                t["t3_proba_col"]: np.round(proba * 100.0, 1),
            }).sort_values(by=t["t3_proba_col"], ascending=False)

            fig_proba = px.bar(
                proba_df,
                x=t["t3_class_col"],
                y=t["t3_proba_col"],
                title=t["t3_proba_chart"],
            )
            fig_proba.update_layout(yaxis_range=[0, 100], height=360)
            st.plotly_chart(fig_proba, use_container_width=True)

        with z_col:
            st.markdown(f"#### {t['t3_z_title']}")
            z_table = compute_literature_zscores(target_row, df, feature_cols=active_cols)

            if z_table.empty:
                st.info(t["t3_z_empty"])
            else:
                z_col_name = next(
                    (c for c in z_table.columns if "z" in c.lower() and "score" in c.lower()),
                    None,
                )

                if z_col_name is not None:
                    plot_df = z_table.copy()
                    plot_df["_z_num"] = pd.to_numeric(plot_df[z_col_name], errors="coerce")
                    plot_df = plot_df.dropna(subset=["_z_num"])

                    if not plot_df.empty:
                        param_candidates = [t["t3_col_param"], "Paramètre", "Parameter", "feature"]
                        pp_candidates = [t["t3_col_pp"], "Profil / Paramètre", "Profile / Parameter"]
                        prof_candidates = [t["t3_col_profile"], "Profil", "Profile"]

                        label_col = next((c for c in param_candidates if c in plot_df.columns), None)
                        if label_col is None:
                            label_col = next((c for c in pp_candidates if c in plot_df.columns), plot_df.columns[0])

                        prof_col = next((c for c in prof_candidates if c in plot_df.columns), None)
                        if prof_col is not None and label_col != prof_col:
                            plot_df["_label"] = (
                                plot_df[prof_col].astype(str) + " — " + plot_df[label_col].astype(str)
                            )
                        else:
                            plot_df["_label"] = plot_df[label_col].astype(str)

                        out_lbl = t["t3_z_out"]
                        in_lbl = t["t3_z_in"]
                        plot_df["_statut"] = np.where(
                            plot_df["_z_num"].abs() > 2.0,
                            out_lbl,
                            in_lbl,
                        )

                        fig_z = px.bar(
                            plot_df.sort_values("_z_num", ascending=True),
                            x="_z_num",
                            y="_label",
                            orientation="h",
                            color="_statut",
                            color_discrete_map={
                                out_lbl: "#002FA7",
                                in_lbl: "#9E9E9E",
                            },
                            labels={
                                "_z_num": t["t3_z_axis_x"],
                                "_label": "",
                                "_statut": t["t3_z_status"],
                            },
                            title=t["t3_z_chart_title"],
                            height=300,
                        )
                        fig_z.add_vline(x=0, line_width=1.2, line_color="#111111")
                        fig_z.add_vline(x=2, line_width=1.5, line_dash="dash", line_color="#002FA7")
                        fig_z.add_vline(x=-2, line_width=1.5, line_dash="dash", line_color="#002FA7")
                        fig_z.update_layout(
                            plot_bgcolor="#F4F4F4",
                            paper_bgcolor="#F4F4F4",
                            font=dict(color="#111111"),
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_z, use_container_width=True)

                    def _highlight_z(val):
                        try:
                            num = float(val)
                        except (TypeError, ValueError):
                            return ""
                        if np.isnan(num):
                            return ""
                        if abs(num) > 2.0:
                            return "background-color: #DCE5FF; color: #002FA7; font-weight: 700;"
                        return ""

                    styled_z = z_table.style.map(_highlight_z, subset=[z_col_name])
                    st.dataframe(styled_z, hide_index=True, use_container_width=True, height=240)
                else:
                    st.dataframe(z_table, hide_index=True, use_container_width=True, height=310)

                z_csv = z_table.to_csv(index=False).encode("utf-8")
                z_filename = (
                    f"zscores_{chosen_id}.csv"
                    if eval_mode in ["Cellule existante dans la base", "Existing cell in database"] and not df.empty
                    else (
                        "zscores_saisie_manuelle.csv"
                        if lang_choice == "Français"
                        else "zscores_manual_entry.csv"
                    )
                )
                st.download_button(
                    label=t["t3_z_dl"],
                    data=z_csv,
                    file_name=z_filename,
                    mime="text/csv",
                    key="rf_zscores_dl_csv",
                )

        # 4. Importance des variables & métadonnées d'entraînement
        rf_estimator = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
        if hasattr(rf_estimator, "feature_importances_"):
            imp_df = pd.DataFrame({
                t["t3_feat_col"]: list(X_train_df.columns),
                t["t3_imp_col"]: rf_estimator.feature_importances_,
            }).sort_values(by=t["t3_imp_col"], ascending=True)

            fig_imp = px.bar(
                imp_df,
                x=t["t3_imp_col"],
                y=t["t3_feat_col"],
                orientation="h",
                title=t["t3_feat_imp"],
                height=340,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        single_cell_df = df[df.get("record_type", "single_cell") == "single_cell"] if "record_type" in df.columns else df
        with st.expander(t["t3_exp"]):
            class_summary = ", ".join([f"{k} (n={v})" for k, v in y_train_ser.value_counts().items()])
            sources_str = ", ".join(map(str, single_cell_df["source"].dropna().unique().tolist())) if "source" in single_cell_df.columns else "—"
            st.markdown(
                f"Modèle entraîné sur **{len(y_train_ser)} enregistrements `single_cell`** "
                f"[{class_summary}] (sources : {sources_str})."
            )
            if "doi" in single_cell_df.columns:
                train_dois = single_cell_df["doi"].dropna().unique().tolist()
                if train_dois:
                    st.markdown("**DOI :**")
                    for d in train_dois:
                        doi_url = d if str(d).startswith("http") else f"https://doi.org/{d}"
                        st.markdown(f"- [{d}]({doi_url})")

# ==========================================
# ONGLET 4 : AJOUTER UN NEURONE
# ==========================================
def _get_pg_columns(pg_conn, table_name: str) -> set[str]:
    rows = pg_conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    ).fetchall()
    return {r[0] for r in rows}


def update_existing_neuron(
    db_path: str,
    cell_id: str,
    updates: dict[str, object],
    table_name: str = "neurons_electrophys",
) -> int:
    clean_updates = dict(updates)
    if "known_identity" in clean_updates and clean_updates["known_identity"] is not None:
        harmonized = harmonize_identity_value(clean_updates["known_identity"])
        clean_updates["known_identity"] = harmonized if harmonized else "Unknown"

    db_url = _get_supabase_url()
    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as pg_conn:
                existing_cols = _get_pg_columns(pg_conn, table_name)
                valid_updates = {
                    col: val
                    for col, val in clean_updates.items()
                    if val is not None and col in existing_cols and col != "cell_id"
                }
                if not valid_updates:
                    return 0

                set_clause = ", ".join([f"{col} = %s" for col in valid_updates])
                params = list(valid_updates.values()) + [cell_id]
                cur = pg_conn.execute(
                    f"UPDATE {table_name} SET {set_clause} WHERE cell_id = %s",
                    tuple(params),
                )
                pg_conn.commit()
                return cur.rowcount
        except Exception as exc:
            st.warning(f"Mise à jour Supabase indisponible ({exc}), repli sur SQLite local.")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        existing_cols = {
            row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        valid_updates = {
            col: val
            for col, val in clean_updates.items()
            if val is not None and col in existing_cols and col != "cell_id"
        }
        if not valid_updates:
            return 0

        set_clause = ", ".join([f"{col} = ?" for col in valid_updates])
        params = list(valid_updates.values()) + [cell_id]
        cursor.execute(
            f"UPDATE {table_name} SET {set_clause} WHERE cell_id = ?",
            tuple(params),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def insert_new_neuron(
    db_path: str,
    values: tuple,
    table_name: str = "neurons_electrophys",
) -> None:
    placeholders_pg = ", ".join(["%s"] * len(values))
    placeholders_sqlite = ", ".join(["?"] * len(values))

    db_url = _get_supabase_url()
    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as pg_conn:
                pg_conn.execute(
                    f"""
                    INSERT INTO {table_name} VALUES ({placeholders_pg})
                    ON CONFLICT (cell_id) DO NOTHING
                    """,
                    values,
                )
                pg_conn.commit()
                return
        except Exception as exc:
            st.warning(f"Ajout Supabase indisponible ({exc}), repli sur SQLite local.")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders_sqlite})",
            values,
        )
        conn.commit()
    finally:
        conn.close()

def _has_write_access(lang: str) -> bool:
    try:
        expected_code = str(st.secrets.get("INSERT_ACCESS_CODE", "")).strip()
    except Exception:
        expected_code = ""

    if not expected_code:
        return True

    if st.session_state.get("t4_write_unlocked", False):
        return True

    st.info(
        "L'ajout et la mise à jour de neurones sont protégés par un code d'accès."
        if lang == "Français"
        else "Adding and updating neurons is protected by an access code."
    )

    with st.form("t4_access_form"):
        entered_code = st.text_input(
            "Code d'accès contributeur" if lang == "Français" else "Contributor access code",
            type="password",
        )
        unlock_btn = st.form_submit_button(
            "Déverrouiller" if lang == "Français" else "Unlock"
        )

    if unlock_btn:
        if entered_code.strip() == expected_code:
            st.session_state["t4_write_unlocked"] = True
            st.rerun()
        else:
            st.error(
                "Code d'accès incorrect."
                if lang == "Français"
                else "Incorrect access code."
            )

    return False
  
with tab4:
    st.subheader(t["t4_sub"])
    st.markdown(t["t4_desc"])

    if not _has_write_access(lang_choice):
        st.stop()

    try:
        if str(st.secrets.get("INSERT_ACCESS_CODE", "")).strip():
            if st.button(
                "Verrouiller l'onglet" if lang_choice == "Français" else "Lock tab",
                key="t4_lock_btn",
            ):
                st.session_state["t4_write_unlocked"] = False
                st.rerun()
    except Exception:
        pass
    
    canonical_ids = get_canonical_identities()
    identity_options = ["Unknown"] + canonical_ids

    mode = st.radio(
        t["t4_mode_label"],
        options=[t["t4_mode_add"], t["t4_mode_update"]],
        horizontal=True,
        key="t4_mode_radio",
    )

    if mode == t["t4_mode_add"]:
        record_type = st.radio(
            "Type d'enregistrement",
            options=["single_cell", "group_summary"],
            format_func=lambda x: (
                "Cellule individuelle (single_cell)"
                if x == "single_cell"
                else "Moyenne de littérature (group_summary)"
            ),
            horizontal=True,
            key="t4_record_type_radio",
        )

        with st.form("new_neuron_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                new_id = st.text_input(t["t4_id"])
                new_source = st.selectbox(
                    t["t4_source"],
                    ["Lab_ULiege", "Literature", "Allen_Institute", "Other"],
                    index=1 if record_type == "group_summary" else 0,
                )
                new_region = st.text_input(t["t4_region"], value="DRN")
            with f_col2:
                new_prep = st.selectbox(
                    t["t4_prep"],
                    ["Acute_Slice", "Organotypic_Slice", "Cell_Culture"],
                )
                new_age = st.text_input(t["t4_age"], value="DIV7")
                new_ident = st.selectbox(t["t4_ident"], identity_options)
            with f_col3:
                new_rmp = st.number_input("RMP (mV)", value=-60.0)
                new_tau = st.number_input("Tau (ms)", value=15.0)
                new_cm = st.number_input("Cm (pF)", value=35.0)

            f_col4, f_col5 = st.columns(2)
            with f_col4:
                new_rheo = st.number_input(t["t4_rheo"], value=40.0)
                new_thr = st.number_input(t["t4_thr"], value=-40.0)
                new_amp = st.number_input(t["t4_amp"], value=75.0)
            with f_col5:
                new_hw = st.number_input(t["t4_hw"], value=1.2)
                new_ahp_amp = st.number_input(t["t4_ahp_amp"], value=10.0)
                new_ahp_dur = st.number_input(t["t4_ahp_dur"], value=40.0)

            # Champs supplémentaires uniquement pour les moyennes de littérature
            if record_type == "group_summary":
                st.markdown("**Métadonnées & dispersion (littérature)**")
                l_col1, l_col2, l_col3 = st.columns(3)
                with l_col1:
                    new_n_cells = st.number_input("Effectif (n)", min_value=1, value=10, step=1)
                with l_col2:
                    new_disp_type = st.selectbox("Type de dispersion", ["SD", "SEM"])
                with l_col3:
                    new_doi = st.text_input("DOI / Référence", placeholder="10.1016/...")

                d_col1, d_col2, d_col3 = st.columns(3)
                with d_col1:
                    disp_rmp = st.text_input("± RMP (mV)", value="")
                    disp_tau = st.text_input("± Tau (ms)", value="")
                    disp_cm = st.text_input("± Cm (pF)", value="")
                with d_col2:
                    disp_rheo = st.text_input("± Rhéobase (pA)", value="")
                    disp_thr = st.text_input("± Seuil PA (mV)", value="")
                    disp_amp = st.text_input("± Amplitude PA (mV)", value="")
                with d_col3:
                    disp_hw = st.text_input("± Demi-largeur PA (ms)", value="")
                    disp_ahp_amp = st.text_input("± Amplitude AHP (mV)", value="")
                    disp_ahp_dur = st.text_input("± Durée AHP (ms)", value="")

            submitted = st.form_submit_button(t["t4_btn"])

            if submitted:
                clean_id = new_id.strip()
                if not clean_id:
                    st.error(t["t4_err_id"])
                else:
                    if new_ident == "Unknown":
                        final_ident = "Unknown"
                    else:
                        harmonized = harmonize_identity_value(new_ident)
                        final_ident = harmonized if harmonized else "Unknown"

                    def _parse_disp(txt: str) -> float | None:
                        s = txt.strip()
                        if not s:
                            return None
                        return to_sd(
                            value=float(s.replace(",", ".")),
                            dispersion_type=new_disp_type,
                            n_cells=int(new_n_cells),
                        )

                    record = {
                        "cell_id": clean_id,
                        "source": new_source,
                        "brain_region": new_region.strip() or "DRN",
                        "preparation": new_prep,
                        "age_or_div": new_age.strip() or "Unknown",
                        "known_identity": final_ident,
                        "rmp_mv": new_rmp,
                        "tau_ms": new_tau,
                        "cm_pf": new_cm,
                        "rheobase_pa": new_rheo,
                        "ap_threshold_mv": new_thr,
                        "ap_amplitude_mv": new_amp,
                        "ap_half_width_ms": new_hw,
                        "ahp_amplitude_mv": new_ahp_amp,
                        "ahp_duration_ms": new_ahp_dur,
                        "record_type": record_type,
                        "n_cells": int(new_n_cells) if record_type == "group_summary" else 1,
                        "dispersion_type": new_disp_type if record_type == "group_summary" else None,
                        "reference_doi": (new_doi.strip() or None) if record_type == "group_summary" else None,
                        "rmp_mv_sd": _parse_disp(disp_rmp) if record_type == "group_summary" else None,
                        "tau_ms_sd": _parse_disp(disp_tau) if record_type == "group_summary" else None,
                        "cm_pf_sd": _parse_disp(disp_cm) if record_type == "group_summary" else None,
                        "rheobase_pa_sd": _parse_disp(disp_rheo) if record_type == "group_summary" else None,
                        "ap_threshold_mv_sd": _parse_disp(disp_thr) if record_type == "group_summary" else None,
                        "ap_amplitude_mv_sd": _parse_disp(disp_amp) if record_type == "group_summary" else None,
                        "ap_half_width_ms_sd": _parse_disp(disp_hw) if record_type == "group_summary" else None,
                        "ahp_amplitude_mv_sd": _parse_disp(disp_ahp_amp) if record_type == "group_summary" else None,
                        "ahp_duration_ms_sd": _parse_disp(disp_ahp_dur) if record_type == "group_summary" else None,
                    }

                    insert_neuron_record(db_path=DB_PATH, record=record)
                    st.cache_data.clear()
                    st.success(t["t4_success"].format(clean_id))
    
    else:
        if df.empty or "cell_id" not in df.columns:
            st.info(t["t4_no_cells"])
        else:
            cell_ids = df["cell_id"].dropna().astype(str).unique().tolist()
            selected_cell = st.selectbox(
                t["t4_select_cell"],
                options=cell_ids,
                key="t4_update_cell_select",
            )
            current_row = df[df["cell_id"].astype(str) == selected_cell].iloc[0]

            current_harm = harmonize_identity_value(current_row.get("known_identity"))
            default_ident = current_harm if current_harm in canonical_ids else "Unknown"
            ident_idx = identity_options.index(default_ident)

            with st.form(f"update_neuron_form_{selected_cell}"):
                u1, u2 = st.columns(2)
                with u1:
                    upd_ident = st.selectbox(
                        t["t4_ident"],
                        options=identity_options,
                        index=ident_idx,
                    )
                    upd_age = st.text_input(
                        t["t4_age"],
                        value="" if pd.isna(current_row.get("age_or_div")) else str(current_row.get("age_or_div")),
                    )
                with u2:
                    upd_rheo_txt = st.text_input(
                        t["t4_rheo"],
                        value="" if pd.isna(current_row.get("rheobase_pa")) else str(current_row.get("rheobase_pa")),
                    )

                st.markdown(f"**{t['t4_morpho_section']}**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    upd_branches_txt = st.text_input(
                        t["t4_num_branches"],
                        value="" if pd.isna(current_row.get("num_branches")) else str(current_row.get("num_branches")),
                    )
                    upd_junctions_txt = st.text_input(
                        t["t4_num_junctions"],
                        value="" if pd.isna(current_row.get("num_junctions")) else str(current_row.get("num_junctions")),
                    )
                with m2:
                    upd_maxlen_txt = st.text_input(
                        t["t4_max_branch_len"],
                        value="" if pd.isna(current_row.get("max_branch_length_um")) else str(current_row.get("max_branch_length_um")),
                    )
                    upd_critrad_txt = st.text_input(
                        t["t4_crit_radius"],
                        value="" if pd.isna(current_row.get("critical_radius_um")) else str(current_row.get("critical_radius_um")),
                    )
                with m3:
                    upd_sholl_txt = st.text_input(
                        t["t4_sholl"],
                        value="" if pd.isna(current_row.get("sholl_intersections")) else str(current_row.get("sholl_intersections")),
                    )

                upd_submitted = st.form_submit_button(t["t4_update_btn"])

                if upd_submitted:
                    def _to_float_or_none(val: str) -> float | None:
                        s = val.strip()
                        if not s:
                            return None
                        return float(s.replace(",", "."))

                    payload: dict[str, object] = {
                        "known_identity": upd_ident,
                        "age_or_div": upd_age.strip() if upd_age.strip() else None,
                        "rheobase_pa": _to_float_or_none(upd_rheo_txt),
                        "num_branches": _to_float_or_none(upd_branches_txt),
                        "num_junctions": _to_float_or_none(upd_junctions_txt),
                        "max_branch_length_um": _to_float_or_none(upd_maxlen_txt),
                        "critical_radius_um": _to_float_or_none(upd_critrad_txt),
                        "sholl_intersections": _to_float_or_none(upd_sholl_txt),
                    }

                    rows_updated = update_existing_neuron(
                        db_path=DB_PATH,
                        cell_id=selected_cell,
                        updates=payload,
                    )
                    st.cache_data.clear()
                    if rows_updated > 0:
                        st.success(t["t4_update_success"].format(selected_cell))
                    else:
                        st.info(t["t4_update_noop"].format(selected_cell))

# ==========================================
# ONGLET 5 : MODÉLISATION IN SILICO
# ==========================================
with tab5:
    render_modeling_tab(df_filtered, t)

                # streamlit run app.py
                # pour l'arrêter, ctrl + C