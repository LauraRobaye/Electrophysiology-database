from __future__ import annotations

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import psycopg
import json
from src.update_neuron import insert_neuron_record, to_sd, update_neuron
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
            "04 // MODÉLISATION IN SILICO",
            "05 // INSERTION"
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
        "t4_rec_type_label": "Type d'enregistrement",
        "t4_rec_single": "Cellule individuelle (single_cell)",
        "t4_rec_group": "Moyenne de littérature (group_summary)",
        "t4_insert_caption": (
            "Les champs **cell_id** et **source** sont obligatoires. "
            "Cochez **Non rapporté** dès qu'une donnée est absente de l'article."
        ),
        "t4_meta_sec": "Métadonnées et référence bibliographique",
        "t4_phys_sec": "Paramètres électrophysiologiques",
        "t4_lit_sec": "Métadonnées de littérature & dispersion",
        "t4_disp_sec": "Dispersion associée aux paramètres (± SD ou SEM)",
        "t4_n_cells_lbl": "Effectif (n_cells)",
        "t4_disp_na": "Dispersion non rapportée",
        "t4_disp_type": "Type de dispersion",
        "t4_ref_doi": "DOI / Référence (reference_doi)",
        "t4_not_reported": "Non rapporté",
        "t4_required_ph": "Obligatoire",
        "t4_err_source": "Le champ **source** est obligatoire.",
        "t4_mode_delete": "Supprimer un neurone existant",
        "t4_delete_warn": "La suppression est définitive. Vérifiez l'enregistrement avant confirmation.",
        "t4_delete_confirm": "Je confirme vouloir supprimer définitivement ce neurone",
        "t4_delete_btn": "Supprimer définitivement",
        "t4_delete_err_confirm": "Cochez la case de confirmation avant de supprimer.",
        "t4_delete_success": "Le neurone '{0}' a été supprimé.",

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
            "04 // IN SILICO MODELING",
            "05 // INSERTION"
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
        "t4_rec_type_label": "Record type",
        "t4_rec_single": "Single cell (single_cell)",
        "t4_rec_group": "Literature summary (group_summary)",
        "t4_insert_caption": (
            "**cell_id** and **source** fields are required. "
            "Check **Not reported** whenever a value is missing from the paper."
        ),
        "t4_meta_sec": "Metadata and bibliographic reference",
        "t4_phys_sec": "Electrophysiological parameters",
        "t4_lit_sec": "Literature metadata & dispersion",
        "t4_disp_sec": "Parameter dispersion (± SD or SEM)",
        "t4_n_cells_lbl": "Sample size (n_cells)",
        "t4_disp_na": "Dispersion not reported",
        "t4_disp_type": "Dispersion type",
        "t4_ref_doi": "DOI / Reference (reference_doi)",
        "t4_not_reported": "Not reported",
        "t4_required_ph": "Required",
        "t4_err_source": "The **source** field is required.",
        "t4_mode_delete": "Delete an existing neuron",
        "t4_delete_warn": "Deletion is permanent. Review the record before confirming.",
        "t4_delete_confirm": "I confirm that I want to permanently delete this neuron",
        "t4_delete_btn": "Delete permanently",
        "t4_delete_err_confirm": "Check the confirmation box before deleting.",
        "t4_delete_success": "Neuron '{0}' has been deleted.",
        
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
FIELD_LABELS_BY_LANG = {
    "Français": {
        "cell_id": "Identifiant cellule (cell_id) *",
        "source": "Source bibliographique (source) *",
        "brain_region": "Région cérébrale (brain_region)",
        "cell_type": "Type / identité cellulaire",
        "known_identity": "Identité connue (known_identity)",
        "preparation": "Préparation (preparation)",
        "age_or_div": "Âge ou DIV (age_or_div)",
        "species": "Espèce",
        "n_cells": "Nombre de cellules (n_cells)",
        "rmp_mv": "RMP (mV)",
        "tau_ms": "Constante de temps tau (ms)",
        "cm_pf": "Capacitance membranaire Cm (pF)",
        "rheobase_pa": "Rhéobase (pA)",
        "ap_threshold_mv": "Seuil PA (mV)",
        "ap_amplitude_mv": "Amplitude PA (mV)",
        "ap_half_width_ms": "Demi-largeur PA (ms)",
        "ahp_amplitude_mv": "Amplitude AHP (mV)",
        "ahp_duration_ms": "Durée AHP (ms)",
    },
    "English": {
        "cell_id": "Cell identifier (cell_id) *",
        "source": "Bibliographic source (source) *",
        "brain_region": "Brain region (brain_region)",
        "cell_type": "Cell type / identity",
        "known_identity": "Known identity (known_identity)",
        "preparation": "Preparation (preparation)",
        "age_or_div": "Age or DIV (age_or_div)",
        "species": "Species",
        "n_cells": "Number of cells (n_cells)",
        "rmp_mv": "RMP (mV)",
        "tau_ms": "Membrane time constant tau (ms)",
        "cm_pf": "Membrane capacitance Cm (pF)",
        "rheobase_pa": "Rheobase (pA)",
        "ap_threshold_mv": "AP threshold (mV)",
        "ap_amplitude_mv": "AP amplitude (mV)",
        "ap_half_width_ms": "AP half-width (ms)",
        "ahp_amplitude_mv": "AHP amplitude (mV)",
        "ahp_duration_ms": "AHP duration (ms)",
    },
}


def render_field_with_na(col_name: str, series_dtype, key_prefix: str = "ins", lang: str = "Français"):
    """Rend un champ adapté au type de colonne avec option 'Non rapporté' / 'Not reported'."""
    t_local = LANGUAGES.get(lang, LANGUAGES["Français"])
    labels_map = FIELD_LABELS_BY_LANG.get(lang, FIELD_LABELS_BY_LANG["Français"])
    label = labels_map.get(col_name, col_name)
    is_required_no_na = col_name in NON_NA_REQUIRED_COLS

    if is_required_no_na:
        val = st.text_input(
            label,
            placeholder=t_local["t4_required_ph"],
            key=f"{key_prefix}_{col_name}_txt",
        ).strip()
        return val if val else None

    col_input, col_na = st.columns([4, 1.3], vertical_alignment="bottom")
    not_reported = col_na.checkbox(
        t_local["t4_not_reported"],
        value=True if col_name not in {"brain_region"} else False,
        key=f"{key_prefix}_{col_name}_na",
    )

    if not_reported:
        if pd.api.types.is_numeric_dtype(series_dtype) or col_name in FEATURE_COLS:
            col_input.number_input(
                label,
                value=0.0,
                disabled=True,
                key=f"{key_prefix}_{col_name}_num_na",
            )
        else:
            col_input.text_input(
                label,
                value="",
                disabled=True,
                key=f"{key_prefix}_{col_name}_txt_na",
            )
        return None

    if pd.api.types.is_integer_dtype(series_dtype) and col_name not in FEATURE_COLS:
        val = col_input.number_input(
            label,
            value=1,
            step=1,
            key=f"{key_prefix}_{col_name}_int",
        )
        return int(val)

    if pd.api.types.is_float_dtype(series_dtype) or col_name in FEATURE_COLS:
        val = col_input.number_input(
            label,
            value=0.0,
            step=0.1,
            format="%.4f",
            key=f"{key_prefix}_{col_name}_flt",
        )
        return float(val)

    val = col_input.text_input(
        label,
        value="",
        key=f"{key_prefix}_{col_name}_str",
    ).strip()
    return val if val else None

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
st.sidebar.caption("Source : Supabase" if _get_supabase_url() else "Source : SQLite local")

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

has_region_filter = len(selected_regions) < len(regions_disponibles)
region_mask = (
    df['brain_region'].isin(selected_regions)
    if has_region_filter
    else (df['brain_region'].isin(selected_regions) | df['brain_region'].isna())
)

df_filtered = df[
    region_mask &
    df['source'].isin(selected_sources)
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
        ["Leaky Integrate-and-Fire (Brian2)", "Hodgkin-Huxley / Ion current (Python / NEURON)"],
        horizontal=True,
        key="mod_formalism"
    )

    if formalism == "Leaky Integrate-and-Fire (Brian2)":
        refrac = round(ap_hw * 2.0, 2) if ap_hw else 2.0
        code_out = f'''from brian2 import *

# Parameters from Electrophysiology-database (n = {len(sub_df)})
# Brain area : {sel_region} | Preparation : {sel_prep} | Indentity : {sel_id}

E_L = {E_L} * mV            # Resting potential
V_th = {V_th} * mV          # AP threshold
V_reset = {V_reset} * mV    # Reset potential
R_in = {R_in} * Mohm        # Input resistance
tau_m = {tau_m} * ms        # Membrane time constant
I_inj = {I_rheo} * pA       # Injection current

eqs = """
dV/dt = (-(V - E_L) + R_in * I_inj) / tau_m : volt (unless refractory)
"""

neuron = NeuronGroup(1, eqs, threshold='V > V_th', reset='V = V_reset', refractory={refrac}*ms, method='exact')
neuron.V = E_L

mon = StateMonitor(neuron, 'V', record=True)
run(500 * ms)
'''
    else:
        area_str = f"{soma_area} um^2" if soma_area else "Non specified"
        g_max_val = f"{g_max} * nS" if g_max is not None else "None"
        v_half_val = f"{v_half} * mV" if v_half is not None else "None"
        slope_val = f"{slope_k} * mV" if slope_k is not None else "None"
        gamma_val = f"{gamma_sc} * pS" if gamma_sc is not None else "None"
        ec50_val = f"{ec50_ca} * uM" if ec50_ca is not None else "None"

        code_out = f'''# Biophysics and kinetic parameters for Hodgkin-Huxley / Ion current
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
# ONGLET 5 : AJOUTER UN NEURONE
# ==========================================

NON_NA_REQUIRED_COLS = {"cell_id", "source"}
AUTO_IGNORED_COLS = {"id", "created_at", "updated_at"}

FIELD_LABELS_BY_LANG = {
    "Français": {
        "cell_id": "Identifiant cellule (cell_id) *",
        "source": "Source bibliographique (source) *",
        "brain_region": "Région cérébrale (brain_region)",
        "cell_type": "Type / identité cellulaire",
        "known_identity": "Identité connue (known_identity)",
        "preparation": "Préparation (preparation)",
        "age_or_div": "Âge ou DIV (age_or_div)",
        "species": "Espèce",
        "n_cells": "Nombre de cellules (n_cells)",
        "reference_doi": "DOI / Référence (reference_doi)",
        "rmp_mv": "RMP (mV)",
        "tau_ms": "Constante de temps tau (ms)",
        "cm_pf": "Capacitance membranaire Cm (pF)",
        "rheobase_pa": "Rhéobase (pA)",
        "ap_threshold_mv": "Seuil PA (mV)",
        "ap_amplitude_mv": "Amplitude PA (mV)",
        "ap_half_width_ms": "Demi-largeur PA (ms)",
        "ahp_amplitude_mv": "Amplitude AHP (mV)",
        "ahp_duration_ms": "Durée AHP (ms)",
    },
    "English": {
        "cell_id": "Cell identifier (cell_id) *",
        "source": "Bibliographic source (source) *",
        "brain_region": "Brain region (brain_region)",
        "cell_type": "Cell type / identity",
        "known_identity": "Known identity (known_identity)",
        "preparation": "Preparation (preparation)",
        "age_or_div": "Age or DIV (age_or_div)",
        "species": "Species",
        "n_cells": "Number of cells (n_cells)",
        "reference_doi": "DOI / Reference (reference_doi)",
        "rmp_mv": "RMP (mV)",
        "tau_ms": "Membrane time constant tau (ms)",
        "cm_pf": "Membrane capacitance Cm (pF)",
        "rheobase_pa": "Rheobase (pA)",
        "ap_threshold_mv": "AP threshold (mV)",
        "ap_amplitude_mv": "AP amplitude (mV)",
        "ap_half_width_ms": "AP half-width (ms)",
        "ahp_amplitude_mv": "AHP amplitude (mV)",
        "ahp_duration_ms": "AHP duration (ms)",
    },
}


def render_field_with_na(col_name: str, series_dtype, key_prefix: str = "ins", lang: str = "Français"):
    """Rend un champ adapté au type de colonne avec option 'Non rapporté' / 'Not reported'."""
    t_local = LANGUAGES.get(lang, LANGUAGES["Français"])
    labels_map = FIELD_LABELS_BY_LANG.get(lang, FIELD_LABELS_BY_LANG["Français"])
    label = labels_map.get(col_name, col_name)
    is_required_no_na = col_name in NON_NA_REQUIRED_COLS

    if is_required_no_na:
        val = st.text_input(
            label,
            placeholder=t_local["t4_required_ph"],
            key=f"{key_prefix}_{col_name}_txt",
        ).strip()
        return val if val else None

    col_input, col_na = st.columns([4, 1.3], vertical_alignment="bottom")
    not_reported = col_na.checkbox(
        t_local["t4_not_reported"],
        value=True if col_name not in {"brain_region"} else False,
        key=f"{key_prefix}_{col_name}_na",
    )

    if not_reported:
        if pd.api.types.is_numeric_dtype(series_dtype) or col_name in FEATURE_COLS:
            col_input.number_input(
                label,
                value=0.0,
                disabled=True,
                key=f"{key_prefix}_{col_name}_num_na",
            )
        else:
            col_input.text_input(
                label,
                value="",
                disabled=True,
                key=f"{key_prefix}_{col_name}_txt_na",
            )
        return None

    if pd.api.types.is_integer_dtype(series_dtype) and col_name not in FEATURE_COLS:
        val = col_input.number_input(
            label,
            value=1,
            step=1,
            key=f"{key_prefix}_{col_name}_int",
        )
        return int(val)

    if pd.api.types.is_float_dtype(series_dtype) or col_name in FEATURE_COLS:
        val = col_input.number_input(
            label,
            value=0.0,
            step=0.1,
            format="%.4f",
            key=f"{key_prefix}_{col_name}_flt",
        )
        return float(val)

    val = col_input.text_input(
        label,
        value="",
        key=f"{key_prefix}_{col_name}_str",
    ).strip()
    return val if val else None


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


def delete_neuron_by_id(
    db_path: str,
    cell_id: str,
    table_name: str = "neurons_electrophys",
) -> int:
    """Supprime un neurone par cell_id (Supabase en priorité, repli sur SQLite)."""
    db_url = _get_supabase_url()
    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as pg_conn:
                cur = pg_conn.execute(
                    f"DELETE FROM {table_name} WHERE cell_id = %s",
                    (cell_id,),
                )
                pg_conn.commit()
                return cur.rowcount
        except Exception as exc:
            st.warning(f"Suppression Supabase indisponible ({exc}), repli sur SQLite local.")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {table_name} WHERE cell_id = ?",
            (cell_id,),
        )
        conn.commit()
        return cursor.rowcount
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


with tab5:
    st.subheader(t["t4_sub"])
    st.markdown(t["t4_desc"])

    if _has_write_access(lang_choice):
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
            options=[t["t4_mode_add"], t["t4_mode_update"], t["t4_mode_delete"]],
            horizontal=True,
            key="t4_mode_radio",
        )

        if mode == t["t4_mode_add"]:
            record_type = st.radio(
                t["t4_rec_type_label"],
                options=["single_cell", "group_summary"],
                format_func=lambda x: t["t4_rec_single"] if x == "single_cell" else t["t4_rec_group"],
                horizontal=True,
                key="t4_record_type_radio",
            )

            st.caption(t["t4_insert_caption"])

            excluded_in_form = AUTO_IGNORED_COLS | {
                "record_type",
                "n_cells",
                "dispersion_type",
                "reference_doi",
            }
            sd_cols = [c for c in df.columns if c.endswith("_sd")]
            excluded_in_form |= set(sd_cols)

            all_cols = [c for c in df.columns if c not in excluded_in_form]

            ordered_cols: list[str] = []
            for req in ["cell_id", "source"]:
                if req in all_cols:
                    ordered_cols.append(req)
            for c in all_cols:
                if c not in ordered_cols and c not in FEATURE_COLS:
                    ordered_cols.append(c)
            for c in FEATURE_COLS:
                if c in all_cols and c not in ordered_cols:
                    ordered_cols.append(c)

            record: dict[str, object] = {
                "record_type": record_type,
                "n_cells": 1,
                "dispersion_type": None,
                "reference_doi": None,
            }

            st.markdown(f"#### {t['t4_meta_sec']}")
            meta_cols = [c for c in ordered_cols if c not in FEATURE_COLS]
            for idx, col_name in enumerate(meta_cols):
                if idx % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if idx % 2 == 0 else c2):
                    val = render_field_with_na(
                        col_name, df[col_name].dtype, key_prefix="add", lang=lang_choice
                    )
                    if col_name == "known_identity" and isinstance(val, str):
                        harmonized = harmonize_identity_value(val)
                        val = harmonized if harmonized else "Unknown"
                    record[col_name] = val

            st.markdown(f"#### {t['t4_phys_sec']}")
            phys_cols = [c for c in ordered_cols if c in FEATURE_COLS]
            for idx, col_name in enumerate(phys_cols):
                if idx % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if idx % 2 == 0 else c2):
                    record[col_name] = render_field_with_na(
                        col_name, df[col_name].dtype, key_prefix="add", lang=lang_choice
                    )

            for sd_col in sd_cols:
                record[sd_col] = None

            if record_type == "group_summary":
                st.markdown(f"#### {t['t4_lit_sec']}")
                g1, g2, g3 = st.columns(3)
                with g1:
                    n_cells_val = render_field_with_na(
                        "n_cells", pd.Series(dtype="int64").dtype, key_prefix="add_grp", lang=lang_choice
                    )
                    record["n_cells"] = int(n_cells_val) if n_cells_val is not None else None
                with g2:
                    disp_na = st.checkbox(t["t4_disp_na"], value=True, key="add_grp_disp_na")
                    disp_type = st.selectbox(
                        t["t4_disp_type"],
                        ["SD", "SEM"],
                        disabled=disp_na,
                        key="add_grp_disp_type",
                    )
                    record["dispersion_type"] = None if disp_na else disp_type
                with g3:
                    doi_val = render_field_with_na(
                        "reference_doi", pd.Series(dtype="object").dtype, key_prefix="add_grp", lang=lang_choice
                    )
                    record["reference_doi"] = doi_val

                if sd_cols:
                    st.markdown(f"#### {t['t4_disp_sec']}")
                    for idx, sd_col in enumerate(sd_cols):
                        if idx % 2 == 0:
                            s1, s2 = st.columns(2)
                        with (s1 if idx % 2 == 0 else s2):
                            raw_disp = render_field_with_na(
                                sd_col, df[sd_col].dtype, key_prefix="add_sd", lang=lang_choice
                            )
                            if raw_disp is not None and record["dispersion_type"] is not None:
                                record[sd_col] = to_sd(
                                    value=float(raw_disp),
                                    dispersion_type=record["dispersion_type"],
                                    n_cells=record["n_cells"],
                                )
                            else:
                                record[sd_col] = raw_disp

            if st.button(t["t4_btn"], type="primary", key="t4_add_submit_btn"):
                clean_id = (record.get("cell_id") or "").strip() if isinstance(record.get("cell_id"), str) else ""
                clean_source = (record.get("source") or "").strip() if isinstance(record.get("source"), str) else ""

                if not clean_id:
                    st.error(t["t4_err_id"])
                elif not clean_source:
                    st.error(t["t4_err_source"])
                else:
                    record["cell_id"] = clean_id
                    record["source"] = clean_source
                    insert_neuron_record(db_path=DB_PATH, record=record)
                    st.cache_data.clear()
                    st.success(t["t4_success"].format(clean_id))

        elif mode == t["t4_mode_update"]:
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

                def _initial_str(col_name: str) -> str:
                    if col_name not in current_row.index:
                        return ""
                    val = current_row.get(col_name)
                    return "" if pd.isna(val) else str(val)

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
                            value=_initial_str("age_or_div"),
                        )
                    with u2:
                        upd_rheo_txt = st.text_input(
                            t["t4_rheo"],
                            value=_initial_str("rheobase_pa"),
                        )

                    st.markdown(f"**{t['t4_morpho_section']}**")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        upd_branches_txt = st.text_input(
                            t["t4_num_branches"],
                            value=_initial_str("num_branches"),
                        )
                        upd_junctions_txt = st.text_input(
                            t["t4_num_junctions"],
                            value=_initial_str("num_junctions"),
                        )
                    with m2:
                        upd_maxlen_txt = st.text_input(
                            t["t4_max_branch_len"],
                            value=_initial_str("max_branch_length_um"),
                        )
                        upd_critrad_txt = st.text_input(
                            t["t4_crit_radius"],
                            value=_initial_str("critical_radius_um"),
                        )
                    with m3:
                        upd_sholl_txt = st.text_input(
                            t["t4_sholl"],
                            value=_initial_str("sholl_intersections"),
                        )

                    upd_submitted = st.form_submit_button(t["t4_update_btn"])

                    if upd_submitted:
                        def _to_float_or_none(raw_val: str) -> float | None:
                            s = raw_val.strip()
                            if not s:
                                return None
                            try:
                                return float(s.replace(",", "."))
                            except ValueError:
                                return None

                        clean_age = upd_age.strip()
                        payload: dict[str, object] = {
                            "known_identity": None if upd_ident == "Unknown" else upd_ident,
                            "age_or_div": clean_age if clean_age else None,
                            "rheobase_pa": _to_float_or_none(upd_rheo_txt),
                            "num_branches": _to_float_or_none(upd_branches_txt),
                            "num_junctions": _to_float_or_none(upd_junctions_txt),
                            "max_branch_length_um": _to_float_or_none(upd_maxlen_txt),
                            "critical_radius_um": _to_float_or_none(upd_critrad_txt),
                            "sholl_intersections": _to_float_or_none(upd_sholl_txt),
                        }

                        non_empty_payload = {
                            k: v for k, v in payload.items() if v is not None
                        }

                        rows_updated = update_neuron(
                            cell_id=selected_cell,
                            db_path=DB_PATH,
                            **non_empty_payload,
                        )
                        st.cache_data.clear()
                        if rows_updated > 0:
                            st.success(t["t4_update_success"].format(selected_cell))
                            st.rerun()
                        else:
                            st.info(t["t4_update_noop"].format(selected_cell))

        elif mode == t["t4_mode_delete"]:
            if df.empty or "cell_id" not in df.columns:
                st.info(t["t4_no_cells"])
            else:
                st.warning(t["t4_delete_warn"])
                cell_ids = df["cell_id"].dropna().astype(str).unique().tolist()
                selected_to_del = st.selectbox(
                    t["t4_select_cell"],
                    options=cell_ids,
                    key="t4_delete_cell_select",
                )

                preview_cols = [
                    c
                    for c in [
                        "cell_id",
                        "source",
                        "brain_region",
                        "preparation",
                        "known_identity",
                        "record_type",
                    ]
                    if c in df.columns
                ]
                st.dataframe(
                    df.loc[df["cell_id"].astype(str) == selected_to_del, preview_cols],
                    hide_index=True,
                    use_container_width=True,
                )

                confirmed = st.checkbox(
                    t["t4_delete_confirm"],
                    value=False,
                    key=f"t4_del_confirm_{selected_to_del}",
                )

                if st.button(
                    t["t4_delete_btn"],
                    type="primary",
                    disabled=not confirmed,
                    key="t4_delete_submit_btn",
                ):
                    if not confirmed:
                        st.error(t["t4_delete_err_confirm"])
                    else:
                        deleted_count = delete_neuron_by_id(
                            db_path=DB_PATH,
                            cell_id=selected_to_del,
                        )
                        st.cache_data.clear()
                        if deleted_count > 0:
                            st.success(t["t4_delete_success"].format(selected_to_del))
                            st.rerun()
                        else:
                            st.info(t["t4_update_noop"].format(selected_to_del))

# ==========================================
# ONGLET 4 : MODÉLISATION IN SILICO
# ==========================================
with tab4:
    render_modeling_tab(df_filtered, t)

                # streamlit run app.py
                # pour l'arrêter, ctrl + C