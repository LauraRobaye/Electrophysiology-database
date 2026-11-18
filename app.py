import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration de la page
st.set_page_config(
    page_title="Neuro-Electrophys Comparator",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# DESIGN SYSTEM : BRUTALISME & BLEU KLEIN
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
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
    conn.close()
    
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# --- BARRE LATÉRALE (FILTRES GLOBAUX) ---
st.sidebar.title("PARAMÈTRES")
st.sidebar.markdown("---")
st.sidebar.markdown("Filtres d'exploration de la base locale.")

regions_disponibles = df['brain_region'].dropna().unique().tolist()
selected_regions = st.sidebar.multiselect(
    "Région(s) cérébrale(s) :",
    options=regions_disponibles,
    default=regions_disponibles
)

sources_disponibles = df['source'].dropna().unique().tolist()
selected_sources = st.sidebar.multiselect(
    "Source(s) des données :",
    options=sources_disponibles,
    default=sources_disponibles
)

df_filtered = df[
    (df['brain_region'].isin(selected_regions)) & 
    (df['source'].isin(selected_sources))
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Neurones sélectionnés", len(df_filtered))

# --- EN-TÊTE PRINCIPAL ---
st.title("NEURO-ELECTROPHY COMPARATOR")
st.markdown("""
> **Module d'analyse comparative & prédiction phénotypique** — Tranches aiguës vs Cultures organotypiques (DRN).
""")
st.markdown("---")

# --- ONGLETS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "01 // DATABASE", 
    "02 // ACP (PCA)", 
    "03 // RANDOM FOREST", 
    "04 // INSERTION"
])

# ==========================================
# ONGLET 1 : EXPLORATION DES DONNÉES
# ==========================================
with tab1:
    st.subheader("Aperçu du jeu de données filtré")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total neurones", len(df_filtered))
    col2.metric("Préparations", df_filtered['preparation'].nunique())
    col3.metric("Identités connues", len(df_filtered[df_filtered['known_identity'] != 'Unknown']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le jeu filtré (CSV)",
        data=csv,
        file_name='electrophys_filtered_export.csv',
        mime='text/csv',
    )

# ==========================================
# ONGLET 2 : ACP INTERACTIVE
# ==========================================
with tab2:
    st.subheader("Analyse en Composantes Principales (PCA)")
    
    if len(df_filtered) < 5:
        st.warning("Sélectionnez au moins 5 neurones dans les filtres pour calculer l'ACP.")
    else:
        col_opt1, col_opt2 = st.columns(2)
        color_by = col_opt1.selectbox("Colorer par :", ['preparation', 'known_identity', 'brain_region', 'source'])
        symbol_by = col_opt2.selectbox("Symbole par :", ['age_or_div', 'preparation', 'known_identity'])
        
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
        # Harmonisation des couleurs du graphique avec le Bleu Klein et fond gris clair
        fig.update_layout(
            plot_bgcolor='#F4F4F4',
            paper_bgcolor='#F4F4F4',
            font=dict(color='#111111', family="sans-serif"),
            title=dict(text="PROJECTION MULTIDIMENSIONNELLE", font=dict(color='#002FA7', size=16, family="sans-serif"))
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
    st.subheader("Simulateur de classification phénotypique")
    st.markdown("Estimation de l'identité neurochimique via Random Forest sur les paramètres électrophysiologiques.")
    
    df_train = df[df['known_identity'].isin(['Serotonin', 'Non_serotonin'])].copy()
    
    if len(df_train) < 10:
        st.error("Pas assez de neurones étiquetés dans la base pour entraîner le modèle.")
    else:
        imputer_rf = SimpleImputer(strategy='median')
        X_train = imputer_rf.fit_transform(df_train[FEATURE_COLS])
        y_train = df_train['known_identity']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            rmp = st.number_input("RMP (mV)", value=-62.0, step=1.0)
            tau = st.number_input("Tau (ms)", value=18.0, step=1.0)
            cm = st.number_input("Capacitance Cm (pF)", value=40.0, step=1.0)
        with c2:
            rheo = st.number_input("Rhéobase (pA)", value=45.0, step=5.0)
            thr = st.number_input("Seuil du PA (mV)", value=-39.0, step=1.0)
            amp = st.number_input("Amplitude du PA (mV)", value=75.0, step=1.0)
        with c3:
            hw = st.number_input("Demi-largeur PA (ms)", value=1.4, step=0.1)
            ahp_amp = st.number_input("Amplitude AHP (mV)", value=11.0, step=0.5)
            ahp_dur = st.number_input("Durée AHP (ms)", value=45.0, step=2.0)
            
        input_data = np.array([[rmp, tau, cm, rheo, thr, amp, hw, ahp_amp, ahp_dur]])
        proba = rf.predict_proba(input_data)[0]
        classes = list(rf.classes_)
        
        idx_5ht = classes.index('Serotonin')
        prob_5ht = proba[idx_5ht] * 100
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        prediction_label = "Serotonin" if prob_5ht >= 50 else "Non_serotonin"
        res_col1.success(f"### PRÉDICTION : **{prediction_label}**")
        res_col2.info(f"### PROBABILITÉ SÉROTONERGIQUE : **{prob_5ht:.1f} %**")

# ==========================================
# ONGLET 4 : AJOUTER UN NEURONE
# ==========================================
with tab4:
    st.subheader("Encoder un nouvel enregistrement")
    st.markdown("Injection directe d'une cellule dans la base de données SQLite locale.")
    
    with st.form("new_neuron_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            new_id = st.text_input("ID de la cellule (ex: Lab_N260)*")
            new_source = st.selectbox("Source", ["Lab_ULiege", "Literature", "Allen_Institute", "Other"])
            new_region = st.text_input("Région cérébrale", value="DRN")
        with f_col2:
            new_prep = st.selectbox("Préparation", ["Acute_Slice", "Organotypic_Slice", "Dissociated_Culture"])
            new_age = st.text_input("Âge ou DIV", value="DIV7")
            new_ident = st.selectbox("Identité connue", ["Unknown", "Serotonin", "Non_serotonin"])
        with f_col3:
            new_rmp = st.number_input("RMP (mV)", value=-60.0)
            new_tau = st.number_input("Tau (ms)", value=15.0)
            new_cm = st.number_input("Cm (pF)", value=35.0)
            
        f_col4, f_col5 = st.columns(2)
        with f_col4:
            new_rheo = st.number_input("Rhéobase (pA)", value=40.0)
            new_thr = st.number_input("Seuil PA (mV)", value=-40.0)
            new_amp = st.number_input("Amplitude PA (mV)", value=75.0)
        with f_col5:
            new_hw = st.number_input("Demi-largeur PA (ms)", value=1.2)
            new_ahp_amp = st.number_input("Amplitude AHP (mV)", value=10.0)
            new_ahp_dur = st.number_input("Durée AHP (ms)", value=40.0)
            
        submitted = st.form_submit_button("💾 ENREGISTRER DANS LA BASE")
        
        if submitted:
            if not new_id:
                st.error("L'ID de la cellule est obligatoire !")
            else:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO neurons_electrophys VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL
                    )
                """, (new_id, new_source, new_region, new_prep, new_age, new_ident,
                      new_rmp, new_tau, new_cm, new_rheo, new_thr, new_amp, new_hw, new_ahp_amp, new_ahp_dur))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.success(f"Le neurone **{new_id}** a été ajouté avec succès !")

                # streamlit run app.py
                # pour l'arrêter, ctrl + C