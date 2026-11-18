import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def run_pca_analysis(db_path="data/electrophys_database.db"):
    # 1. Connexion à SQLite et chargement des données
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
    conn.close()
    
    print(f"Nombre total de neurones chargés pour la PCA : {len(df)}")
    
    # 2. Choix des caractéristiques électrophysiologiques et passives
    feature_cols = [
        'rmp_mv', 'tau_ms', 'cm_pf', 
        'rheobase_pa', 'ap_threshold_mv', 'ap_amplitude_mv', 
        'ap_half_width_ms', 'ahp_amplitude_mv', 'ahp_duration_ms'
    ]
    
    # Nettoyage des virgules européennes et conversion en float
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extraction des features et gestion des valeurs manquantes (imputation par la médiane)
    X = df[feature_cols]
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # 3. Standardisation indispensable pour la PCA (moyenne = 0, variance = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 4. Exécution de la PCA (on garde les 2 premières composantes principales)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Ajout des coordonnées PCA dans le DataFrame
    df['PC1'] = principal_components[:, 0]
    df['PC2'] = principal_components[:, 1]
    
    # Affichage de la variance expliquée
    explained_variance = pca.explained_variance_ratio_ * 100
    print(f"\n--- VARIANCE EXPLIQUÉE PAR LES AXES ---")
    print(f"PC1 : {explained_variance[0]:.2f}% de la variance")
    print(f"PC2 : {explained_variance[1]:.2f}% de la variance")
    print(f"Total cumulé : {sum(explained_variance):.2f}%")
    
    # 5. Visualisation graphique : Culture vs Tranches aiguës
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='preparation', 
        style='age_or_div', 
        data=df, 
        palette='Set1', 
        s=80, 
        alpha=0.8
    )
    
    plt.title("ACP Comparative : Tranches Aiguës vs Cultures Organotypiques (DRN)", fontsize=12, fontweight='bold')
    plt.xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)", fontsize=10)
    plt.ylabel(f"PC2 ({explained_variance[1]:.1f}% variance)", fontsize=10)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Préparation / Âge")
    plt.tight_layout()
    
    # Sauvegarde du graphique
    plt.savefig("data/pca_preparation_comparison.png", dpi=300)
    print("\nGraphique de l'ACP sauvegardé dans 'data/pca_preparation_comparison.png'.")
    plt.show()

    # 6. Analyse des contributions (Loadings) pour comprendre ce que mesurent PC1 et PC2
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=feature_cols)
    print("\n--- CONTRIBUTIONS DES VARIABLES (Loadings) ---")
    print(loading_matrix.to_string())

if __name__ == "__main__":
    run_pca_analysis()