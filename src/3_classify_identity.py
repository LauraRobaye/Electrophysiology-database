import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def run_identity_classifier(db_path="data/electrophys_database.db"):
    # 1. Connexion à SQLite et chargement des données
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
    conn.close()
    
    print(f"Nombre total de neurones chargés depuis la base : {len(df)}")
    
    # 2. Choix des caractéristiques (Features) électrophysiologiques
    feature_cols = [
        'rmp_mv', 'tau_ms', 'cm_pf', 
        'rheobase_pa', 'ap_threshold_mv', 'ap_amplitude_mv', 
        'ap_half_width_ms', 'ahp_amplitude_mv', 'ahp_duration_ms'
    ]
    
    # Nettoyage des virgules européennes (ex: '-56,83' -> '-56.83') et conversion en float
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filtrage des cellules ayant une étiquette d'identité connue vs inconnue
    df_train = df[df['known_identity'].isin(['Serotonin', 'Non_serotonin'])].copy()
    df_pred = df[df['known_identity'] == 'Unknown'].copy()
    
    print(f"Neurones d'entraînement (connus) : {len(df_train)}")
    print(f"Neurones à prédire (inconnus / Unknown) : {len(df_pred)}")
    
    if len(df_train) < 5:
        print("Attention : Pas assez de neurones étiquetés ('5-HT' ou 'Non-5-HT') pour entraîner un modèle.")
        return

    # Préparation des données d'entraînement
    X_train = df_train[feature_cols]
    y_train = df_train['known_identity']
    
    # 3. Traitement des valeurs manquantes (imputation par la médiane)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # 4. Évaluation du modèle par validation croisée (Stratified K-Fold)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_imputed, y_train, cv=cv, scoring='accuracy')
    
    print(f"\n--- ÉVALUATION DU MODÈLE ---")
    print(f"Scores de précision (Accuracy) par pli : {cv_scores}")
    print(f"Précision moyenne : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    
    # 5. Entraînement final sur tout le dataset étiqueté
    rf.fit(X_train_imputed, y_train)
    
    # 6. Extraction de l'importance des variables (Feature Importance)
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    print(f"\n--- IMPORTANCE DES VARIABLES ---")
    print(feat_imp.to_string(index=False))
    
    # 7. Prédiction sur les neurones "Unknown"
    if len(df_pred) > 0:
        X_pred = df_pred[feature_cols]
        X_pred_imputed = imputer.transform(X_pred)
        
        probabilities = rf.predict_proba(X_pred_imputed)
        classes = rf.classes_
        
        idx_5ht = list(classes).index('Serotonin') if 'Serotonin' in classes else 0
        prob_5ht = probabilities[:, idx_5ht]
        
        predictions = rf.predict(X_pred_imputed)
        
        df_pred['predicted_identity'] = predictions
        df_pred['prob_serotoninergic'] = prob_5ht
        
        print(f"\n--- RÉSULTATS DE PRÉDICTION POUR LES NEURONES INCONNUS ---")
        summary_preds = df_pred[['cell_id', 'preparation', 'age_or_div', 'predicted_identity', 'prob_serotoninergic']]
        print(summary_preds.head(15).to_string(index=False))
        
        # Sauvegarde dans SQLite
        df_pred.to_sql('predicted_neurons_results', sqlite3.connect(db_path), if_exists='replace', index=False)
        print("\nRésultats sauvegardés dans la table SQLite 'predicted_neurons_results'.")

    # 8. Visualisation graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title("Importance des paramètres électrophysiologiques (Random Forest)")
    plt.xlabel("Importance relative")
    plt.ylabel("Paramètre")
    plt.tight_layout()
    plt.savefig("data/feature_importance_rf.png", dpi=300)
    print("\nGraphique sauvegardé dans 'data/feature_importance_rf.png'.")
    plt.show()

if __name__ == "__main__":
    run_identity_classifier()