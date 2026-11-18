import sqlite3

databases = ["data/public_electrophys.db", "data/private_electrophys.db"]

# Regroupe toutes les colonnes : métadonnées de modèle, expression et cinétique de canaux
all_new_cols = [
    ("expression_system", "TEXT DEFAULT 'Native'"),
    ("channel_target", "TEXT DEFAULT 'Whole-cell native'"),
    ("genotype_status", "TEXT DEFAULT 'WT'"),
    ("mutation_details", "TEXT DEFAULT 'None'"),
    ("g_max_ns", "REAL"),
    ("current_density_pa_pf", "REAL"),
    ("single_channel_conductance_ps", "REAL"),
    ("v_half_act_mv", "REAL"),
    ("slope_factor_k_mv", "REAL"),
    ("ec50_ca_um", "REAL")
]

for db_path in databases:
    print(f"Mise à jour de : {db_path} ...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for col_name, col_type in all_new_cols:
        try:
            cursor.execute(f"ALTER TABLE neurons_electrophys ADD COLUMN {col_name} {col_type};")
            print(f" -> Colonne ajoutée : {col_name}")
        except sqlite3.OperationalError:
            print(f" -> (Déjà existante) : {col_name}")
    conn.commit()
    conn.close()

print("\nMise à jour complète du schéma de la base de données réussie !")