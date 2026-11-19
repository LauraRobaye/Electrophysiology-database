from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from identity import harmonize_identity_series

import sqlite3
import pandas as pd
import os

def import_existing_csv(csv_path="data/Table_ACP.csv", db_path="data/electrophys_database.db"):
    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier {csv_path} est introuvable.")
        print("Vérifie que ton fichier CSV est bien placé dans le dossier 'data/'.")
        return
        
    print(f"Lecture de ton fichier CSV : {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    print(f"Nombre total de neurones trouvés dans le CSV : {len(df)}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    records = []
    for _, row in df.iterrows():
        # 1. Identifiants et métadonnées (avec gestion souple des noms de colonnes)
        cell_id = str(row.get('Neurone', row.get('cell_id', 'Unknown_Cell')))
        source = 'My_Recordings'
        brain_region = 'DRN'
        
        # Gestion flexible de la préparation
        raw_prep = str(row.get('Préparation', row.get('Preparation', row.get('Type', 'Organotypic_Slice')))).strip().lower()
        if 'acute' in raw_prep or 'aigu' in raw_prep:
            preparation = 'Acute_Slice'
        else:
            preparation = 'Organotypic_Slice'
            
        age_or_div = str(row.get('Age_DIV', 'Unknown'))
        
        # Normalisation de l'identité
        raw_id = row.get('Identity', row.get('Identite', row.get('known_identity', 'Unknown')))
        harmonized_id = harmonize_identity_value(raw_id)
        known_identity = harmonized_id if harmonized_id else 'Unknown'
            
        sex = str(row.get('Sex', 'Unknown'))

        # 2. Propriétés Électrophysiologiques et Passives
        rmp_mv = row.get('Membrane_potential')
        tau_ms = row.get('Tau')
        cm_pf = row.get('Capacitance')
        rin_mohm = row.get('Résistance', row.get('Resistance'))
        rheobase_pa = row.get('Rheobase')
        ap_threshold_mv = row.get('Threshold')
        ap_amplitude_mv = row.get('Amplitude')
        ap_half_width_ms = row.get('Half_width')
        ahp_amplitude_mv = row.get('AHP_amplitude')
        ahp_duration_ms = row.get('AHP_duration')
        
        # 3. Morphologie (absente de ce CSV, donc None)
        num_branches = None
        num_junctions = None
        max_branch_length_um = None
        critical_radius_um = None
        sholl_intersections = None

        records.append((
            cell_id, source, brain_region, preparation, age_or_div, known_identity,
            rmp_mv, tau_ms, cm_pf, rheobase_pa, ap_threshold_mv, ap_amplitude_mv,
            ap_half_width_ms, ahp_amplitude_mv, ahp_duration_ms,
            num_branches, num_junctions, max_branch_length_um, critical_radius_um, sholl_intersections
        ))

    # Insertion en masse dans SQLite
    cursor.executemany("""
    INSERT OR REPLACE INTO neurons_electrophys VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """, records)
    
    conn.commit()
    conn.close()
    print(f"Succès ! {len(records)} neurones importés avec succès dans la base SQLite.")

if __name__ == "__main__":
    import_existing_csv()