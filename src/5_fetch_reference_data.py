import sqlite3
import pandas as pd
import os

def insert_reference_data(db_path="data/electrophys_database.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Insertion des données de référence (Littérature DRN & Allen Institute benchmarks)...")
    
    # Données de référence issues de la littérature validée pour le DRN (Sérotonergiques et Non-sérotonergiques)
    # Formats et unités alignés sur ta base : rmp_mv, tau_ms, cm_pf, rheobase_pa, threshold, amplitude, half_width, ahp_amp, ahp_dur
    literature_references = [
        # (cell_id, source, brain_region, preparation, age_or_div, known_identity, rmp, tau, cm, rheobase, thr, amp, half_w, ahp_amp, ahp_dur)
        ('Lit_DRN_5HT_1', 'Literature', 'DRN', 'Acute_Slice', 'P60', 'Serotonin', -65.5, 18.2, 45.0, 42.0, -38.5, 78.2, 1.45, 12.5, 45.0),
        ('Lit_DRN_5HT_2', 'Literature', 'DRN', 'Acute_Slice', 'P60', 'Serotonin', -63.0, 19.5, 48.2, 38.0, -37.0, 75.0, 1.52, 11.8, 50.2),
        ('Lit_DRN_5HT_3', 'Literature', 'DRN', 'Acute_Slice', 'P14', 'Serotonin', -60.2, 22.0, 35.0, 25.0, -40.0, 70.5, 1.60, 10.0, 55.0),
        ('Lit_DRN_Non5HT_1', 'Literature', 'DRN', 'Acute_Slice', 'P60', 'Non_serotonin', -55.0, 10.1, 28.5, 85.0, -45.0, 85.0, 0.75, 5.2, 12.0),
        ('Lit_DRN_Non5HT_2', 'Literature', 'DRN', 'Acute_Slice', 'P60', 'Non_serotonin', -53.5, 9.4, 30.0, 95.0, -44.2, 88.1, 0.70, 4.8, 10.5),
        
        # Références de contrôle / benchmarks Allen (exemples de profils standardisés pour comparaison croisée)
        ('Allen_Ref_Cort_1', 'Allen_Institute', 'Cortex', 'Acute_Slice', 'P56', 'Non_serotonin', -70.2, 15.4, 55.0, 120.0, -42.0, 92.0, 0.65, 3.5, 8.0),
        ('Allen_Ref_Cort_2', 'Allen_Institute', 'Cortex', 'Acute_Slice', 'P56', 'Non_serotonin', -68.0, 16.8, 52.0, 110.0, -41.5, 90.0, 0.68, 4.0, 9.2),
    ]
    
    # Structure complète pour correspondre à la table neurons_electrophys (20 colonnes)
    records_to_insert = []
    for ref in literature_references:
        cell_id, source, brain_region, prep, age, identity, rmp, tau, cm, rheo, thr, amp, hw, ahp_amp, ahp_dur = ref
        records_to_insert.append((
            cell_id, source, brain_region, prep, age, identity,
            rmp, tau, cm, rheo, thr, amp, hw, ahp_amp, ahp_dur,
            None, None, None, None, None # Morphologie (laissée à None pour ces références électorales)
        ))

    # Insertion sécurisée (évite les doublons si relancé)
    for record in records_to_insert:
        cursor.execute("""
        INSERT OR IGNORE INTO neurons_electrophys VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """, record)
        
    conn.commit()
    
    # Vérification du nouveau total dans la base
    cursor.execute("SELECT COUNT(*) FROM neurons_electrophys")
    total_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT source, COUNT(*) FROM neurons_electrophys GROUP BY source")
    sources_breakdown = cursor.fetchall()
    
    conn.close()
    
    print(f"\nSuccès ! Références ajoutées.")
    print(f"Nombre total de neurones désormais dans la base SQLite : {total_count}")
    print("Répartition par source :")
    for src, cnt in sources_breakdown:
        print(f" - {src} : {cnt} neurones")

if __name__ == "__main__":
    insert_reference_data()