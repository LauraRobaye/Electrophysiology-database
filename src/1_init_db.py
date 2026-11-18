import sqlite3
import os

def create_database(db_path="data/electrophys_database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS neurons_electrophys (
        cell_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,          -- 'My_Organotypic', 'Literature_DRN'
        brain_region TEXT,             -- 'DRN'
        preparation TEXT,              -- 'Organotypic_Slice', 'Acute_Slice'
        age_or_div TEXT,               -- 'DIV1', 'DIV7', 'DIV14', 'Adult', etc.
        known_identity TEXT,           -- '5-HT', 'Non-5-HT', 'Unknown'
        
        -- Propriétés Passives & Électrophysiologiques
        rmp_mv REAL,                   -- Potentiel de membrane au repos (mV)
        tau_ms REAL,                   -- Tau (ms)
        cm_pf REAL,                    -- Capacitance membranaire (pF)
        rheobase_pa REAL,              -- Rhéobase (pA)
        ap_threshold_mv REAL,          -- Threshold (mV)
        ap_amplitude_mv REAL,          -- Amplitude du potentiel d'action (mV)
        ap_half_width_ms REAL,         -- Half width (ms)
        ahp_amplitude_mv REAL,         -- Amplitude de l'AHP (mV)
        ahp_duration_ms REAL,          -- Durée de l'AHP (ms)
        
        -- Propriétés Morphologiques
        num_branches INTEGER,          -- Nombre de branches
        num_junctions INTEGER,         -- Nombre de jonctions
        max_branch_length_um REAL,     -- Longueur maximale des branches (µm)
        critical_radius_um REAL,       -- Rayon critique (µm)
        sholl_intersections REAL       -- Sholl intersections
    );
    """)

    conn.commit()
    conn.close()
    print(f"Base de données SQLite initialisée avec succès : {db_path}")

if __name__ == "__main__":
    create_database()