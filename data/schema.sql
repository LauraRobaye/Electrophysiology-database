-- Schéma de la base de données SQLite : electrophys_database.db
-- Généré automatiquement pour le projet de thèse

CREATE TABLE neurons_electrophys (
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

CREATE TABLE "predicted_neurons_results" (
"cell_id" TEXT,
  "source" TEXT,
  "brain_region" TEXT,
  "preparation" TEXT,
  "age_or_div" TEXT,
  "known_identity" TEXT,
  "rmp_mv" REAL,
  "tau_ms" REAL,
  "cm_pf" REAL,
  "rheobase_pa" REAL,
  "ap_threshold_mv" REAL,
  "ap_amplitude_mv" REAL,
  "ap_half_width_ms" REAL,
  "ahp_amplitude_mv" REAL,
  "ahp_duration_ms" REAL,
  "num_branches" TEXT,
  "num_junctions" TEXT,
  "max_branch_length_um" TEXT,
  "critical_radius_um" TEXT,
  "sholl_intersections" TEXT,
  "predicted_identity" TEXT,
  "prob_serotoninergic" REAL
);

