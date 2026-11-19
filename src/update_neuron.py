from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from identity import harmonize_identity_value


def update_neuron(
    cell_id: str,
    db_path: str = "data/public_electrophys.db",
    table_name: str = "neurons_electrophys",
    **fields: Any,
) -> int:
    """
    Met à jour un ou plusieurs champs d'un neurone existant sans écraser
    les colonnes non fournies (laissées à None ou omises).
    """
    if not cell_id or not str(cell_id).strip():
        raise ValueError("cell_id ne peut pas être vide.")

    # Harmonise automatiquement known_identity si fourni
    if "known_identity" in fields and fields["known_identity"] is not None:
        fields["known_identity"] = harmonize_identity_value(fields["known_identity"])

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Vérifie les colonnes existantes de la table pour éviter toute faute de frappe
        existing_cols = {
            row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

        updates = {
            col: val
            for col, val in fields.items()
            if val is not None and col in existing_cols and col != "cell_id"
        }

        if not updates:
            print("Aucune colonne valide à mettre à jour (toutes à None ou absentes de la table).")
            return 0

        set_clause = ", ".join([f"{col} = ?" for col in updates])
        params = list(updates.values()) + [str(cell_id).strip()]

        cursor.execute(
            f"UPDATE {table_name} SET {set_clause} WHERE cell_id = ?",
            tuple(params),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


if __name__ == "__main__":
    # Exemples d'utilisation :

    # 1. Morphologie + identité
    n1 = update_neuron(
        cell_id="CELL_001",
        known_identity="Serotonin",
        num_branches=14,
        max_branch_length_um=210.5,
    )

    # 2. Uniquement un paramètre électrophysiologique ou une métadonnée
    n2 = update_neuron(
        cell_id="CELL_002",
        rheobase_pa=55.0,
        age_or_div="DIV14",
    )

    print(f"Mises à jour -> CELL_001: {n1}, CELL_002: {n2}")