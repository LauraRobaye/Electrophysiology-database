from __future__ import annotations

import sqlite3
import sys
import math
import os
import pandas as pd
import psycopg
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from identity import harmonize_identity_value

def to_sd(
    value: float | None,
    dispersion_type: str | None,
    n_cells: int | None,
) -> float | None:
    """Convertit une dispersion rapportée (SD ou SEM) en écart type (SD)."""
    if value is None:
        return None
    disp = (dispersion_type or "SD").strip().upper()
    if disp == "SEM" and n_cells is not None and n_cells > 0:
        return float(value) * math.sqrt(n_cells)
    return float(value)

def update_neuron(
    cell_id: str,
    db_path: str = "data/public_electrophys.db",
    table_name: str = "neurons_electrophys",
    **fields: Any,
) -> int:
    """
    Met à jour un ou plusieurs champs d'un neurone existant en priorité sur
    Supabase, avec repli sur SQLite local. Les champs à None ou absents de la
    table sont ignorés afin de ne pas écraser de données existantes.
    """
    clean_id = str(cell_id).strip() if cell_id is not None else ""
    if not clean_id:
        raise ValueError("cell_id ne peut pas être vide.")

    cleaned_fields: dict[str, Any] = {}
    for col, val in fields.items():
        if col == "cell_id":
            continue
        norm_val = _normalize_sql_value(val)
        if norm_val is not None:
            cleaned_fields[col] = norm_val

    if "known_identity" in cleaned_fields:
        harmonized = harmonize_identity_value(cleaned_fields["known_identity"])
        cleaned_fields["known_identity"] = harmonized if harmonized else "Unknown"

    if not cleaned_fields:
        print("Aucune colonne valide à mettre à jour (toutes à None ou vides).")
        return 0

    db_url = _resolve_supabase_url()
    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as pg_conn:
                existing_cols = {
                    r[0]
                    for r in pg_conn.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        """,
                        (table_name,),
                    ).fetchall()
                }
                updates = {
                    col: val
                    for col, val in cleaned_fields.items()
                    if col in existing_cols
                }
                if not updates:
                    print(
                        f"Aucune colonne correspondante dans la table Supabase '{table_name}'."
                    )
                    return 0

                set_clause = ", ".join(f'"{col}" = %s' for col in updates)
                params = list(updates.values()) + [clean_id]

                cur = pg_conn.execute(
                    f'UPDATE "{table_name}" SET {set_clause} WHERE "cell_id" = %s',
                    tuple(params),
                )
                pg_conn.commit()
                return cur.rowcount
        except Exception as exc:
            print(f"Mise à jour Supabase indisponible ({exc}), repli sur SQLite local.")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        existing_cols = {
            row[1]
            for row in cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        }
        updates = {
            col: val
            for col, val in cleaned_fields.items()
            if col in existing_cols
        }
        if not updates:
            print(
                f"Aucune colonne correspondante dans la table SQLite '{table_name}'."
            )
            return 0

        set_clause = ", ".join(f'"{col}" = ?' for col in updates)
        params = list(updates.values()) + [clean_id]

        cursor.execute(
            f'UPDATE "{table_name}" SET {set_clause} WHERE "cell_id" = ?',
            tuple(params),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

def _resolve_supabase_url() -> str | None:
    try:
        import streamlit as st
        url = str(st.secrets.get("SUPABASE_DB_URL", "")).strip()
        if url:
            return url
    except Exception:
        pass
    env_url = (os.environ.get("SUPABASE_DB_URL") or "").strip()
    return env_url or None


def _normalize_sql_value(val: Any) -> Any:
    """Convertit NaN ou chaîne vide en None pour produire un vrai NULL en SQL."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and val.strip() == "":
        return None
    return val


def insert_neuron_record(
    db_path: str = "data/public_electrophys.db",
    record: dict[str, Any] | None = None,
    table_name: str = "neurons_electrophys",
) -> None:
    """
    Insère un neurone ou un résumé de groupe dans neurons_electrophys.
    Ne conserve que les colonnes existantes de la table et garde None -> NULL en SQL.
    """
    if not record:
        return

    clean_record = dict(record)
    cell_id = _normalize_sql_value(clean_record.get("cell_id"))
    if not cell_id:
        raise ValueError("cell_id ne peut pas être vide.")
    clean_record["cell_id"] = str(cell_id).strip()

    if "known_identity" in clean_record:
        harm_id = _normalize_sql_value(clean_record.get("known_identity"))
        if harm_id is not None:
            harmonized = harmonize_identity_value(harm_id)
            clean_record["known_identity"] = harmonized if harmonized else "Unknown"
        else:
            clean_record["known_identity"] = None

    db_url = _resolve_supabase_url()
    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=5) as pg_conn:
                existing_cols = {
                    r[0]
                    for r in pg_conn.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        """,
                        (table_name,),
                    ).fetchall()
                }
                valid_items = [
                    (col, _normalize_sql_value(val))
                    for col, val in clean_record.items()
                    if col in existing_cols
                ]
                if not valid_items:
                    return

                cols_sql = ", ".join(f'"{col}"' for col, _ in valid_items)
                placeholders = ", ".join(["%s"] * len(valid_items))
                params = tuple(val for _, val in valid_items)

                pg_conn.execute(
                    f"""
                    INSERT INTO "{table_name}" ({cols_sql})
                    VALUES ({placeholders})
                    ON CONFLICT ("cell_id") DO NOTHING
                    """,
                    params,
                )
                pg_conn.commit()
                return
        except Exception as exc:
            print(f"Ajout Supabase indisponible ({exc}), repli sur SQLite local.")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        existing_cols = {
            row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        valid_items = [
            (col, _normalize_sql_value(val))
            for col, val in clean_record.items()
            if col in existing_cols
        ]
        if not valid_items:
            return

        cols_sql = ", ".join(f'"{col}"' for col, _ in valid_items)
        placeholders = ", ".join(["?"] * len(valid_items))
        params = tuple(val for _, val in valid_items)

        cursor.execute(
            f"""
            INSERT OR IGNORE INTO "{table_name}" ({cols_sql})
            VALUES ({placeholders})
            """,
            params,
        )
        conn.commit()
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