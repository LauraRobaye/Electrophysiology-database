from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import psycopg
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from identity import harmonize_identity_value


def migrate_public_db(sqlite_path: str = "data/public_electrophys.db") -> None:
    db_url = st.secrets.get("SUPABASE_DB_URL", "").strip()
    if not db_url:
        raise ValueError("SUPABASE_DB_URL est manquant dans .streamlit/secrets.toml")

    src_conn = sqlite3.connect(ROOT_DIR / sqlite_path)
    src_conn.row_factory = sqlite3.Row
    try:
        rows = src_conn.execute("SELECT * FROM neurons_electrophys").fetchall()
        if not rows:
            print("Aucune ligne à migrer.")
            return
        columns = list(rows[0].keys())
    finally:
        src_conn.close()

    id_idx = columns.index("known_identity") if "known_identity" in columns else None

    prepared_rows = []
    for r in rows:
        vals = [r[col] for col in columns]
        if id_idx is not None and vals[id_idx] is not None:
            harm = harmonize_identity_value(vals[id_idx])
            vals[id_idx] = harm if harm else "Unknown"
        prepared_rows.append(tuple(vals))

    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_assignments = ", ".join(
        [f"{col} = EXCLUDED.{col}" for col in columns if col != "cell_id"]
    )

    sql = f"""
        INSERT INTO neurons_electrophys ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (cell_id) DO UPDATE SET
            {update_assignments}
    """

    with psycopg.connect(db_url) as pg_conn:
        with pg_conn.cursor() as cur:
            cur.executemany(sql, prepared_rows)
        pg_conn.commit()

    print(f"Migration terminée : {len(prepared_rows)} neurone(s) synchronisé(s) vers Supabase.")


if __name__ == "__main__":
    migrate_public_db()