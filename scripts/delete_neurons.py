from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import tomllib
import psycopg


DEFAULT_DB_PATH = "data/electrophys_database.db"
DEFAULT_TABLE = "neurons_electrophys"


def _resolve_supabase_url(explicit_url: str | None = None) -> str | None:
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()

    import os

    env_url = os.environ.get("SUPABASE_DB_URL", "").strip()
    if env_url:
        return env_url

    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            secret_url = str(data.get("SUPABASE_DB_URL", "")).strip()
            if secret_url:
                return secret_url
        except Exception:
            pass

    return None


def _preview_sqlite(db_path: str, table_name: str, cell_ids: list[str]) -> list[tuple]:
    placeholders = ", ".join(["?"] * len(cell_ids))
    query = f"""
        SELECT cell_id, source, brain_region, preparation, record_type
        FROM {table_name}
        WHERE cell_id IN ({placeholders})
        ORDER BY cell_id
    """
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(query, tuple(cell_ids)).fetchall()
    finally:
        conn.close()


def _delete_sqlite(db_path: str, table_name: str, cell_ids: list[str]) -> int:
    placeholders = ", ".join(["?"] * len(cell_ids))
    query = f"DELETE FROM {table_name} WHERE cell_id IN ({placeholders})"
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(query, tuple(cell_ids))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def _preview_pg(db_url: str, table_name: str, cell_ids: list[str]) -> list[tuple]:
    query = f"""
        SELECT cell_id, source, brain_region, preparation, record_type
        FROM {table_name}
        WHERE cell_id = ANY(%s)
        ORDER BY cell_id
    """
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        return conn.execute(query, (cell_ids,)).fetchall()


def _delete_pg(db_url: str, table_name: str, cell_ids: list[str]) -> int:
    query = f"DELETE FROM {table_name} WHERE cell_id = ANY(%s)"
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        cur = conn.execute(query, (cell_ids,))
        conn.commit()
        return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supprimer de manière sécurisée une liste de cell_id dans neurons_electrophys."
    )
    parser.add_argument(
        "cell_ids",
        nargs="*",
        help="Identifiants cell_id à supprimer (ex: Test_01 Test_02).",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Chemin SQLite local si pas de Supabase (défaut : {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--supabase-url",
        default=None,
        help="URL PostgreSQL Supabase (optionnel, sinon lu depuis l'env ou .streamlit/secrets.toml).",
    )
    parser.add_argument(
        "--sqlite-only",
        action="store_true",
        help="Force l'utilisation de la base SQLite locale même si SUPABASE_DB_URL est défini.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche uniquement les lignes correspondantes sans rien supprimer.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Supprime sans demander de confirmation interactive.",
    )

    args = parser.parse_args()

    # Liste des cell_id : arguments CLI ou saisie interactive
    if args.cell_ids:
        raw_ids = args.cell_ids
    else:
        entered = input("Entrez les cell_id à supprimer (séparés par des virgules ou espaces) : ").strip()
        raw_ids = [item.strip() for item in entered.replace(",", " ").split() if item.strip()]

    cell_ids = list(dict.fromkeys(raw_ids))
    if not cell_ids:
        print("Aucun cell_id fourni. Abandon.")
        return

    use_pg = False if args.sqlite_only else (_resolve_supabase_url(args.supabase_url) is not None)
    db_url = _resolve_supabase_url(args.supabase_url) if use_pg else None

    if use_pg and db_url:
        backend_label = "Supabase (PostgreSQL)"
        rows = _preview_pg(db_url, DEFAULT_TABLE, cell_ids)
    else:
        backend_label = f"SQLite ({args.db_path})"
        rows = _preview_sqlite(args.db_path, DEFAULT_TABLE, cell_ids)

    print(f"\nCible : {backend_label}")
    print(f"Enregistrements trouvés ({len(rows)} / {len(cell_ids)} demandés) :")
    if not rows:
        print("  -> Aucune ligne correspondante trouvée.")
        return

    for r in rows:
        print(f"  - cell_id={r[0]} | source={r[1]} | region={r[2]} | prep={r[3]} | type={r[4]}")

    found_ids = {r[0] for r in rows}
    missing_ids = [cid for cid in cell_ids if cid not in found_ids]
    if missing_ids:
        print(f"  (Non trouvés en base : {', '.join(missing_ids)})")

    if args.dry_run:
        print("\nMode --dry-run actif : aucune ligne supprimée.")
        return

    if not args.yes:
        confirm = input(
            f"\nConfirmer la suppression définitive de ces {len(rows)} ligne(s) dans {backend_label} ? [oui/N] : "
        ).strip().lower()
        if confirm not in {"oui", "o", "yes", "y"}:
            print("Suppression annulée.")
            return

    if use_pg and db_url:
        deleted = _delete_pg(db_url, DEFAULT_TABLE, cell_ids)
    else:
        deleted = _delete_sqlite(args.db_path, DEFAULT_TABLE, cell_ids)

    print(f"\nTerminé : {deleted} ligne(s) supprimée(s).")


if __name__ == "__main__":
    main()

# Utilisation du script
# 1. Aperçu sans suppression (--dry-run)
#python scripts/delete_neurons.py Exemple_1 Exemple_2 --dry-run

# 2. Suppression interactive avec confirmation [oui/N]
#python scripts/delete_neurons.py Exemple_1 Exemple_2

# 3. Cibler uniquement la base SQLite locale
#python scripts/delete_neurons.py Exemple_1 --sqlite-only