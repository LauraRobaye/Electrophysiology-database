import sqlite3

def inspect_db_schema(db_path="data/electrophys_database.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Lister toutes les tables de la base
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"=== TABLES DANS LA BASE ({db_path}) ===")
    for table in tables:
        table_name = table[0]
        print(f"\n[Table] : {table_name}")
        
        # Récupérer les détails de chaque colonne pour cette table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"{'ID':<4} | {'Nom de la colonne':<25} | {'Type':<15}")
        print("-" * 50)
        for col in columns:
            col_id, name, data_type, not_null, default_val, pk = col
            print(f"{col_id:<4} | {name:<25} | {data_type:<15}")
            
    conn.close()

if __name__ == "__main__":
    inspect_db_schema()