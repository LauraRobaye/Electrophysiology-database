import sqlite3

def export_db_schema(db_path="data/electrophys_database.db", output_path="data/schema.sql"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Récupérer toutes les définitions SQL de la base (tables, index, etc.)
    cursor.execute("SELECT sql FROM sqlite_master WHERE sql NOT NULL;")
    schema_elements = cursor.fetchall()
    
    conn.close()
    
    # Écriture dans le fichier de sortie
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("-- Schéma de la base de données SQLite : electrophys_database.db\n")
        f.write("-- Généré automatiquement pour le projet de thèse\n\n")
        for element in schema_elements:
            if element[0]:
                f.write(element[0] + ";\n\n")
                
    print(f"Schéma exporté avec succès dans '{output_path}' !")

if __name__ == "__main__":
    export_db_schema()