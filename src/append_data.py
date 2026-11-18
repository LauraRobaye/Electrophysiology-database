import sqlite3
import csv

DB_PATH = "data/electrophys_database.db"
CSV_PATH = "data/incoming_batch.csv"

def append_batch():
    print(f"Chargement des nouveaux enregistrements depuis '{CSV_PATH}'...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    added_count = 0
    skipped_count = 0
    
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Saute automatiquement l'en-tête
        print(f"En-tête détecté et ignoré : {header}")
        
        for idx, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                continue
                
            cell_id = row[0].strip()
            print(f"DEBUG [Ligne {idx}] -> cell_id lu : '{cell_id}'")
            
            # Vérifie si le cell_id existe déjà dans la base
            cursor.execute("SELECT COUNT(*) FROM neurons_electrophys WHERE cell_id = ?", (cell_id,))
            exists = cursor.fetchone()[0]
            
            if exists > 0:
                print(f" -> [DOUBLON] Déjà dans la base : {cell_id}")
                skipped_count += 1
                continue
                
            # Normalise la ligne pour qu'elle ait exactement 20 colonnes (complète par None si besoin)
            values = []
            for i in range(20):
                if i < len(row) and row[i].strip() != '':
                    values.append(row[i].strip())
                else:
                    values.append(None)
            
            cursor.execute("""
            INSERT OR IGNORE INTO neurons_electrophys VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, tuple(values))
            added_count += 1
            print(f" -> [SUCCÈS] Ajouté : {cell_id}")

    conn.commit()
    conn.close()
    
    print("\nMise à jour terminée avec succès !")
    print(f" - Ajoutés : {added_count} neurones")
    print(f" - Ignorés : {skipped_count} neurones")

if __name__ == "__main__":
    append_batch()

    # python3 src/append_data.py