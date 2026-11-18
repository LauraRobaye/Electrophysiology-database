import sqlite3

databases = ["data/public_electrophys.db", "data/private_electrophys.db"]

for db_path in databases:
    print(f"Mise à jour de : {db_path} ...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE neurons_electrophys ADD COLUMN doi TEXT;")
        print(" -> Colonne 'doi' ajoutée avec succès !")
    except sqlite3.OperationalError:
        print(" -> (La colonne 'doi' existe déjà)")
    conn.commit()
    conn.close()

print("\nMise à jour du schéma (DOI) terminée !")