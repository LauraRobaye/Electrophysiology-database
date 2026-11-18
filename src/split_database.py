import sqlite3
import pandas as pd
import os

OLD_DB = "data/electrophys_database.db"
PUBLIC_DB = "data/public_electrophys.db"
PRIVATE_DB = "data/private_electrophys.db"

def split_database():
    if not os.path.exists(OLD_DB):
        print(f"Erreur : La base '{OLD_DB}' est introuvable.")
        return

    # 1. Charger toutes les données actuelles
    conn = sqlite3.connect(OLD_DB)
    df = pd.read_sql("SELECT * FROM neurons_electrophys", conn)
    conn.close()

    print(f"Total des neurones dans la base originale : {len(df)}")
    print("Sources présentes dans la base :", df['source'].unique())

    # 2. Séparation : les références de littérature vont dans le public, le reste en privé
    is_public = df['source'].str.lower().isin(['literature', 'allen', 'reference'])

    df_public = df[is_public]
    df_private = df[~is_public]

    print(f" -> Neurones publics (vers {PUBLIC_DB}) : {len(df_public)}")
    print(f" -> Neurones privés (vers {PRIVATE_DB}) : {len(df_private)}")

    # 3. Sauvegarder dans la base publique
    conn_pub = sqlite3.connect(PUBLIC_DB)
    df_public.to_sql("neurons_electrophys", conn_pub, if_exists="replace", index=False)
    conn_pub.close()

    # 4. Sauvegarder dans la base privée
    conn_priv = sqlite3.connect(PRIVATE_DB)
    df_private.to_sql("neurons_electrophys", conn_priv, if_exists="replace", index=False)
    conn_priv.close()

    print("\nSéparation réussie ! Tes deux bases sont prêtes.")

if __name__ == "__main__":
    split_database()