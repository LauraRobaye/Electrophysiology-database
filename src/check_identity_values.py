import sqlite3
import pandas as pd

conn = sqlite3.connect("data/electrophys_database.db")
df = pd.read_sql_query("SELECT DISTINCT known_identity FROM neurons_electrophys", conn)
conn.close()

print("Valeurs exactes trouvées dans la colonne 'known_identity' :")
print(df)