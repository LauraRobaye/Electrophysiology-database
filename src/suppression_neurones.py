# Dans le terminal
# d'abord
python3

# puis
import sqlite3
conn = sqlite3.connect("data/electrophys_database.db")
cursor = conn.cursor()
cursor.execute("DELETE FROM neurons_electrophys WHERE cell_id = 'Literature'")
conn.commit()
conn.close()
print("Nettoyage effectué avec succès !")

# pour finir
exit()