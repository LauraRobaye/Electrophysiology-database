# Electrophysiology Database

An interactive Streamlit dashboard designed to analyze and compare neuronal electrophysiological properties across brain regions (focusing on Dorsal Raphe Nucleus at first but expending to all brain regions), supporting data-driven neuroscientific research, promote *in silico* modeling and the 3Rs ethical framework (Reduction and Refinement).

---

## Key Features

* **Interactive PCA explorer:** Visualize multidimensional electrophysiological features in a reduced dimensional space to highlight electrophysiological phenotype clusters.
* **Open Science dual-Database architecture:** Implements a strict separation between public reference datasets (literature controls, Allen Brain Institute data) and private, non-published PhD thesis recordings.
* **Robust ingestion pipeline:** Automated script (`append_data.py`) with strict validation to ingest batch CSV files safely into local SQLite databases.
* **Dynamic data merging:** Automatically merges public and local private databases on-the-fly for local analysis while keeping sensitive research data strictly offline during cloud deployment.
* **In silico modeling** Uses data from the litterature to predict the behaviour of *in silico* neurones.

---

## Tech Stack

* **Language:** Python 3.9+
* **App & UI:** Streamlit, Plotly
* **Data Management:** Pandas, SQLite, Python native `csv` library
* **Version Control:** Git & GitHub

---

## Project Structure

```text
drn-electrophys-comparator/
├── data/
│   ├── public_electrophys.db      # Public reference data (tracked on GitHub)
│   ├── private_electrophys.db     # Local thesis recordings (ignored by Git)
│   └── incoming_batch.csv         # Buffer for new data batch ingestion
├── src/
│   ├── append_data.py             # Robust batch ingestion script
│   └── split_database.py          # Migration script for public/private splitting
├── app.py                         # Main Streamlit application
├── requirements.txt               # Project dependencies
├── .gitignore                     # Security rules for private assets
└── README.md