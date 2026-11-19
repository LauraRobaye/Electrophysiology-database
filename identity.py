from __future__ import annotations

import pandas as pd

# Dictionnaire de normalisation : variantes (en minuscules/nettoyées) -> étiquette canonique
IDENTITY_MAPPING = {
    # Sérotonergique
    "5-ht": "Serotonergic",
    "5ht": "Serotonergic",
    "serotonin": "Serotonergic",
    "serotonergic": "Serotonergic",
    "sérotonergique": "Serotonergic",
    "drn 5-ht": "Serotonergic",
    # GABAergique
    "gaba": "GABAergic",
    "gabaergic": "GABAergic",
    "gabaergique": "GABAergic",
    "interneuron": "GABAergic",
    "interneurone": "GABAergic",
    # Dopaminergique
    "da": "Dopaminergic",
    "dopamine": "Dopaminergic",
    "dopaminergic": "Dopaminergic",
    "dopaminergique": "Dopaminergic",
    # Glutamatergique
    "glutamate": "Glutamatergic",
    "glutamatergic": "Glutamatergic",
    "glutamatergique": "Glutamatergic",
    # Cholinergique
    "ach": "Cholinergic",
    "cholinergic": "Cholinergic",
    "cholinergique": "Cholinergic",
    # Noradrénergique
    "ne": "Noradrenergic",
    "na": "Noradrenergic",
    "noradrenergic": "Noradrenergic",
    "noradrénergique": "Noradrenergic",
}


def get_canonical_identities() -> list[str]:
    """Retourne la liste triée et sans doublon des identités canoniques."""
    return sorted(set(IDENTITY_MAPPING.values()))


def harmonize_identity_value(val) -> str | None:
    """Normalise une valeur unique de known_identity."""
    if val is None or pd.isna(val):
        return None
    raw = str(val).strip()
    if raw == "" or raw.lower() in {"unknown", "inconnu", "na", "nan", "none", "-"}:
        return None
    key = " ".join(raw.lower().replace("_", " ").split())
    if key in IDENTITY_MAPPING:
        return IDENTITY_MAPPING[key]
    return raw.capitalize()


def harmonize_identity_series(series: pd.Series) -> pd.Series:
    """Normalise une série pandas de known_identity."""
    return series.apply(harmonize_identity_value)