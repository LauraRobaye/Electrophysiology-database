from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: list[str] = [
    "rmp_mv",
    "tau_ms",
    "cm_pf",
    "rheobase_pa",
    "ap_threshold_mv",
    "ap_amplitude_mv",
    "ap_half_width_ms",
    "ahp_amplitude_mv",
    "ahp_duration_ms",
]


@dataclass
class EvaluationResult:
    n_samples: int
    n_classes: int
    balanced_accuracy: float
    f1_macro: float
    report: str


def build_classifier(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def prepare_single_cell_data(
    df: pd.DataFrame,
    target_col: str = "known_identity",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    work = df.copy()

    if "record_type" in work.columns:
        work = work[work["record_type"] == "single_cell"]

    work = work[work[target_col].notna() & (work[target_col] != "Unknown")]
    X = work[cols].apply(pd.to_numeric, errors="coerce")
    y = work[target_col].astype(str)
    return X, y


def evaluate_cv(
    df: pd.DataFrame,
    target_col: str = "known_identity",
    n_splits: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    X, y = prepare_single_cell_data(df, target_col=target_col)
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0

    if len(y) < 4 or len(class_counts) < 2 or min_class_count < 2:
        raise ValueError(
            "Données insuffisantes pour une validation croisée stratifiée "
            "(il faut au moins 2 classes avec au moins 2 exemples chacune)."
        )

    splits = min(n_splits, min_class_count)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    pipe = build_classifier(random_state=random_state)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    return EvaluationResult(
        n_samples=len(y),
        n_classes=len(class_counts),
        balanced_accuracy=float(balanced_accuracy_score(y, y_pred)),
        f1_macro=float(f1_score(y, y_pred, average="macro", zero_division=0)),
        report=str(classification_report(y, y_pred, zero_division=0)),
    )


def compute_literature_zscores(
    cell_row: pd.Series,
    literature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule le z-score d'une cellule par rapport aux profils group_summary ayant un SD."""
    cols = feature_cols or [c for c in FEATURE_COLS if c in literature_df.columns]
    lit = literature_df.copy()
    if "record_type" in lit.columns:
        lit = lit[lit["record_type"] == "group_summary"]

    rows_out: list[dict[str, object]] = []
    for _, prof in lit.iterrows():
        entry: dict[str, object] = {
            "cell_id": prof.get("cell_id"),
            "known_identity": prof.get("known_identity"),
            "preparation": prof.get("preparation"),
            "n_cells": prof.get("n_cells"),
            "reference_doi": prof.get("reference_doi"),
        }
        for col in cols:
            sd_col = f"{col}_sd"
            x_val = pd.to_numeric(cell_row.get(col), errors="coerce")
            mu_val = pd.to_numeric(prof.get(col), errors="coerce")
            sd_val = pd.to_numeric(prof.get(sd_col), errors="coerce") if sd_col in lit.columns else np.nan
            if pd.notna(x_val) and pd.notna(mu_val) and pd.notna(sd_val) and float(sd_val) > 0:
                entry[f"z_{col}"] = round((float(x_val) - float(mu_val)) / float(sd_val), 2)
            else:
                entry[f"z_{col}"] = np.nan
        rows_out.append(entry)

    return pd.DataFrame(rows_out)

if __name__ == "__main__":
    import sqlite3
    from pathlib import Path

    db_file = Path(__file__).resolve().parent.parent / "data" / "public_electrophys.db"
    conn = sqlite3.connect(db_file)
    df_test = pd.read_sql_query("SELECT * FROM neurons_electrophys", conn)
    conn.close()

    X, y = prepare_single_cell_data(df_test)
    print("Effectifs single_cell par classe :")
    print(y.value_counts())

    if len(y) >= 4 and y.nunique() >= 2 and y.value_counts().min() >= 2:
        res = evaluate_cv(df_test)
        print(f"\nBalanced accuracy : {res.balanced_accuracy:.3f} | F1 macro : {res.f1_macro:.3f}")
        print(res.report)
    else:
        print("\nPas encore assez de cellules unitaires par classe pour lancer la validation croisée.")