"""Build and load the one-sided participant feature dataset.

Each row is one speed-date interaction. Preprocessed survey features describe
participant ``iid`` only. Identification and target columns are included for
modeling and evaluation.

See ``docs/preprocessing_and_feature_selection.md`` for the full rationale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Speed Dating Data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "one_sided_participant_features.csv"

# --- column sets ---
ID_COLS = ["iid", "id", "wave", "pid"]
TARGET_COLS = ["dec", "match"]

INTEREST_COLS = [
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
    "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
    "music", "shopping", "yoga",
]

KEEP_CORE = [
    "attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1",
    "attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1",
]

KEEP_ADDITIONAL = [
    "age", "race", "field_cd", "career_c", "goal", "date", "go_out",
    "imprace", "imprelig", "exphappy",
    "attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1", "shar1_1",
    *INTEREST_COLS,
]

FEATURE_COLS = sorted([*KEEP_CORE, *KEEP_ADDITIONAL, "gender"])
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in {"gender", "race", "field_cd", "career_c"}]
CATEGORICAL_COLS = ["race", "field_cd", "career_c"]

EXCLUDE_MISSING_THRESHOLD = 0.30
OUTPUT_COLS = ID_COLS + FEATURE_COLS + TARGET_COLS

ENCODINGS = ("utf-8", "latin-1", "cp1252")


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, usecols=usecols, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not decode CSV with utf-8, latin-1, or cp1252")


def _first_mode(series: pd.Series):
    modes = series.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else np.nan


def impute_model_profiles(
    profiles: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    exclude_threshold: float = EXCLUDE_MISSING_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    """Impute KEEP features and return profiles plus an exclusion mask."""
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS

    out = profiles[FEATURE_COLS].copy()
    missing_share = out.isna().mean(axis=1)
    exclude_mask = missing_share > exclude_threshold

    impute_base = out.loc[~exclude_mask]
    global_num = impute_base[numeric_cols].median()
    global_cat = impute_base[categorical_cols].agg(_first_mode)

    for col in numeric_cols:
        fill = impute_base.groupby("gender")[col].transform("median")
        fill = fill.fillna(global_num[col])
        out.loc[~exclude_mask, col] = out.loc[~exclude_mask, col].fillna(fill)

    for col in categorical_cols:
        gender_mode = impute_base.groupby("gender")[col].agg(_first_mode)
        for gender, value in gender_mode.items():
            mask = (~exclude_mask) & (out["gender"] == gender) & out[col].isna()
            out.loc[mask, col] = value
        out.loc[~exclude_mask, col] = out.loc[~exclude_mask, col].fillna(global_cat[col])

    remaining = out.loc[~exclude_mask, FEATURE_COLS].isna().sum().sum()
    if remaining:
        raise ValueError(f"Imputation incomplete: {remaining} missing values remain")

    return out, exclude_mask


def build_one_sided_dataset(
    raw_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build one-sided interaction rows with preprocessed ``iid`` features."""
    raw_path = raw_path or RAW_DATA_PATH
    output_path = output_path or OUTPUT_PATH

    load_cols = list(dict.fromkeys(ID_COLS + FEATURE_COLS + TARGET_COLS))
    raw = _read_csv(raw_path, load_cols)
    raw = raw.dropna(subset=["pid"]).copy()
    raw["pid"] = raw["pid"].astype(int)

    profiles_raw = raw.drop_duplicates("iid").set_index("iid")[FEATURE_COLS]
    profiles_imputed, excluded = impute_model_profiles(profiles_raw)
    usable_iids = profiles_imputed.index[~excluded]

    interactions = raw[raw["iid"].isin(usable_iids)].copy()
    feature_frame = profiles_imputed.loc[interactions["iid"].values].set_index(interactions.index)
    interactions[FEATURE_COLS] = feature_frame.values

    # Raw file has occasional missing/inconsistent ``id``; ``iid`` is canonical.
    interactions["id"] = interactions["iid"]
    interactions["dec"] = interactions["dec"].fillna(0).astype(int)
    interactions["match"] = interactions["match"].fillna(0).astype(int)

    dataset = interactions[OUTPUT_COLS].sort_values(ID_COLS).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def load_one_sided_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the preprocessed one-sided participant feature CSV."""
    path = path or OUTPUT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python load_one_sided_data.py` first."
        )

    df = _read_csv(path, OUTPUT_COLS)

    if len(df) < 1000:
        raise ValueError(f"Expected thousands of interaction rows, got {len(df):,}")

    df["id"] = df["id"].fillna(df["iid"]).astype(int)
    df["pid"] = df["pid"].astype(int)
    df["dec"] = df["dec"].fillna(0).astype(int)
    df["match"] = df["match"].fillna(0).astype(int)

    missing_features = df[FEATURE_COLS].isna().sum().sum()
    if missing_features:
        raise ValueError(f"Loaded dataset has {missing_features} missing feature values")

    return df


if __name__ == "__main__":
    df = build_one_sided_dataset()
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    print(f"Participants: {df['iid'].nunique():,} | Columns: {len(df.columns)}")
    print(f"Mutual match rate: {df['match'].mean():.1%}")
