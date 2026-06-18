"""Feature engineering for S9 notebooks (ported from feature_engineering_experiment.ipynb).

Builds the E4_Full column set from keep-w12 parquet:
  - 89 raw _A/_B columns (gender_B dropped as perfect complement of gender_A)
  - 45 engineered pair-level columns (pref match, interest diffs, diff_age, etc.)

Pair-level transforms use only _A/_B columns (no label leakage). Missing values in
raw columns are kept until per-split mean imputation in s9_foundation.process_split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm

from load_one_sided_data import FEATURE_COLS

ID_COLS = ["iid", "pid", "wave", "dec", "dec_o", "match"]

DIMS = ["attr", "sinc", "intel", "fun", "amb"]
PREF_COLS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1"]
INTEREST_COLS = [
    "sports", "tvsports", "exercise", "dining", "museums", "art",
    "hiking", "gaming", "clubbing", "reading", "tv", "theater",
    "movies", "concerts", "music", "shopping", "yoga",
]

ENGINEERED_COLS = {
    "group1": ["pref_match_A_to_B", "pref_match_B_to_A", "pref_match_mutual"],
    "group2": [f"diff_{c}" for c in INTEREST_COLS] + ["interest_cosine"],
    "group3": [
        "diff_age", "diff_imprace", "diff_imprelig", "diff_go_out", "diff_date",
        "same_race", "same_field", "same_career", "same_goal",
    ],
    "group4": [
        "pref_entropy_A", "pref_entropy_B", "diff_pref_entropy",
        "aspiration_gap_attr_A", "aspiration_gap_sinc_A", "aspiration_gap_intel_A",
        "aspiration_gap_fun_A", "aspiration_gap_amb_A",
        "aspiration_gap_attr_B", "aspiration_gap_sinc_B", "aspiration_gap_intel_B",
        "aspiration_gap_fun_B", "aspiration_gap_amb_B",
        "exphappy_diff", "exphappy_sum",
    ],
}

E4_GROUPS = ["group1", "group2", "group3", "group4"]


def find_perfect_complement_b_features(data: pd.DataFrame, raw_cols: list[str]) -> list[str]:
    """Drop _B columns where feature_A + feature_B == 1 on every row (dummy trap)."""
    drop_b = []
    for b_col in raw_cols:
        if not b_col.endswith("_B"):
            continue
        a_col = b_col[:-2] + "_A"
        if a_col not in data.columns:
            continue
        a = data[a_col]
        b = data[b_col]
        if a.notna().all() and b.notna().all() and np.allclose(a + b, 1.0, atol=1e-6):
            drop_b.append(b_col)
    return drop_b


def build_two_sided_raw(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Join partner profiles; keep all interaction rows (NaNs imputed per split later)."""
    prof = clean_df.drop_duplicates("iid")[["iid"] + FEATURE_COLS]
    meta = ID_COLS + FEATURE_COLS
    out = clean_df[meta].copy()
    out = out.rename(columns={c: f"{c}_A" for c in FEATURE_COLS})
    partner = prof.rename(
        columns={"iid": "_pid_join", **{c: f"{c}_B" for c in FEATURE_COLS}}
    )
    out = out.merge(partner, left_on="pid", right_on="_pid_join", how="left")
    return out.drop(columns=["_pid_join"])


def add_group1_features(data: pd.DataFrame) -> None:
    data["pref_match_A_to_B"] = sum(
        data[f"{d}1_1_A"] * data[f"{d}3_1_B"] for d in DIMS
    )
    data["pref_match_B_to_A"] = sum(
        data[f"{d}1_1_B"] * data[f"{d}3_1_A"] for d in DIMS
    )
    data["pref_match_mutual"] = np.sqrt(
        data["pref_match_A_to_B"] * data["pref_match_B_to_A"]
    )


def add_group2_features(data: pd.DataFrame) -> None:
    for col in INTEREST_COLS:
        data[f"diff_{col}"] = np.abs(data[f"{col}_A"] - data[f"{col}_B"])

    a_mat = data[[f"{c}_A" for c in INTEREST_COLS]].to_numpy(dtype=float)
    b_mat = data[[f"{c}_B" for c in INTEREST_COLS]].to_numpy(dtype=float)
    denom = norm(a_mat, axis=1) * norm(b_mat, axis=1)
    data["interest_cosine"] = np.where(denom > 0, np.sum(a_mat * b_mat, axis=1) / denom, 0.0)


def add_group3_features(data: pd.DataFrame) -> None:
    data["diff_age"] = np.abs(data["age_A"] - data["age_B"])
    data["diff_imprace"] = np.abs(data["imprace_A"] - data["imprace_B"])
    data["diff_imprelig"] = np.abs(data["imprelig_A"] - data["imprelig_B"])
    data["diff_go_out"] = np.abs(data["go_out_A"] - data["go_out_B"])
    data["diff_date"] = np.abs(data["date_A"] - data["date_B"])
    data["same_race"] = (data["race_A"] == data["race_B"]).astype(int)
    data["same_field"] = (data["field_cd_A"] == data["field_cd_B"]).astype(int)
    data["same_career"] = (data["career_c_A"] == data["career_c_B"]).astype(int)
    data["same_goal"] = (data["goal_A"] == data["goal_B"]).astype(int)


def _pref_entropy(row: pd.Series, suffix: str) -> float:
    vals = np.array([row[f"{c}_{suffix}"] for c in PREF_COLS], dtype=float)
    vals = vals / (vals.sum() + 1e-9)
    return float(-np.sum(vals * np.log(vals + 1e-9)))


def add_group4_features(data: pd.DataFrame) -> None:
    data["pref_entropy_A"] = data.apply(lambda r: _pref_entropy(r, "A"), axis=1)
    data["pref_entropy_B"] = data.apply(lambda r: _pref_entropy(r, "B"), axis=1)
    data["diff_pref_entropy"] = np.abs(data["pref_entropy_A"] - data["pref_entropy_B"])

    for d in DIMS:
        data[f"aspiration_gap_{d}_A"] = data[f"{d}2_1_A"] - data[f"{d}3_1_A"]
        data[f"aspiration_gap_{d}_B"] = data[f"{d}2_1_B"] - data[f"{d}3_1_B"]

    data["exphappy_diff"] = data["exphappy_A"] - data["exphappy_B"]
    data["exphappy_sum"] = data["exphappy_A"] + data["exphappy_B"]


def apply_feature_groups(data: pd.DataFrame, groups: list[str]) -> list[str]:
    if "group1" in groups:
        add_group1_features(data)
    if "group2" in groups:
        add_group2_features(data)
    if "group3" in groups:
        add_group3_features(data)
    if "group4" in groups:
        add_group4_features(data)

    engineered = [c for g in groups for c in ENGINEERED_COLS[g]]
    for col in engineered:
        if data[col].isna().any():
            data[col] = data[col].fillna(0)
    return engineered


def build_feature_list(raw_cols: list[str], groups: list[str]) -> list[str]:
    extra = [c for g in groups for c in ENGINEERED_COLS[g]]
    return raw_cols + extra


def theme_for_feature(name: str, raw_cols: list[str]) -> str:
    if name not in raw_cols:
        return "Engineered"
    return "Profile A" if name.endswith("_A") else "Profile B"


def normalize_importance(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if total <= 0:
        return series
    return series / total


def cumulative_coverage(sorted_scores: pd.Series, n_features: int) -> float:
    if n_features <= 0 or len(sorted_scores) == 0:
        return 0.0
    n_features = min(n_features, len(sorted_scores))
    return float(sorted_scores.iloc[:n_features].sum() / sorted_scores.sum())


def select_features_by_coverage(
    sorted_scores: pd.Series,
    target: float,
    min_features: int = 1,
    max_features: int | None = None,
) -> list[str]:
    if len(sorted_scores) == 0:
        return []

    cum = sorted_scores.cumsum() / sorted_scores.sum()
    if (cum >= target).any():
        n_target = int((cum >= target).argmax()) + 1
    else:
        n_target = len(sorted_scores)

    n_selected = max(min_features, n_target)
    if max_features is not None:
        n_selected = min(max_features, n_selected)

    return sorted_scores.head(n_selected).index.tolist()


def select_reduced_features_lr(
    coef_folds: list[pd.Series],
    coverage_target: float = 0.95,
) -> tuple[list[str], pd.Series, float]:
    """Average |LR coef| across folds, normalize, keep smallest prefix >= coverage_target."""
    avg_lr = pd.concat(coef_folds, axis=1).mean(axis=1).sort_values(ascending=False)
    norm_lr = normalize_importance(avg_lr)
    reduced = select_features_by_coverage(norm_lr, coverage_target)
    coverage = cumulative_coverage(norm_lr, len(reduced))
    return reduced, norm_lr, coverage


def build_s9_dataset(keep_parquet: Path, groups: list[str] | None = None) -> dict:
    """Load keep-w12, engineer features, return artifacts for S9 notebooks."""
    groups = groups or E4_GROUPS
    clean_df = pd.read_parquet(keep_parquet)
    data = build_two_sided_raw(clean_df)

    raw_all = sorted(c for c in data.columns if c.endswith("_A") or c.endswith("_B"))
    excluded = find_perfect_complement_b_features(data, raw_all)
    raw_feature_cols = [c for c in raw_all if c not in excluded]

    apply_feature_groups(data, groups)
    feature_cols = build_feature_list(raw_feature_cols, groups)
    excluded_set = set(excluded)
    assert not excluded_set & set(feature_cols), "Redundant features in model matrix"

    theme_map = {f: theme_for_feature(f, raw_feature_cols) for f in feature_cols}
    themes = sorted(set(theme_map.values()))

    return {
        "DATA": data,
        "RAW_FEATURE_COLS": raw_feature_cols,
        "EXCLUDED_FEATURE_COLS": excluded,
        "ENGINEERED_FEATURE_COLS": [c for g in groups for c in ENGINEERED_COLS[g]],
        "S1_FEATS": feature_cols,
        "CB_FEATS": feature_cols,
        "THEME_MAP": theme_map,
        "THEMES": themes,
        "FEATURE_GROUPS": groups,
    }
