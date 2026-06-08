"""Build and load the two-sided pair feature matrix.

Each row is a directed pair (A→B) with A's and B's preprocessed survey features
side by side. See ``docs/build_two_sided_features.md`` for the full rationale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_one_sided_data import (
    ENCODINGS,
    FEATURE_COLS,
    OUTPUT_PATH as ONE_SIDED_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "two_sided_pair_features.csv"

INTERACTION_COLS = ["iid", "pid", "wave", "dec", "match"]


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            kwargs = {"encoding": encoding, "low_memory": False}
            if usecols is not None:
                kwargs["usecols"] = usecols
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not decode CSV with utf-8, latin-1, or cp1252")


def build_two_sided_dataset(
    one_sided_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build directed pair rows with ``_A`` and ``_B`` feature columns."""
    one_sided_path = one_sided_path or ONE_SIDED_PATH
    output_path = output_path or OUTPUT_PATH

    one_sided = _read_csv(one_sided_path)

    participants = one_sided.drop_duplicates("iid")[["iid", *FEATURE_COLS]]
    interactions = one_sided[INTERACTION_COLS].copy()

    feature_cols = [c for c in participants.columns if c != "iid"]

    df = interactions.merge(
        participants.rename(columns={c: f"{c}_A" for c in feature_cols}),
        on="iid",
        how="left",
    )

    df = df.merge(
        participants.rename(columns={c: f"{c}_B" for c in feature_cols}),
        left_on="pid",
        right_on="iid",
        how="left",
        suffixes=("", "_drop"),
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])

    assert len(df) == len(interactions), (
        f"Row count mismatch: {len(df)} vs {len(interactions)}"
    )

    a_cols = [c for c in df.columns if c.endswith("_A")]
    b_cols = [c for c in df.columns if c.endswith("_B")]

    missing_a = df[a_cols].isna().sum()
    if missing_a.any():
        raise ValueError(f"Unexpected missing A features:\n{missing_a[missing_a > 0]}")

    rows_before = len(df)
    incomplete_b = df[b_cols].isna().any(axis=1)
    if incomplete_b.any():
        dropped_pids = sorted(int(x) for x in df.loc[incomplete_b, "pid"].unique())
        print(
            f"Dropping {incomplete_b.sum()} rows with missing B features "
            f"(partner not in usable profiles: {dropped_pids})"
        )
        df = df.loc[~incomplete_b].reset_index(drop=True)

    assert df[a_cols + b_cols].isna().sum().sum() == 0, "Feature columns must be complete"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Shape: {df.shape} (dropped {rows_before - len(df)} incomplete partner rows)")
    print(f"A feature columns: {len(a_cols)}")
    print(f"B feature columns: {len(b_cols)}")
    print(f"Saved {len(df)} rows with {len(df.columns)} columns to {output_path}")

    return df


def load_two_sided_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the two-sided pair feature CSV."""
    path = path or OUTPUT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python build_two_sided_data.py` first."
        )

    df = _read_csv(path)

    if len(df) < 1000:
        raise ValueError(f"Expected thousands of pair rows, got {len(df):,}")

    expected_meta = set(INTERACTION_COLS)
    if not expected_meta.issubset(df.columns):
        missing = expected_meta - set(df.columns)
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    df["pid"] = df["pid"].astype(int)
    df["dec"] = df["dec"].fillna(0).astype(int)
    df["match"] = df["match"].fillna(0).astype(int)

    feature_cols = [c for c in df.columns if c.endswith("_A") or c.endswith("_B")]
    missing_features = df[feature_cols].isna().sum().sum()
    if missing_features:
        raise ValueError(
            f"Loaded dataset has {missing_features} missing feature values; rebuild with "
            "`python build_two_sided_data.py`"
        )

    return df


if __name__ == "__main__":
    build_two_sided_dataset()
