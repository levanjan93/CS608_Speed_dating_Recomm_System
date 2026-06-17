# Stage 0 — Foundation cells (run before any stage document)

These seven cells are the shared substrate every stage document builds on:
config, provenance, data (+ wave-12 reconstruction), feature groups, the 10
splitting schemes, the leakage-safe preprocessor, and the reference metrics.

**They are copied verbatim from `PIPELINE_NOTEBOOK_CODE.md` Cells 1–7** so
there is exactly one behavior, documented twice. If you ever edit a cell
here, edit it there too (or delete it there) — drift between the two copies
is precisely the discrepancy class our record-keeping exists to catch.

Assembly order for the notebook:

```
pipeline_foundation.ipynb  (Cells 0-1 … 0-7 — run all cells first)
content_based_model.ipynb  (STAGE1_CONTENT_BASED.md Cells S1-0 … S1-5)
... later stage documents ...
```

The runnable notebook is `pipeline_foundation.ipynb` (generated from this doc / `pipeline_foundation.py`).
Regenerate with: `python _build_foundation_nb.py`

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib` (figures).
Stage 1 additionally needs `pip install "interpret>=0.4"`.

---

## Cell 0-1 — Imports, paths, frozen config

```python
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT          = Path(r"C:\SMU Subjects\Third Term\CS608-Recommender Systems\SpeedDatingRecommender")
RAW_CSV       = ROOT / "Speed Dating Data.csv"
CLEAN_PARQUET = ROOT / "speed_dating_clean.parquet"
TEAM_DIR      = ROOT / "Teammates' code" / "CS608_Speed_dating_Recomm_System-main"
RESULTS       = ROOT / "results"
for sub in ["runs", "explanations", "stability", "figures"]:
    (RESULTS / sub).mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- Frozen hyperparameters -------------------------------------------------
# Tuned ONCE on scheme S8, then FROZEN=True. The full-matrix runner refuses
# to run while FROZEN is False. Stage documents add their keys here
# (e.g. Stage 1 sets s1_model after its decision cell).
CONFIG = {
    # Stage 1 FROZEN after S8 tuning + 10-scheme sweep: LR beat EBM 10/10,
    # C insensitive -> C=0.1. Deployed/reported on scheme S9.
    "s1_model": "lr",     # frozen Stage-1 winner (vs "ebm")
    "s1_C": 0.1,          # frozen on S8 CV
    "s2_k": 25,           # Stage 2 lookalikes (chosen for locality; 25/50/100 within LOWO noise)
    "s2_tau": 0.2,        # Stage 2 partner-resemblance threshold
    "s2_m": 1.0,          # Stage 2 shrinkage pseudocount
    "fm_k": 8,            # FM latent dim (stages 3 and 4b)
    "fm_lr": 0.05,
    "fm_epochs": 400,
    "fm_l2": 1e-4,
    "fm_patience": 40,
    "seed": SEED,
    "deploy_scheme": "S9",   # LOWO over 20 non-19 waves + wave-19 holdout, wave 12 kept
    "metrics_version": "v1-experiments_splits-compatible",
    "FROZEN": True,       # Stage 1 frozen; flip stage-by-stage as later stages settle
}
print("Config loaded. FROZEN =", CONFIG["FROZEN"])
```

**Verify:** no import errors; `results/` subfolders exist.

---

## Cell 0-2 — Provenance utilities (record-keeping requirement)

```python
def file_sha(path: Path, n_hex: int = 12) -> str:
    """Short SHA-256 fingerprint of a file (identity of input data)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def df_fingerprint(df: pd.DataFrame) -> dict:
    """Cheap dataset identity: shape, wave list, label rates."""
    return {
        "n_rows": int(len(df)),
        "n_waves": int(df["wave"].nunique()),
        "waves": sorted(int(w) for w in df["wave"].unique()),
        "dec_rate": round(float(df["dec"].mean()), 4),
        "match_rate": round(float(df["match"].mean()), 4),
    }


def params_hash(obj) -> str:
    """Stable hash of any JSON-serializable parameter structure."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def write_run_record(scheme: str, split_tag: str, record: dict) -> Path:
    """One JSON per (scheme x split) + append a line to experiment_log.csv."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS / "runs" / f"{scheme}_{split_tag}_{ts}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    log_path = RESULTS / "experiment_log.csv"
    line = {
        "ts": ts, "scheme": scheme, "split": split_tag,
        "data_sha": record.get("data_sha", ""),
        "n_train": record.get("n_train", ""), "n_eval": record.get("n_eval", ""),
        **{k: round(v, 4) for k, v in record.get("metrics", {}).items()
           if isinstance(v, (int, float))},
    }
    pd.DataFrame([line]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False
    )
    return path

print("Provenance utils ready.")
```

**Verify:** call `params_hash(CONFIG)` twice → same 12-char string.

---

## Cell 0-3 — Load data, reconstruct wave 12 (gated), build the two-sided table

Key decisions (plan §3.1–3.2):
- keep-w12 variant is **reconstructed from raw** with the exact cleaning
  recipe, and only accepted if re-deriving wave 11 reproduces the clean
  parquet byte-for-byte on every model column;
- B-side pre-event profile joined on `pid` with `_B` suffix (teammates'
  convention);
- `int_cos` (interest cosine) computed statically with a ≥8-shared-dims rule
  (no fitted parameters → leakage-safe).

```python
INTEREST_COLS = [
    "sports", "tvsports", "exercise", "dining", "museums", "art",
    "hiking", "gaming", "clubbing", "reading", "tv", "theater",
    "movies", "concerts", "music", "shopping", "yoga",
]
OPP_BELIEF_COLS = ["attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
SELF_COLS = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]
NORM_COLS = ["attr1_1_norm", "sinc1_1_norm", "intel1_1_norm",
             "fun1_1_norm",  "amb1_1_norm",  "shar1_1_norm"]
PREF_RAW  = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
ID_COLS   = ["iid", "pid", "gender", "wave", "condtn", "round",
             "position", "order", "dec", "dec_o", "match"]


def _recode_met(s):
    out = pd.Series(np.nan, index=s.index)
    out[s == 1] = 1
    out[s.isin([0, 2])] = 0
    return out


def _derive_clean(raw_rows, clean_cols):
    """Row-local cleaning transforms -> clean-parquet schema (from experiments_splits.py)."""
    df = raw_rows.copy()
    df = df[df["pid"].notna()].copy()
    df[INTEREST_COLS] = df[INTEREST_COLS].clip(upper=10)   # raw has >10 entries
    df["met_before"]   = _recode_met(df["met"])
    df["met_before_o"] = _recode_met(df["met_o"])
    df["scale_type"] = np.where(df["wave"].isin([6, 7, 8, 9]), "1-10", "100pt")
    psum = df[PREF_RAW].sum(axis=1, min_count=6)
    valid = psum > 0
    pref_norm = [c.replace("1_1", "1_1_norm") for c in PREF_RAW]
    df[pref_norm] = np.nan
    df.loc[valid, pref_norm] = df.loc[valid, PREF_RAW].div(psum[valid], axis=0).values
    df["age_diff"] = (df["age"] - df["age_o"]).abs()
    both = df["race"].notna() & df["race_o"].notna()
    df["race_match"] = np.where(both, (df["race"] == df["race_o"]).astype(float), np.nan)
    return df.reindex(columns=clean_cols)


def _validate_reconstruction(clean_df, raw, check_wave=11):
    """Re-derive a known wave from raw; require equality on model columns."""
    CAT_COLS = ["race", "race_o", "field_cd", "career_c", "goal", "scale_type"]
    CONT = (["age", "age_o", "imprace", "imprelig", "exphappy",
             "int_corr", "date", "go_out", "age_diff"]
            + INTEREST_COLS + OPP_BELIEF_COLS)
    BIN  = ["samerace", "met_before", "met_before_o", "race_match"]
    check_cols = ID_COLS + CONT + SELF_COLS + NORM_COLS + CAT_COLS + BIN
    recon  = _derive_clean(raw[raw["wave"] == check_wave], clean_df.columns)
    actual = clean_df[clean_df["wave"] == check_wave]
    recon  = recon.sort_values(["iid", "pid"]).reset_index(drop=True)
    actual = actual.sort_values(["iid", "pid"]).reset_index(drop=True)
    assert len(recon) == len(actual), "row count mismatch on validation wave"
    bad = []
    for c in check_cols:
        a, b = actual[c], recon[c]
        if pd.api.types.is_numeric_dtype(a):
            if (a.fillna(-9e9) - b.fillna(-9e9)).abs().max() > 1e-9:
                bad.append(c)
        elif (a.fillna("NA").astype(str) != b.fillna("NA").astype(str)).any():
            bad.append(c)
    assert not bad, f"Reconstruction validation FAILED on: {bad}"


def _row_cosine(A: np.ndarray, B: np.ndarray, min_dims: int = 8) -> np.ndarray:
    """Per-row cosine over mutually observed dims; NaN if < min_dims shared."""
    mask = ~(np.isnan(A) | np.isnan(B))
    A0, B0 = np.where(mask, A, 0.0), np.where(mask, B, 0.0)
    num = (A0 * B0).sum(1)
    den = np.sqrt((A0**2).sum(1)) * np.sqrt((B0**2).sum(1))
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return np.where(mask.sum(1) >= min_dims, out, np.nan)


# Profile columns joined onto the B side (pre-event only -> cold-start legal)
PROFILE_JOIN = SELF_COLS + INTEREST_COLS + NORM_COLS + ["gender"]


def build_two_sided(clean_df: pd.DataFrame) -> pd.DataFrame:
    prof = clean_df.drop_duplicates("iid")[["iid"] + PROFILE_JOIN].copy()
    out = clean_df.merge(
        prof.rename(columns={c: f"{c}_B" for c in PROFILE_JOIN}),
        left_on="pid", right_on="iid", how="left", suffixes=("", "_dropme"),
    )
    out = out.drop(columns=[c for c in out.columns if c.endswith("_dropme")])
    out["int_cos"] = _row_cosine(
        out[INTEREST_COLS].to_numpy(float),
        out[[f"{c}_B" for c in INTEREST_COLS]].to_numpy(float),
    )
    return out


# ---- execute ----------------------------------------------------------------
clean20 = pd.read_parquet(CLEAN_PARQUET)
raw = pd.read_csv(RAW_CSV, encoding="latin-1").rename(columns={"Unnamed: 0": "iid"})
_validate_reconstruction(clean20, raw, check_wave=11)
print("Wave-11 reconstruction gate: PASSED")

w12 = _derive_clean(raw[raw["wave"] == 12], clean20.columns)
clean21 = pd.concat([clean20, w12], ignore_index=True)

DATA = {
    "drop": build_two_sided(clean20),   # 20 waves
    "keep": build_two_sided(clean21),   # 21 waves
}
DATA_SHA = {"clean_parquet": file_sha(CLEAN_PARQUET), "raw_csv": file_sha(RAW_CSV)}

for k, d in DATA.items():
    fp = df_fingerprint(d)
    print(f"{k}: rows={fp['n_rows']:,}  waves={fp['n_waves']}  "
          f"dec={fp['dec_rate']}  match={fp['match_rate']}")
b_missing = DATA["drop"][[f"{c}_B" for c in PROFILE_JOIN]].isna().mean().mean()
print(f"mean missing share in _B cols (pre-impute): {b_missing:.2%}")
print(f"int_cos NaN share: {DATA['drop']['int_cos'].isna().mean():.2%}")
```

**Verify:**
- gate prints PASSED (if it raises, the parquet changed — STOP and diff);
- `drop`: rows = 7,976, waves = 20, match ≈ 0.168;
- `keep`: rows = 8,368, waves = 21, match ≈ 0.165;
- `_B` missingness small (<5%); `int_cos` NaN share <2%.

---

## Cell 0-4 — Column groups, feature lists, theme map

```python
CONT_COLS = (["age", "age_o", "imprace", "imprelig", "exphappy",
              "int_corr", "date", "go_out", "age_diff"]
             + INTEREST_COLS + OPP_BELIEF_COLS)
BIN_COLS  = ["samerace", "met_before", "met_before_o", "race_match"]
SELF_Z      = [c + "_z" for c in SELF_COLS]                 # A-side z-scores
SELF_B      = [f"{c}_B" for c in SELF_COLS]
SELF_B_Z    = [c + "_z" for c in SELF_B]                    # B-side z-scores
INTEREST_B  = [f"{c}_B" for c in INTEREST_COLS]
NORM_B      = [f"{c}_B" for c in NORM_COLS]

TRAITS = ["attr", "sinc", "intel", "fun", "amb"]
ALIGN_COLS = [f"align_{t}" for t in TRAITS] + ["align_shar"]

# Stage 1 CB feature list (27, all numeric, plan section 4)
S1_FEATS = (ALIGN_COLS + NORM_COLS + SELF_B_Z
            + ["int_cos", "int_corr",
               "age_diff", "samerace", "race_match", "imprace", "imprelig",
               "date", "go_out", "exphappy", "met_before", "met_before_o"])

# Stage 2 pair-kNN distance space
KNN_COLS = (CONT_COLS + SELF_Z + NORM_COLS + BIN_COLS
            + SELF_B_Z + INTEREST_B + NORM_B + ["int_cos"])

# Stage 3 FM context block (numeric only by design, plan section 4)
CTX_COLS = ["age_diff", "samerace", "race_match", "met_before", "met_before_o",
            "int_corr", "int_cos", "exphappy", "scale_type_code", "condtn"]

THEME_MAP = {
    **{f"align_{t}": th for t, th in zip(
        TRAITS, ["Attractiveness", "Sincerity", "Intelligence", "Fun", "Ambition"])},
    "align_shar": "Shared interests",
    **{c: th for c, th in zip(
        NORM_COLS, ["Attractiveness", "Sincerity", "Intelligence",
                    "Fun", "Ambition", "Shared interests"])},
    **{c: th for c, th in zip(
        SELF_B_Z, ["Attractiveness", "Sincerity", "Intelligence", "Fun", "Ambition"])},
    "int_cos": "Shared interests", "int_corr": "Shared interests",
    "age_diff": "Demographics fit", "samerace": "Demographics fit",
    "race_match": "Demographics fit", "imprace": "Demographics fit",
    "imprelig": "Demographics fit",
    "date": "Dating attitude", "go_out": "Dating attitude",
    "exphappy": "Dating attitude",
    "met_before": "Familiarity", "met_before_o": "Familiarity",
}
THEMES = sorted(set(THEME_MAP.values()))
assert set(S1_FEATS) <= set(THEME_MAP), "every Stage-1 feature needs a theme"
print(f"S1 feats: {len(S1_FEATS)} | kNN dims: {len(KNN_COLS)} | themes: {THEMES}")
```

**Verify:** S1 feats = 27; no assert errors; 6 themes listed.

---

## Cell 0-5 — The 10 splitting schemes (+ structural assertions)

Fold-placement rules for keep-w12 variants are the plan §6 table, verbatim.

```python
DECK_FOLDS = {"fold_1": [1, 4, 5, 17, 19, 20, 21],
              "fold_2": [3, 6, 7, 11, 13, 15, 18],
              "fold_3": [2, 8, 9, 10, 14, 16]}
S2_CANON_CV = {"cv0": [2, 5, 6, 7, 11, 13],     # run_preprocessing.py scheme_2
               "cv1": [4, 8, 17, 20, 21],
               "cv2": [3, 10, 14, 16, 19]}
HOLD_A, HOLD_B, HOLD_W19 = [1, 9, 15, 18], [8, 13, 14, 19], [19]


def make_3_folds(pool_waves, df):
    """Round-robin pool waves by ascending match rate into 3 balanced folds."""
    mr = (df[df["wave"].isin(pool_waves)]
          .groupby("wave")["match"].mean().sort_values())
    folds = {0: [], 1: [], 2: []}
    for i, w in enumerate(mr.index.tolist()):
        folds[i % 3].append(int(w))
    return folds


def _cv_splits(folds: dict, tag_prefix="cv"):
    keys = list(folds)
    return [{"tag": f"{tag_prefix}{i}", "kind": "cv",
             "train_waves": sorted(w for j, k2 in enumerate(keys) if j != i
                                   for w in folds[k2]),
             "eval_waves": sorted(folds[k])}
            for i, k in enumerate(keys)]


def scheme_splits(scheme_id: str, df: pd.DataFrame):
    """Return [split dicts] for S1..S10. Each split:
    {tag, kind: cv|lowo|holdout, train_waves, eval_waves}."""
    waves = sorted(int(w) for w in df["wave"].unique())

    def lowo(rotation_waves, all_waves):
        return [{"tag": f"wave{w}", "kind": "lowo",
                 "train_waves": [x for x in all_waves if x != w],
                 "eval_waves": [w]} for w in rotation_waves]

    if scheme_id in ("S1", "S2"):                      # plain LOWO
        return lowo(waves, waves)

    if scheme_id in ("S3", "S4"):                      # deck tri-fold
        folds = {k: list(v) for k, v in DECK_FOLDS.items()}
        if scheme_id == "S3":
            folds["fold_3"] = folds["fold_3"] + [12]   # smallest by participants
        return _cv_splits(folds, "fold")

    if scheme_id in ("S5", "S6"):                      # canonical cv + holdout A
        folds = {k: list(v) for k, v in S2_CANON_CV.items()}
        if scheme_id == "S5":
            folds["cv2"] = folds["cv2"] + [12]         # smallest by pair count
        pool = sorted(w for ws in folds.values() for w in ws)
        out = _cv_splits(folds)
        out.append({"tag": "holdout", "kind": "holdout",
                    "train_waves": pool, "eval_waves": sorted(HOLD_A)})
        return out

    if scheme_id in ("S7", "S8"):                      # round-robin cv + holdout B
        pool = [w for w in waves if w not in HOLD_B]
        folds = make_3_folds(pool, df)
        out = _cv_splits({f"rr{k}": v for k, v in folds.items()})
        out.append({"tag": "holdout", "kind": "holdout",
                    "train_waves": sorted(pool), "eval_waves": sorted(HOLD_B)})
        return out

    if scheme_id in ("S9", "S10"):                     # LOWO + wave-19 holdout
        rot = [w for w in waves if w not in HOLD_W19]
        out = lowo(rot, rot)                           # 19 never in rotation
        out.append({"tag": "holdout_w19", "kind": "holdout",
                    "train_waves": rot, "eval_waves": HOLD_W19})
        return out

    raise ValueError(scheme_id)


SCHEME_VARIANT = {"S1": "keep", "S2": "drop", "S3": "keep", "S4": "drop",
                  "S5": "keep", "S6": "drop", "S7": "keep", "S8": "drop",
                  "S9": "keep", "S10": "drop"}

# ---- structural assertions ---------------------------------------------------
for sid, var in SCHEME_VARIANT.items():
    df = DATA[var]
    waves = set(int(w) for w in df["wave"].unique())
    for sp in scheme_splits(sid, df):
        tr, ev = set(sp["train_waves"]), set(sp["eval_waves"])
        assert tr.isdisjoint(ev), f"{sid}/{sp['tag']}: train∩eval"
        assert tr | ev <= waves,  f"{sid}/{sp['tag']}: unknown wave"
        sub = df[df["wave"].isin(ev)]
        pairs = set(map(tuple, sub[["iid", "pid"]].astype(int).values))
        sample = list(pairs)[:25]
        assert all((b, a) in pairs for a, b in sample), \
            f"{sid}/{sp['tag']}: mirror pair missing"
    n = len(scheme_splits(sid, df))
    print(f"{sid} ({var}): {n} splits OK")
```

**Verify:**
- split counts: S1=21, S2=20, S3/S4=3, S5–S8=4, S9=21, S10=20;
- no assertion errors;
- spot-check: `scheme_splits("S8", DATA["drop"])[0]["eval_waves"]` should be
  `[1, 3, 4, 9, 15, 18]` (round-robin fold 0 — same folds as
  `experiments_splits.py` printed).

---

## Cell 0-6 — Leakage-safe preprocessor (fit on train only)

Adds, per split: median imputation → standardization → gender z-scores for
**both** sides (B z-scored by B's own gender from the profile join) →
the 6 alignment features → `scale_type_code`.

```python
IMP_COLS = (CONT_COLS + SELF_COLS + NORM_COLS + BIN_COLS
            + SELF_B + INTEREST_B + NORM_B + ["int_cos"])
SCALE_COLS = CONT_COLS + INTEREST_B + ["int_cos"]


def fit_preprocessor(train_df: pd.DataFrame) -> dict:
    fills = train_df[IMP_COLS].median()
    scaler = StandardScaler().fit(train_df[SCALE_COLS].fillna(fills[SCALE_COLS]))
    t = train_df.copy()
    t[IMP_COLS] = t[IMP_COLS].fillna(fills)
    z = {}
    for g in (0, 1):
        a_rows, b_rows = t[t["gender"] == g], t[t["gender_B"] == g]
        for c in SELF_COLS:
            z[f"A|{g}|{c}"] = (float(a_rows[c].mean()), float(a_rows[c].std()))
            z[f"B|{g}|{c}"] = (float(b_rows[f"{c}_B"].mean()),
                               float(b_rows[f"{c}_B"].std()))
    return {"fills": fills, "scaler": scaler, "z": z}


def apply_preprocessor(frame: pd.DataFrame, pp: dict) -> pd.DataFrame:
    out = frame.copy()
    out[IMP_COLS] = out[IMP_COLS].fillna(pp["fills"])
    out[SCALE_COLS] = pp["scaler"].transform(out[SCALE_COLS])
    for side, gcol, suff in (("A", "gender", ""), ("B", "gender_B", "_B")):
        for c in SELF_COLS:
            zc = f"{c}{suff}_z"
            out[zc] = np.nan
            for g in (0, 1):
                mu, sd = pp["z"][f"{side}|{g}|{c}"]
                m = out[gcol] == g
                out.loc[m, zc] = ((out.loc[m, f"{c}{suff}"] - mu) / sd) if sd > 0 else 0.0
    # alignment terms: A's stated weight x B's (gender-relative) self-rating.
    # int_cos is standardized => align_shar sign reads "above/below-average overlap".
    for t_, pref, bz in zip(TRAITS,
                            NORM_COLS[:5], SELF_B_Z):
        out[f"align_{t_}"] = out[pref] * out[bz]
    out["align_shar"] = out["shar1_1_norm"] * out["int_cos"]
    out["scale_type_code"] = (out["scale_type"] == "100pt").astype(int)
    return out


def process_split(df, train_waves, eval_waves):
    tr_raw = df[df["wave"].isin(train_waves)].copy()
    ev_raw = df[df["wave"].isin(eval_waves)].copy()
    pp = fit_preprocessor(tr_raw)
    return apply_preprocessor(tr_raw, pp), apply_preprocessor(ev_raw, pp), pp


# smoke test on S8 fold 0
_sp = scheme_splits("S8", DATA["drop"])[0]
_tr, _ev, _pp = process_split(DATA["drop"], _sp["train_waves"], _sp["eval_waves"])
_needed = set(S1_FEATS + KNN_COLS + CTX_COLS)
assert not _tr[list(_needed)].isna().any().any(), "NaN left in train features"
assert not _ev[list(_needed)].isna().any().any(), "NaN left in eval features"
print(f"smoke OK: train={len(_tr):,} eval={len(_ev):,} | "
      f"align_attr range [{_tr['align_attr'].min():.2f}, {_tr['align_attr'].max():.2f}]")
```

**Verify:** smoke prints; no NaN asserts fire; align ranges look like
(weight × z) products, roughly [−2.5, +2.5].

---

## Cell 0-7 — Metrics (byte-compatible with `experiments_splits.py`)

```python
def reciprocal_scores(test_df, p):
    """Fuse directed scores: unilateral, harmonic mean, geometric mean."""
    lut = {(int(i), int(j)): float(s)
           for (i, j), s in zip(zip(test_df["iid"], test_df["pid"]), p)}

    def hm(a, b):
        return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)

    rev = [lut.get((int(j), int(i)), float(s))
           for (i, j, s) in zip(test_df["iid"], test_df["pid"], p)]
    p = np.asarray(p, dtype=float)
    rev = np.asarray(rev, dtype=float)
    return {"uni": p,
            "recip_hm": np.array([hm(a, b) for a, b in zip(p, rev)]),
            "recip_gm": np.sqrt(p * rev),
            "rev": rev}


def mirror_col(df, col):
    """Value of `col` on the mirror row (pid,iid); falls back to own value."""
    lut = {(int(i), int(j)): v
           for i, j, v in zip(df["iid"], df["pid"], df[col])}
    return np.array([lut.get((int(j), int(i)), v)
                     for i, j, v in zip(df["iid"], df["pid"], df[col])])


def ndcg_at5(df, score_col):
    """Mean NDCG@5 per user (iid); skip users with no positive match."""
    vals = []
    for _, g in df.groupby("iid"):
        if g["match"].sum() == 0:
            continue
        vals.append(ndcg_score(g[["match"]].T.values,
                               g[[score_col]].T.values, k=5))
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def prec_at_k(df, score_col, k):
    """Mean precision@k over users with >=1 true match."""
    vals = []
    for _, g in df.groupby("iid"):
        if g["match"].sum() == 0:
            continue
        vals.append(g.nlargest(k, score_col)["match"].mean())
    return float(np.mean(vals)) if vals else float("nan")


def coverage_at5(df, score_col):
    """Share of true matched (unordered) pairs appearing in either member's top-5."""
    top5 = {}
    for iid, g in df.groupby("iid"):
        top5[int(iid)] = set(g.nlargest(5, score_col)["pid"].astype(int))
    pairs = {tuple(sorted((int(a), int(b))))
             for a, b in df.loc[df["match"] == 1, ["iid", "pid"]].values}
    if not pairs:
        return float("nan")
    hit = sum((b in top5.get(a, set())) or (a in top5.get(b, set()))
              for a, b in pairs)
    return hit / len(pairs)

print("Metrics ready (v1, reference-compatible).")
```

**Verify:** quick sanity — `ndcg_at5` on a frame where the match is always
top-ranked returns 1.0.

---

## Done — proceed to `STAGE1_CONTENT_BASED.md`

After these seven cells run clean, Cell S1-0's prerequisite check will pass.
