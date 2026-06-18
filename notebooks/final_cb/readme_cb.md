# Stage 1 — Content-Based Model (Scheme S9)

This folder is the **frozen deliverable** for Stage 1 of the speed-dating recommender: a leakage-safe, wave-based content-based (CB) scoring pipeline using logistic regression on engineered pair features.

Downstream teams (explainability, collaborative filtering, hybrid ranking) should treat this as the **source of truth** for feature definitions, evaluation protocol, champion hyperparameters, and exported scores.

---

## Folder layout

```
final_cb/
├── readme_cb.md                          ← this file
├── s9_foundation.ipynb                   ← data, features, S9 splits, metrics
├── s9_content_based_model.ipynb          ← tuning, feature reduction, eval, export
├── s9_feature_engineering.py             ← E4_Full feature build (134 cols)
└── artifacts/
    ├── stage1_s9_feature_list_*.json     ← 91 reduced features + themes
    ├── stage1_s9_summary_*.json          ← run provenance + metrics snapshot
    └── s9_scores_eval.json               ← full eval (AUC, overfit gap, ranking)
```

**Repo dependencies (not in this folder):**

| Path | Role |
|------|------|
| `preprocessing/speed_dating_clean_keep_w12.parquet` | Source data (8,368 rows, 21 waves, wave 12 kept) |
| `load_one_sided_data.py` (repo root) | Raw one-sided column list for `_A`/`_B` join |
| `notebooks/eval_s9_scores.py` | Re-evaluate exports without retraining |
| `results/stage1_s9_scores.{csv,parquet}` | Full score export (written when notebooks run from repo root) |
| `results/stage1_s9_features.{csv,parquet}` | Imputed feature matrix per split |

---

## Quick start

### 1. Environment

From the **repository root**:

```bash
pip install pandas pyarrow numpy scikit-learn jupyter
pip install "interpret>=0.4"   # only needed for EBM tuning cell
```

Run notebooks with the repo root (or any parent of `preprocessing/`) as the working directory so paths resolve correctly.

### 2. Run order

| Step | Notebook | Action |
|------|----------|--------|
| 1 | `s9_foundation.ipynb` | Run **all cells** top-to-bottom |
| 2 | `s9_content_based_model.ipynb` | Run **all cells** in the **same kernel**, or open alone (cell 1 auto-loads foundation) |

Expected foundation output:

```
Excluded redundant: ['gender_B']
Raw: 89 | Engineered: 45 | Total: 134
S9: 21 splits (20 LOWO + 1 holdout)
```

### 3. Re-evaluate existing scores (no retrain)

```bash
python notebooks/eval_s9_scores.py \
  --scores results/stage1_s9_scores.csv \
  --feature-list notebooks/final_cb/artifacts/stage1_s9_feature_list_20260618_162935.json \
  --out-json notebooks/final_cb/artifacts/s9_scores_eval.json
```

Use `--skip-refit` for export + ranking only (skip train/eval gap table).

---

## Methodology

### Problem framing

Each row is a **directed pair** `(iid → pid)` at a speed-dating event (`wave`).

| Column | Meaning |
|--------|---------|
| `dec` | Did `iid` want to meet `pid` again? (**LR training target**) |
| `dec_o` | Partner's decision (reverse direction) |
| `match` | Mutual yes (`dec` and partner's `dec` both 1) — used for **ranking** metrics only |

The CB model predicts **unilateral interest** (`dec`). Ranking and reciprocal scores combine both directions for recommendation.

### Scheme S9 (evaluation protocol)

Designed to avoid leakage across event waves:

- **20 LOWO folds:** for each rotation wave `w ≠ 19`, train on all other rotation waves, score only wave `w` (out-of-fold).
- **1 holdout:** train on all rotation waves, score **wave 19** only (never used in tuning or feature selection).
- **Wave 12 is kept** (not dropped like in some earlier experiments).

Preprocessing is **fit on train waves only** per split:

1. `SimpleImputer(strategy="mean")` on feature columns  
2. `StandardScaler` on train features inside `Stage1LR` (LR only)

**Do not use** `data/two_sided_pair_features.csv` — it applies global imputation and causes leakage.

### Feature set (E4_Full → reduced)

Built in `s9_feature_engineering.py` from keep-w12 parquet:

| Stage | Count | Description |
|-------|------:|-------------|
| Raw `_A` / `_B` | 89 | All profile dummies; **`gender_B` dropped** (perfect complement of `gender_A`) |
| Engineered | 45 | Pref-match, interest diffs, `diff_age`, `same_race`, aspiration gaps, etc. |
| **E4_Full total** | **134** | Full matrix before reduction |
| **Champion set** | **91** | 95% cumulative LR importance (see below) |

Engineered groups:

1. Preference match (`pref_match_A_to_B`, `pref_match_B_to_A`, `pref_match_mutual`)
2. Interest differences + cosine similarity
3. Demographic / lifestyle diffs (`diff_age`, `same_race`, …)
4. Aspiration gaps, pref entropy, `exphappy_diff` / `exphappy_sum`

Feature list with themes: `artifacts/stage1_s9_feature_list_*.json`.

### Feature reduction (Section 3 of content notebook)

Mirrors `feature_engineering_experiment.ipynb` Phase 2, adapted for S9:

1. Fit **LR C=10** on each of the **20 LOWO folds** (wave 19 excluded).
2. Average **|coefficient|** across folds.
3. Normalize to sum to 1; keep the **smallest prefix ≥ 95%** cumulative importance.
4. Result: **91 features** at **95.2%** coverage (LR-only; no SHAP blend).

After reduction, `S1_FEATS` and `CB_FEATS` are updated for tuning, eval, and export.

### Model selection

| Candidate | LOWO AUC (tuning) | Notes |
|-----------|-------------------|--------|
| `lr_C10` | **0.6269** | **Champion** — L2 logistic regression, C=10 |
| `lr_C1` | 0.6243 | |
| `lr_C0.1` | 0.6243 | |
| `ebm` | 0.6017 | Explainable Boosting Machine (not selected) |

Champion: **`LogisticRegression(C=10, penalty="l2", max_iter=2000, random_state=42)`** on **91 reduced features**.

### Metrics

#### AUC (classification — target `dec`, score `score_cb`)

| Metric | Champion value | Interpretation |
|--------|----------------:|----------------|
| LOWO eval AUC (mean ± std) | **0.6468 ± 0.0566** | OOF on 20 rotation waves |
| Holdout eval AUC (wave 19) | **0.6905** | Fully unseen wave |

Per-fold table: `artifacts/s9_scores_eval.json` → `auc_export.folds`.

#### Overfitting check (train vs eval AUC, same refit model)

| Metric | Value |
|--------|------:|
| LOWO mean train AUC | 0.7008 |
| LOWO mean eval AUC | 0.6468 |
| LOWO mean gap (train − eval) | **0.0539** ± 0.0588 |
| Holdout gap | **0.0086** |

A modest positive gap on LOWO is expected for in-sample vs OOF eval. Holdout gap ≈ 0.01 indicates the model does not collapse on unseen wave 19. Details: `s9_scores_eval.json` → `overfit_gap`.

#### Ranking @ k=5 (relevance `match`)

| Split | Condition | MM@5 | NDCG@5 |
|-------|-----------|-----:|-------:|
| Train (waves ≠ 19) | unilateral | 0.4279 | 0.3376 |
| Train | reciprocal HM | 0.4958 | 0.3973 |
| Test (wave 19) | unilateral | 0.6573 | 0.5198 |
| Test | reciprocal HM | 0.6613 | 0.5545 |

- **Unilateral:** rank by `score_cb` (A's predicted interest in B).
- **Reciprocal HM:** rank by harmonic mean of `score_cb` and `score_cb_rev` (partner's reverse score).

Train ranking uses **OOF scores** (harder); test uses the holdout model. Test can exceed train for this reason and because wave 19 has fewer users (~25 with matches).

---

## Exports and downstream consumption

### When notebooks finish (repo root `results/`)

| File | Contents |
|------|----------|
| `stage1_s9_scores.csv` | Per-row `score_cb`, `score_cb_rev`, theme contributions, `split_tag`, IDs, labels |
| `stage1_s9_features.csv` | Imputed **91** feature values per row per split |
| `runs/stage1_s9_feature_list_*.json` | Feature names + themes + reduction metadata |
| `runs/stage1_s9_summary_*.json` | Config, metrics, champion spec |

### Copies in `artifacts/`

Frozen snapshots for GitHub / handoff without requiring a full re-run.

### For explainability teammates

Minimum handoff:

1. `artifacts/stage1_s9_feature_list_*.json` — feature contract (order matters)
2. `artifacts/stage1_s9_summary_*.json` — model spec (LR C=10, 91 feats)
3. `results/stage1_s9_scores.*` — scores + **theme-level** contributions (`contrib_Engineered`, `contrib_Profile A`, `contrib_Profile B`)
4. `results/stage1_s9_features.*` — imputed inputs used at scoring time
5. This README + `s9_feature_engineering.py` — how features are built

**LR native explanations** (exact decomposition, no SHAP required):

```python
contrib_i = scaled_x_i * coef_i   # sums to logit + intercept
```

Implement via `Stage1LR.contributions()` in `s9_content_based_model.ipynb`.

**Important:** exported scores are from **21 fold-specific refits** (one model per S9 split), not a single global pickle. For a deployment/explanation model, refit once on all rotation waves (≠ 19) and save imputer + scaler + LR together.

### For CF / hybrid teammates

Use `stage1_s9_scores.csv`:

| Column | Use |
|--------|-----|
| `iid`, `pid`, `wave` | Keys |
| `score_cb` | Stage-1 unilateral CB score |
| `score_cb_rev` | Reverse direction score |
| `match` | Ground truth for ranking eval |

Combine with your CF scores as needed; reciprocal HM is already defined in foundation (`add_reciprocal_scores`).

---

## Pipeline diagram

```
keep-w12.parquet
       │
       ▼
s9_feature_engineering.py  ──►  E4_Full (134 cols)
       │
       ▼
s9_foundation.ipynb        ──►  S9 splits + per-split mean impute
       │
       ▼
s9_content_based_model.ipynb
   ├─ §2  Tune LR vs EBM (LOWO only)
   ├─ §3  Reduce 134 → 91 (95% LR importance)
   ├─ §4  AUC eval (LOWO + holdout)
   ├─ §5  Ranking MM@5 / NDCG@5
   └─ §6  Export scores + features + JSON
       │
       ▼
eval_s9_scores.py          ──►  AUC + overfit gap + ranking report
```

---

## Design choices and anti-patterns

| Do | Don't |
|----|-------|
| Mean-impute per split on train waves only | Global imputation from full CSV |
| Keep wave 19 for holdout only | Tune or select features on wave 19 |
| Use `dec` for AUC / LR training | Train LR on `match` (different task) |
| Use `match` for ranking NDCG / MM@5 | Report in-sample train AUC as generalization |
| Read feature order from JSON | Reorder or subset features silently |

---

## Reference notebooks (repo root)

| Notebook | Role |
|----------|------|
| `notebooks/feature_engineering_experiment.ipynb` | Original E4 feature design + Phase 2 reduction reference |
| `preprocessing/speed_dating_eda_data_cleaning_keep_w12.ipynb` | How keep-w12 parquet was built |

---

## Contact / provenance

- **Scheme:** S9  
- **Champion:** `lr_C10` on **91** reduced E4 features  
- **Data SHA (keep parquet):** see `data_sha` in `stage1_s9_summary_*.json`  
- **Last eval snapshot:** `artifacts/s9_scores_eval.json` (2026-06-18)

For questions on splits, leakage, or feature definitions, start with `s9_foundation.ipynb`. For model tuning and exports, see `s9_content_based_model.ipynb`.
