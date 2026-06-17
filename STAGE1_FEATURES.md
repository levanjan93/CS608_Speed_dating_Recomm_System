# Stage 1 Content-Based Features (`S1_FEATS`)

Reference for the **29 numeric features** fed to Stage 1 logistic regression / EBM (`content_based_model.ipynb`). Defined in `pipeline_foundation.py` (Cell 0-4) and materialized by `process_split()` (Cell 0-6).

**Target:** `dec` (did participant A want to see partner B again?)  
**Direction:** unilateral A → B. Features describe A’s stated preferences, B’s profile (joined on `pid`), and pair-level context.

---

## Pipeline stages (where values are created)

| Stage | What happens | Code |
|-------|----------------|------|
| **1. Raw CSV** | Columbia `Speed Dating Data.csv` — signup (Time 1) + per-date rows | — |
| **2. Cleaning** | Row filters, recodes, normalized prefs, pair derivations → `speed_dating_clean.parquet` | `_derive_clean()` |
| **3. Two-sided join** | Partner pre-event profile joined as `*_B`; `int_cos` computed | `build_two_sided()` |
| **4. Split preprocessor** | Train-fold median impute → standardize continuous cols → gender z-scores → alignment products | `process_split()` |

All statistics in stage 4 (imputation, scaling, z-scores) are **fit on training waves only** per CV fold.

---

## Feature inventory by theme

### Attractiveness (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_attr` | **Derived** | `attr1_1` (A), `attr3_1` (B via `pid`) | Cleaning: `attr1_1_norm = attr1_1 / Σ prefs`. Join `attr3_1_B`. Preprocessor: gender z-score on B → `attr3_1_B_z`. **`align_attr = attr1_1_norm × attr3_1_B_z`** |
| `attr1_1_norm` | **Derived** | `attr1_1` | Sum-to-1 over A’s six preference weights (`attr1_1`…`shar1_1`); fixes mixed 100-pt vs 1–10 waves |
| `attr3_1_B_z` | **Derived** | `attr3_1` (partner) | Partner self-rating joined as `attr3_1_B`; z-scored within **B’s gender** on train fold |

### Sincerity (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_sinc` | **Derived** | `sinc1_1`, `sinc3_1` (B) | Same pattern as `align_attr` |
| `sinc1_1_norm` | **Derived** | `sinc1_1` | Preference normalization (cleaning) |
| `sinc3_1_B_z` | **Derived** | `sinc3_1` (B) | Partner join + gender z-score |

### Intelligence (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_intel` | **Derived** | `intel1_1`, `intel3_1` (B) | Same alignment pattern |
| `intel1_1_norm` | **Derived** | `intel1_1` | Preference normalization |
| `intel3_1_B_z` | **Derived** | `intel3_1` (B) | Partner join + gender z-score |

### Fun (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_fun` | **Derived** | `fun1_1`, `fun3_1` (B) | Same alignment pattern |
| `fun1_1_norm` | **Derived** | `fun1_1` | Preference normalization |
| `fun3_1_B_z` | **Derived** | `fun3_1` (B) | Partner join + gender z-score |

### Ambition (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_amb` | **Derived** | `amb1_1`, `amb3_1` (B) | Same alignment pattern |
| `amb1_1_norm` | **Derived** | `amb1_1` | Preference normalization |
| `amb3_1_B_z` | **Derived** | `amb3_1` (B) | Partner join + gender z-score |

### Shared interests (4 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `align_shar` | **Derived** | `shar1_1`, 17× interest cols (A & B) | `shar1_1_norm × int_cos` (after `int_cos` is standardized) |
| `shar1_1_norm` | **Derived** | `shar1_1` | Preference normalization |
| `int_cos` | **Derived** | `sports`…`yoga` (A), same on B | Row-wise cosine similarity of 17-dim interest vectors; requires ≥8 mutually observed dims; then **StandardScaler** on train |
| `int_corr` | **Dataset** | `int_corr` | Pre-computed by Columbia: correlation of A’s and B’s interest ratings; **StandardScaler** on train |

### Demographics fit (5 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `age_diff` | **Derived** | `age`, `age_o` | `\|age − age_o\|` (cleaning); **StandardScaler** on train |
| `samerace` | **Dataset** | `samerace` | 1 = same race, 0 = not (study-provided); median-imputed only if missing (rare) |
| `race_match` | **Derived** | `race`, `race_o` | 1 if codes equal, 0 if different, NaN if either missing → **train median** impute (typically 0) |
| `imprace` | **Dataset** | `imprace` | Signup: importance (1–10) that a date be same race/ethnicity (**A only**); **StandardScaler** on train |
| `imprelig` | **Dataset** | `imprelig` | Signup: importance (1–10) that a date be same religion (**A only**); **StandardScaler** on train |

### Dating attitude (3 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `date` | **Dataset** | `date` | How often A goes on dates (coded scale); **StandardScaler** on train |
| `go_out` | **Dataset** | `go_out` | How often A goes out (1–7); **StandardScaler** on train |
| `exphappy` | **Dataset** | `exphappy` | How happy A expects to be from the event (1–10); **StandardScaler** on train |

### Familiarity (2 features)

| Feature | Source | Raw dataset column(s) | Engineering |
|---------|--------|----------------------|-------------|
| `met_before` | **Derived** | `met` | Recode: `met == 1` → 1, `met ∈ {0,2}` → 0 (cleaning) |
| `met_before_o` | **Derived** | `met_o` | Same recode for partner’s “met before?” response |

---

## Summary counts

| Origin | Count | Features |
|--------|------:|----------|
| **Dataset (kept / pair flag from study)** | 8 | `int_corr`, `samerace`, `imprace`, `imprelig`, `date`, `go_out`, `exphappy` + partner columns consumed inside z-scores |
| **Cleaning derivations** | 9 | `*_norm` (6), `age_diff`, `race_match`, `met_before`, `met_before_o` |
| **Two-sided join + geometry** | 1 | `int_cos` |
| **Preprocessor derivations** | 11 | `align_*` (6), `*_B_z` (5) |

**Total in `S1_FEATS`:** 29 (6 alignment + 6 norm + 5 B z-scores + 12 context/attitude/familiarity).

---

## Preprocessing applied before modeling (by column group)

| Group | Median impute | StandardScaler | Other |
|-------|:-------------:|:--------------:|-------|
| `*_norm` (6) | ✓ | — | Passthrough after impute |
| `*_B_z` (5) | ✓ (on raw `*_B` first) | — | Gender z-score on train |
| `align_*` (6) | — | — | Product of imputed/scaled parents |
| `int_cos`, `int_corr`, `age_diff`, `imprace`, `imprelig`, `date`, `go_out`, `exphappy` | ✓ | ✓ | In `CONT_COLS` / `SCALE_COLS` |
| `samerace`, `race_match`, `met_before`, `met_before_o` | ✓ | — | Binary flags in `BIN_COLS` |

---

## Raw columns used but **not** in `S1_FEATS`

These appear in cleaning or the two-sided table but are excluded from Stage 1 CB (used in later stages or omitted by design):

| Raw / derived | Why not in S1 |
|---------------|----------------|
| 17 interests × `_A` / `_B` (34 cols) | Summarized by `int_cos` + `int_corr` |
| `attr2_1`…`shar2_1` (opposite-sex beliefs) | Deferred to Stage 2 kNN (`CONT_COLS`) |
| `race`, `race_o`, `field_cd`, `career_c`, `goal` | Categorical profile codes; pair race via `samerace` / `race_match`; OHE deferred |
| `age`, `age_o` | Pair gap via `age_diff` |
| B preferences (`attr1_1_B`, …) | Reciprocal direction → Stage 4 fusion |
| A self-ratings (`attr3_1_A`, …) on A side | B’s self-ratings used for alignment; A’s enter kNN as `SELF_Z` in Stage 2 |
| `gender`, `gender_B` | Used for z-scoring; not features (opposite-gender pairs) |
| `scale_type` | Encoding deferred to Stage 3 (`scale_type_code`) |

---

## Design intent (one paragraph)

Stage 1 implements the deck’s content-based story: **A’s normalized preferences**, **B’s gender-relative self-ratings**, and **explicit alignment products** (`pref × self_B_z`), plus **pair fit** (age, race, interests) and **A’s attitudes** (dating frequency, race/religion importance, familiarity). The feature set stays all-numeric and theme-mapped so LR contributions aggregate cleanly onto explanation cards.

---

## Code pointers

- Feature list & themes: `pipeline_foundation.py` → `S1_FEATS`, `THEME_MAP`
- Cleaning derivations: `_derive_clean()`
- Partner join & `int_cos`: `build_two_sided()`
- Impute / scale / z-score / align: `fit_preprocessor()`, `apply_preprocessor()`
- Model usage: `STAGE1_CONTENT_BASED.md` → `Stage1LR`, `Stage1EBM`

*Note: some older docs say “27 features”; the current `S1_FEATS` tuple has **29** names as defined above.*
