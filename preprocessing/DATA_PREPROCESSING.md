# Data Preprocessing — complete ordered list (raw CSV → Stage-1 model input)

Companion to `STAGE1_FEATURES.md` (per-feature inventory). This file is the
**step list**: every transformation between `Speed Dating Data.csv` and the
matrix fed to `Stage1LR` / `Stage1EBM`, in execution order, with the leakage
rule each step obeys. Use it to diff our processing against anyone else's
(teammates' pipeline differs on steps A2, A9, C1 — see §5).

Verified against the code actually run on 2026-06-12:
`speed_dating_eda_data_cleaning.ipynb` (+ `_keep_w12` variant) and
`pipeline_foundation.py` (Cells 0-3, 0-6).

---

## 1. Stage A — Cleaning (one-time, dataset-level)

Notebook: `speed_dating_eda_data_cleaning.ipynb` → `speed_dating_clean.parquet`
(7,976 × 121). Variant: `speed_dating_eda_data_cleaning_keep_w12.ipynb` →
`speed_dating_clean_keep_w12.parquet` (8,368 × 121) — identical except step A2
is skipped. No step below fits statistics on the dataset (all row-local), so
skipping A2 cannot change any other wave's values.

| # | Step | Detail | Rows/values affected |
|---|------|--------|----------------------|
| A1 | Load raw CSV | 8,378 rows × 195 cols, latin-1 encoding | — |
| A2 | **Drop wave 12** | Decision-budget rule (≤50% yes) censors `dec`/`match` labels | −392 rows · **skipped in keep-w12 variant** |
| A3 | Drop null-`pid` rows | 10 wave-5 women rated absent "Man #7" (badge gap) — no partner exists | −10 rows |
| A4 | Zero ratings → NaN | 16 **mid-event** rating cols (`attr…prob`, `*_o`): 0 is off-scale (scales start at 1) | zeros → NaN |
| A5 | Clip mid-event ratings >10 → 10 | same 16 cols; data-entry errors | 2 values |
| A6 | Clip interest ratings >10 → 10 | 17 signup interest cols (`sports…yoga`) | 129 values |
| A7 | Recode "met before" | `met`,`met_o`: 1→1; {0,2}→0 (two no-codes used across waves) → `met_before`, `met_before_o` | — |
| A8 | `scale_type` flag | waves 6–9 = `'1-10'`, others = `'100pt'` (different preference-survey scales) | — |
| A9 | **Normalize signup preferences** | `attr1_1…shar1_1` ÷ row-sum → `*_1_1_norm` (sum = 1). Handles 100-pt vs 1–10 *and* bad sums (90/101/120…) in one stroke. NaN if any of the 6 missing or sum ≤ 0 | — |
| A10 | Pair derivations | `age_diff = \|age − age_o\|`; `race_match` = 1/0/NaN (NaN-aware, unlike `samerace`) | — |
| A11 | Drop unused columns, define roles | 195 → 121 cols; `column_roles.json` records feature vs outcome vs non-feature columns | — |
| A12 | Sanity asserts | rows/waves/participants (7,976/20/523 — variant 8,368/21/551); no zeros or >10 in kept ratings; `*_norm` sums ≈ 1; `match == dec & dec_o` | gate |

Note on A4/A5: those 16 mid-event columns are **not** Stage-1 features
(collected during the date → not available at recommendation time). They are
cleaned for later analysis only.

## 2. Stage B — Foundation data build (one-time, per session)

Code: `pipeline_foundation.py` Cell 0-3.

| # | Step | Detail |
|---|------|--------|
| B1 | Load canonical parquet | `speed_dating_clean.parquet` → `DATA["drop"]` basis |
| B2 | Keep-w12 variant | wave 12 re-derived from raw via `_derive_clean` (steps A3–A10 re-applied row-locally), **gated**: re-deriving wave 11 must reproduce the canonical parquet exactly on every model column. Independent cross-check available: `speed_dating_clean_keep_w12.parquet` from the variant notebook — both routes agree (8,368 rows, match 0.1649) |
| B3 | **Two-sided join** | per-participant pre-event profile (5 self-ratings, 17 interests, 6 norm prefs, gender) deduped on `iid`, joined back on `pid` with `_B` suffix → every directed row (A→B) now carries B's profile |
| B4 | `int_cos` | row-wise cosine of A's and B's 17-dim interest vectors over mutually observed dims (NaN if <8 shared). Static geometry — no fitted parameters, so computed once here |

## 3. Stage C — Per-split preprocessor (fit on TRAINING waves only, every split)

Code: `pipeline_foundation.py` Cell 0-6 (`fit_preprocessor` / `apply_preprocessor`
/ `process_split`). **Leakage rule: every statistic below is computed on the
training waves of the current split and merely *applied* to eval rows.**

| # | Step | Fit on train | Applied to |
|---|------|--------------|-----------|
| C1 | Median imputation | medians of `IMP_COLS` (A-side continuous/self/prefs/binary + B-side self/interests/prefs + `int_cos`) | train + eval |
| C2 | Standardization | `StandardScaler` on `SCALE_COLS` (A-side continuous incl. interests, B-side interests, `int_cos`) | train + eval |
| C3 | Gender z-scores | per-gender mean/std of self-ratings — A's by A's gender, **B's by B's gender** (from the profile join) → `attr3_1_z…`, `attr3_1_B_z…` | train + eval |
| C4 | **Alignment products** | none (products of already-transformed parents): `align_t = pref_norm_t(A) × selfz_t(B)` for the 5 traits; `align_shar = shar1_1_norm × int_cos(std)` | train + eval |
| C5 | `scale_type_code` | none (deterministic recode) | train + eval |

Why C4 exists (measured on S8 CV, LR C=1): full 29 features AUC **0.6146** /
NDCG@5 **0.420**; without the 6 align products **0.6110** / **0.412**;
align + context only (no singles, 18 feats) **0.6071**. The products carry
most of the preference→profile signal in fewer features and are the only
terms a linear model has that can *personalize a user's ranking* (see
`STAGE1_FEATURES.md` design-intent and the alignment rationale discussion).

## 4. Stage D — Stage-1 feature selection: the 29 features, one by one

`S1_FEATS` (exact order from `pipeline_foundation.py`): 6 alignment products +
6 A-preference weights + 5 B self-rating z-scores + 12 pair/attitude/
familiarity features. Target `dec` (does A want to see B again); every row is
a directed pair A→B. Theme/provenance tables: `STAGE1_FEATURES.md`.

Reading guide: "standardized" = z-scored with train-fold mean/std (value ≈
how many SDs above/below the training average); all imputation is train-fold
median (step C1).

### Group 1 — Alignment products (1–6): "is B strong where A cares?"

Each is `(A's normalized preference weight) × (B's gender-relative
self-rating z)`. Positive = B is above-average on a trait A cares about;
negative = B is below-average on a trait A cares about; near 0 = either A
doesn't weight the trait or B is average on it. These are the only terms that
let the linear model rank candidates differently for different users.

| # | Feature | Built from |
|---|---------|-----------|
| 1 | `align_attr` | `attr1_1_norm × attr3_1_B_z` — A's weight on attractiveness × B's self-rated attractiveness (z) |
| 2 | `align_sinc` | `sinc1_1_norm × sinc3_1_B_z` — same for sincerity |
| 3 | `align_intel` | `intel1_1_norm × intel3_1_B_z` — same for intelligence |
| 4 | `align_fun` | `fun1_1_norm × fun3_1_B_z` — same for fun |
| 5 | `align_amb` | `amb1_1_norm × amb3_1_B_z` — same for ambition |
| 6 | `align_shar` | `shar1_1_norm × int_cos(std)` — A's weight on shared interests × how similar the two interest profiles actually are. Uses measured interest overlap instead of a self-rating because "shared interests" is a property of the pair, not of B |

### Group 2 — A's stated preference weights (7–12): "what does A say they want?"

Signup survey: *"You have 100 points to distribute among the following
attributes — give more points to those that are more important in a potential
date"* (waves 6–9 instead rated each 1–10). Step A9 divides by the row sum, so
each value is a **share of A's total caring, 0–1, summing to 1** across the
six — comparable across both survey scales. Passthrough after imputation (not
standardized; the natural 0–1 scale is already comparable). These are
constant across all of A's candidates: they calibrate A's overall
yes-tendency and let the products above be interpreted as deviations.

| # | Feature | Meaning |
|---|---------|---------|
| 7 | `attr1_1_norm` | share of A's 100 points on **attractiveness** (dataset mean ≈ 0.22, the largest) |
| 8 | `sinc1_1_norm` | share on **sincerity** |
| 9 | `intel1_1_norm` | share on **intelligence** |
| 10 | `fun1_1_norm` | share on **fun** |
| 11 | `amb1_1_norm` | share on **ambition** (smallest, ≈ 0.10) |
| 12 | `shar1_1_norm` | share on **shared interests/hobbies** |

### Group 3 — B's self-rated profile (13–17): "how does B present?"

Signup survey to B: *"How do you think you measure up? Rate your opinion of
your own attributes, 1–10 (be honest!)"*. Joined onto A's row via `pid`
(step B3), then z-scored **within B's gender** on the training fold (step
C3) — so +1.0 means "one SD above other men/women", removing the documented
gender gap in self-ratings. Caveat the model must live with: these are
self-perceptions, not measured attractiveness — confident ≠ attractive.
(No 6th entry: the survey has no "shared interests" self-rating — that slot
is covered by `int_cos`/`int_corr` below.)

| # | Feature | Meaning |
|---|---------|---------|
| 13 | `attr3_1_B_z` | B's self-rated **attractiveness**, gender-z |
| 14 | `sinc3_1_B_z` | B's self-rated **sincerity**, gender-z |
| 15 | `intel3_1_B_z` | B's self-rated **intelligence**, gender-z |
| 16 | `fun3_1_B_z` | B's self-rated **fun**, gender-z |
| 17 | `amb3_1_B_z` | B's self-rated **ambition**, gender-z |

### Group 4 — Measured pair compatibility (18–22): "how do A and B actually fit?"

| # | Feature | Built from / wording | Reading |
|---|---------|----------------------|---------|
| 18 | `int_cos` | cosine similarity of A's and B's 17 interest ratings (sports…yoga), computed in step B4, ≥8 mutually observed dims required, then standardized | + = more-similar-than-average hobby profiles; − = less similar |
| 19 | `int_corr` | study-provided *"correlation between participant's and partner's ratings of interests"*; standardized | same idea, Columbia's own overlap measure (correlation vs our cosine — kept both, they disagree just enough to be complementary) |
| 20 | `age_diff` | `\|age_A − age_B\|` (step A10); standardized | + = wider age gap than average (EDA: match rate falls ~20% → ~9% as the gap grows) |
| 21 | `samerace` | study-provided flag *"participant and partner were the same race"* (1/0); binary passthrough | 1 = same race |
| 22 | `race_match` | our NaN-aware recomputation from `race`/`race_o` (step A10): 1 same / 0 different / median-imputed if either race missing | transparent version of #21; kept alongside it |

### Group 5 — A's stated matching attitudes (23–25): "how much do demographics matter to A?"

A-side only, 1–10 signup scales, standardized. Like Group 2, constant within
a user — calibration, not ranking.

| # | Feature | Survey wording |
|---|---------|----------------|
| 23 | `imprace` | *"How important is it to you (1–10) that a person you date be of the same racial/ethnic background?"* |
| 24 | `imprelig` | *"How important is it to you (1–10) that a person you date be of the same religious background?"* |
| 25 | `exphappy` | *"Overall, on a scale of 1-10, how happy do you expect to be with the people you meet"* — event optimism |

### Group 6 — A's dating activity (26–27): ⚠ reverse-coded

*"In general, how frequently do you go on dates?"* / *"How often do you go
out (not necessarily on dates)?"* — **1 = several times a week … 7 = almost
never**. After standardization, a **positive value means LESS social
activity** than the training average. Keep this in mind when reading
coefficients and cards (a positive coefficient on `date` means *less*
frequent daters say yes more).

| # | Feature | Meaning after standardization |
|---|---------|-------------------------------|
| 26 | `date` | + = goes on dates more rarely than average |
| 27 | `go_out` | + = goes out (socially) more rarely than average |

### Group 7 — Familiarity (28–29)

Scorecard question *"Have you met this person before?"*, asked to each side;
two different "no" codes used across waves, recoded in step A7 (1→1, {0,2}→0);
binary passthrough, median-imputed (≈2% missing → fills with 0 = "no").

| # | Feature | Meaning |
|---|---------|---------|
| 28 | `met_before` | 1 = **A** says they had met B before the event |
| 29 | `met_before_o` | 1 = **B** says they had met A before (B's perception may differ from A's) |

### What is deliberately NOT in Stage 1

Mid-event ratings (`attr…prob`, `like`) — collected *during* the date, so
using them would break the cold-start premise; B's preference weights
(`*_1_1_norm_B`) — that's the B→A direction, which enters at Stage 4;
categorical codes (race/field/career/goal) — deferred to later stages to keep
Stage 1 fully numeric and card-friendly; raw `age`/`age_o` — only the gap
matters here; A's own self-ratings — A knows themselves, they don't react to
themselves (they return as kNN dimensions in Stage 2).

## 5. Known divergences from teammates' pipeline (for equal-footing reports)

| Step | Ours | Theirs |
|---|---|---|
| A2 wave 12 | dropped (canonical) / kept (variant) | always kept |
| A9 preference scaling | sum-to-1 normalized | raw values (mixed scales) |
| C1 imputation | per-split, train-only medians | gender-conditional, fit on full data pre-split |
| Row filters | A3 only | also drops >30%-missing participants and missing-B-profile rows (8,210 rows) |

## 6. Provenance fingerprints

Every run record stores SHA-256 (first 12 hex) of `speed_dating_clean.parquet`
and `Speed Dating Data.csv` plus row/wave/label-rate fingerprints
(`df_fingerprint`). Current values print in foundation Cell 0-3:
drop = 7,976 rows / 20 waves / dec 0.4240 / match 0.1678;
keep = 8,368 rows / 21 waves / dec 0.4201 / match 0.1649.
