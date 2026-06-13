# Speed Dating Cleaned Dataset — Data Dictionary

**Source:** Columbia Business School Speed Dating Experiment (Fisman & Iyengar, 2002–2004)  
**Cleaned by:** CS608 group project (EDA + cleaning notebook)  
**Rows:** 7,976 directional interactions · **Participants:** 523 · **Waves:** 20 (Wave 12 dropped)

## How to read this file

Each **row is one person's evaluation of one partner** during one 4-minute date (A→B). The same pair appears twice (A→B and B→A).

Column roles are defined in `column_roles.json`:

| Role | Use in modeling |
|------|-----------------|
| `id_outcome_cols` | IDs, wave metadata, targets (`dec`, `dec_o`, `match`) |
| `feature_cols` | Signup / pre-date inputs only — **use these as model features** |
| `non_feature_ratings` | Per-date outcomes — **never use as features** (label leakage); OK for popularity oracle / EDA |

**Imputation:** We did **not** impute missing values in this file. NaNs are real (missing survey answers, "no opinion" ratings, etc.). Imputation and within-gender z-scoring of self-ratings are **deferred to the modeling pipeline** and must be fit on the training fold only.

**Quick load:**

```python
import pandas as pd, json
df = pd.read_parquet("speed_dating_clean.parquet")  # or read_csv
roles = json.load(open("column_roles.json"))
FEATURE_COLS = roles["feature_cols"]
```

---

## Identifiers & outcomes (`id_outcome_cols`)

| Column | Description |
|--------|-------------|
| `iid` | Unique participant ID (person who rated) |
| `pid` | Partner's `iid` (person being rated) |
| `gender` | Rater's gender: 0 = female, 1 = male |
| `wave` | Event wave number (1–21, excluding 12) |
| `condtn` | 1 = limited choice, 2 = extensive choice |
| `round` | Number of people who met in that wave |
| `order` | Which date slot that night (1st date, 2nd date, …) |
| `position` | Station number where they met |
| `dec` | Rater's decision: 1 = yes, 0 = no |
| `dec_o` | Partner's decision toward rater: 1 = yes, 0 = no |
| `match` | 1 if **both** `dec=1` and `dec_o=1`; else 0 |

---

## Model features (`feature_cols`)

Use **`column_roles.json` → `feature_cols`** as the single source of truth. Highlights:

### Demographics & signup (person-level, repeated on each date row)

| Column | Description |
|--------|-------------|
| `age` | Rater's age |
| `age_o` | Partner's age |
| `race` | Rater's race code (1=Black, 2=White, 3=Latino, 4=Asian, 5=Native, 6=Other) |
| `race_o` | Partner's race (same codes) |
| `field_cd` | **Rater's field of study (coded 1–18).** Replaces free-text `field` (dropped). See codebook below. |
| `career_c` | Rater's career field (coded); replaces free-text `career` |
| `goal` | Reason for participating in speed dating (coded) |
| `date` | How many times per month rater goes on dates (1–3 scale) |
| `go_out` | How often rater goes out (1–7 scale) |
| `exphappy` | How happy rater expects to be from the event (1–10) |
| `imprace` | Importance of same race in dating (1–10) |
| `imprelig` | Importance of same religion in dating (1–10) |
| `sports` … `yoga` | Interest ratings (1–10) for 17 activities |

### `field_cd` codebook

| Code | Field |
|------|-------|
| 1 | Law |
| 2 | Math |
| 3 | Social Science / Psychology |
| 4 | Medical / Pharma / Biotech |
| 5 | Engineering |
| 6 | English / Creative Writing / Journalism |
| 7 | History / Religion / Philosophy |
| 8 | Business / Econ / Finance |
| 9 | Education / Academia |
| 10 | Biological Sciences / Chemistry / Physics |
| 11 | Social Work |
| 12 | Undergrad / undecided |
| 13 | Political Science / International Affairs |
| 14 | Film |
| 15 | Fine Arts / Arts Administration |
| 16 | Languages |
| 17 | Architecture |
| 18 | Other |

### Pair-level & derived

| Column | Description |
|--------|-------------|
| `int_corr` | Correlation between rater's and partner's interest vectors at signup |
| `samerace` | 1 if rater and partner same race (raw flag from dataset) |
| `met_before` | 1 = met before, 0 = did not, NaN = unknown (recoded from `met`) |
| `met_before_o` | Same for partner (from `met_o`) |
| `age_diff` | \|age − age_o\| |
| `race_match` | 1 = same race, 0 = different, NaN if either race missing |
| `scale_type` | `'1-10'` (waves 6–9) or `'100pt'` (other waves) — audit flag for preference scale |

### Self-ratings at signup (`attr3_1` … `amb3_1`, scale 1–10)

How the rater rates **themselves** on attractiveness, sincerity, intelligence, fun, ambition. Z-score within gender in modeling (not done here).

### Opposite-sex beliefs (`attr2_1` … `shar2_1`)

What the rater thinks the **average opposite-sex** person looks for (1–10).

### Normalized stated preferences (`attr1_1_norm` … `shar1_1_norm`)

Rater's signup importance weights converted to **proportions summing to 1** (fixes 100-pt vs 1–10 scale mix). **Use these in models, not raw `attr1_1`…`shar1_1`.**

| `_norm` suffix | Trait |
|----------------|-------|
| `attr` | Attractiveness |
| `sinc` | Sincerity |
| `intel` | Intelligence |
| `fun` | Fun |
| `amb` | Ambition |
| `shar` | Shared interests |

---

## Per-date outcomes — NOT features (`non_feature_ratings`)

Recorded **during** the date. Do not use as model inputs.

| Column | Description |
|--------|-------------|
| `attr`, `sinc`, `intel`, `fun`, `amb`, `shar` | Rater's 1–10 ratings of partner on six traits |
| `like` | Overall liking of partner (1–10) |
| `prob` | Estimated probability partner will say yes (1–10) |
| `*_o` mirrors | Same ratings/decisions from the **partner's** side toward the rater |

Zeros in raw ratings were converted to NaN ("did not rate"). Values above 10 were clipped to 10.

---

## Partner signup preferences (`pf_o_*`) — extra columns, not in `feature_cols`

These are the **partner's** stated preference weights from signup (partner's `attr1_1` … `shar1_1`), copied onto each row for convenience.

| Column | Meaning |
|--------|---------|
| `pf_o_att` | Partner's weight on **attractiveness** |
| `pf_o_sin` | Partner's weight on **sincerity** |
| `pf_o_int` | Partner's weight on **intelligence** |
| `pf_o_fun` | Partner's weight on **fun** |
| `pf_o_amb` | Partner's weight on **ambition** |
| `pf_o_sha` | Partner's weight on **shared interests** |

**Scale:** Raw values — 100-point allocation (most waves) or 1–10 (waves 6–9), depending on **partner's** wave. **Not normalized.** For modeling, prefer deriving partner-side normalized prefs yourself or use rater-side `*_norm` features only (already in `feature_cols`).

---

## Other columns kept in the file (not in `feature_cols`)

| Column | Description |
|--------|-------------|
| `met_o` | Raw partner "met before" code (use `met_before_o` instead) |
| `attr1_1` … `shar1_1` | Raw signup preference weights (use `*_norm` for features) |
| `attr1_2` … `amb3_2` | Day-after / follow-up preference blocks (sparse; not features) |
| `match_es` | Expected number of matches |
| `satis_2` | Satisfaction with event |
| `length` | Length of follow-up survey |
| `numdat_2` | Number of dates since event (follow-up) |

---

## Cleaning summary

| Step | What changed |
|------|----------------|
| Drop Wave 12 | Budget rule forced ≤50% yes — not comparable |
| Drop null `pid` | 10 phantom Wave-5 rows with no real partner |
| Ratings | 0 → NaN; clip >10 to 10 |
| Interests | clip >10 to 10 |
| `met` / `met_o` | Recoded to `met_before` / `met_before_o` (0/1) |
| Preferences | Added `scale_type` + `*_1_1_norm` proportions |
| Derived | `age_diff`, `race_match` |
| Dropped | ~85 sparse/redundant/free-text columns |

Full pipeline: `speed_dating_EDA_data_cleaning.ipynb` and `EDA_and_Cleaning_Walkthrough.md`.
