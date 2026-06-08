# Preprocessing and Initial Feature Selection

This document records how the Columbia speed dating dataset was inspected, which pre-survey columns were selected for content-based filtering, and how missing values were handled. The exploratory work lives in [`notebooks/eda.ipynb`](../notebooks/eda.ipynb); the reusable pipeline is implemented in [`load_one_sided_data.py`](../load_one_sided_data.py).

## Source data

| Item | Value |
|------|-------|
| File | `data/Speed Dating Data.csv` |
| Granularity | One row per speed date (participant `iid` met partner `pid` in `wave`) |
| Participants | 551 unique `iid` values |
| Interactions | ~8,368 rows (after dropping rows with missing `pid`) |

**Target columns (interaction-level):**

- `dec` — whether `iid` wants to see `pid` again (1 = yes)
- `match` — mutual match (1 only if both `dec(iid→pid)` and `dec(pid→iid)` are yes)

**Identification columns retained in the modeling table:**

- `iid` — participant identifier (survey owner for one-sided features)
- `id` — same as `iid` in this dataset
- `wave` — event wave (1–21)
- `pid` — partner identifier for that date

## EDA summary

### Match rates (participant level)

- Mean mutual matches per participant: **2.50** (median **2**)
- Distribution of match counts is **right-skewed** (skew ≈ 1.25; mean > median; long tail toward high counts)
- **18%** of participants had zero mutual matches

### Feature screening criteria

Candidate pre-survey columns were grouped into demographics, interests, expectations, preference blocks, and self-rating blocks. Each column was evaluated on:

1. **Missingness** — drop ≥ 30%, review 10–30%, keep < 10%
2. **Cardinality** — prefer coded fields over free text
3. **Variance** — flag low-spread 1–10 scales
4. **Redundancy** — drop duplicate survey blocks correlated with a kept block

## Columns dropped

| Column(s) | Reason |
|-----------|--------|
| `undergra` | 43% missing; high-cardinality free text |
| `mn_sat` | 63% missing |
| `tuition` | 58% missing |
| `income` | 49% missing |
| `expnum` | 77% missing |
| `attr5_1` … `amb5_1` | 43% missing; highly correlated with `*_3_1` self-rating block (e.g. attr r ≈ 0.84) |
| `field` | Use `field_cd` instead (18 levels vs 259 text values) |
| `career` | Use `career_c` instead (17 levels vs 367 text values) |
| `from`, `zipcode` | High-cardinality location text; weak direct signal for content-based similarity |

## Columns under review (not used in v1)

| Column(s) | Reason |
|-----------|--------|
| `attr4_1` … `shar4_1` | ~24% missing; moderate overlap with preference block 1 |

## Columns kept (45 model features)

**Core content-based attributes (prototype):**

- Preference weights: `attr1_1`, `sinc1_1`, `intel1_1`, `fun1_1`, `amb1_1`
- Self-profile: `attr3_1`, `sinc3_1`, `intel3_1`, `fun3_1`, `amb3_1`

**Additional preference block:**

- `attr2_1`, `sinc2_1`, `intel2_1`, `fun2_1`, `amb2_1`, `shar2_1`, `shar1_1`

**Demographics and coded fields:**

- `age`, `race`, `field_cd`, `career_c`, `goal`, `date`, `go_out`, `imprace`, `imprelig`, `exphappy`, `gender`

**Interest vector (1–10 scales):**

- `sports`, `tvsports`, `exercise`, `dining`, `museums`, `art`, `hiking`, `gaming`, `clubbing`, `reading`, `tv`, `theater`, `movies`, `concerts`, `music`, `shopping`, `yoga`

## Missing-value handling

Only **19 of 551** participants had any missing KEEP values. Strategy:

| Step | Rule | Outcome |
|------|------|---------|
| 1. Exclude | Drop participants with **> 30%** missing KEEP features | **7** participants removed (`iid`: 28, 58, 59, 136, 339, 340, 346) — near-empty surveys |
| 2. Impute | Gender-stratified **median** for numeric/ordinal scales | Applied to remaining sparse gaps |
| 2. Impute | Gender-stratified **mode** for coded categoricals (`race`, `field_cd`, `career_c`) | Applied to remaining sparse gaps |
| — | `gender` | 0% missing; no imputation |
| Fallback | Global median/mode when a gender group has no observed values | Ensures complete profiles |

**Result:** **544** usable participant profiles; **28** cell-level imputations across 12 lightly incomplete participants; **0** remaining missing values in model features.

Gender-stratified imputation is appropriate because preferences differ by gender (e.g. `attr1_1` median **15** for men vs **23** for women) and matching is evaluated within opposite-gender pools.

## Output artifact

The script [`load_one_sided_data.py`](../load_one_sided_data.py) builds and loads:

**`data/one_sided_participant_features.csv`**

Each row is one interaction. Columns are ordered as:

1. **IDs:** `iid`, `id`, `wave`, `pid`
2. **Features:** 45 preprocessed KEEP columns (including `gender`)
3. **Targets:** `dec`, `match`

Survey features are **one-sided**: they describe the participant `iid` only (repeated on each of their date rows). Rows involving excluded participants are removed.

### Usage

```bash
# Build the CSV from the raw speed dating file
python load_one_sided_data.py

# Or from Python
from load_one_sided_data import build_one_sided_dataset, load_one_sided_dataset

build_one_sided_dataset()  # writes data/one_sided_participant_features.csv
df = load_one_sided_dataset()
```

## Two-sided pair features

The script [`build_two_sided_data.py`](../build_two_sided_data.py) builds:

**`data/two_sided_pair_features.csv`**

Each row is a directed pair (A→B) with A's and B's preprocessed features (`*_A` / `*_B` columns).

After joining B's profile on `pid`, **79 rows** are dropped where the partner was one of the 7 excluded participants (missing `_B` features). The final file has **8,210** complete pairs with no missing feature values. See [`docs/build_two_sided_features.md`](build_two_sided_features.md) for details.

```bash
python build_two_sided_data.py
```

## References

- EDA notebook: [`notebooks/eda.ipynb`](../notebooks/eda.ipynb)
- Content-based prototype: [`CS608_Project2_ContentBased_Prototype.ipynb`](../CS608_Project2_ContentBased_Prototype.ipynb)
- Two-sided build: [`docs/build_two_sided_features.md`](build_two_sided_features.md)
