# Build Two-Sided Pair Feature Matrix
## Instruction for Cursor Agent

---

## Context

You have a preprocessed file `one_sided_participant_features.csv` where each row
is one directed interaction with participant `iid`'s pre-survey features attached.

You need to construct a two-sided feature matrix where each row represents a
directed pair (A→B) containing features from both A and B side by side.

Participants excluded during one-sided preprocessing (>30% missing KEEP features)
are not present in the participant profile table. Any interaction whose partner
`pid` is excluded will have missing `_B` columns after the join. **Drop those
rows** — a content-based pairwise model requires complete features on both sides.

---

## Input Files

```
data/one_sided_participant_features.csv   — interactions with iid's preprocessed features
```

Participant profiles are derived by deduplicating `one_sided_participant_features.csv`
on `iid`.

---

## Steps

### Step 1 — Load one-sided file

```python
import pandas as pd

one_sided = pd.read_csv('data/one_sided_participant_features.csv')
participants = one_sided.drop_duplicates('iid')
interactions = one_sided[['iid', 'pid', 'wave', 'dec', 'match']]
```

---

### Step 2 — Identify feature columns

```python
feature_cols = [c for c in participants.columns
                if c not in {'iid', 'id', 'pid', 'wave', 'dec', 'match'}]
```

---

### Step 3 — Join A's features using iid

```python
df = interactions.merge(
    participants[['iid', *feature_cols]].rename(
        columns={c: f'{c}_A' for c in feature_cols}
    ),
    on='iid',
    how='left'
)
```

---

### Step 4 — Join B's features using pid

```python
df = df.merge(
    participants[['iid', *feature_cols]].rename(
        columns={c: f'{c}_B' for c in feature_cols}
    ),
    left_on='pid',
    right_on='iid',
    how='left',
    suffixes=('', '_drop')
)

# Drop duplicate iid column from B's side
df = df.drop(columns=[c for c in df.columns if c.endswith('_drop')])
```

---

### Step 5 — Sanity checks (after join, before row drop)

```python
# Row count must match interactions
assert len(df) == len(interactions), \
    f"Row count mismatch: {len(df)} vs {len(interactions)}"

# A side should be complete (one-sided data already excludes sparse iids)
a_cols = [c for c in df.columns if c.endswith('_A')]
missing_a = df[a_cols].isna().sum()
assert not missing_a.any(), f"Unexpected missing A features:\n{missing_a[missing_a > 0]}"

# B side may be incomplete when pid was excluded from one-sided preprocessing
b_cols = [c for c in df.columns if c.endswith('_B')]
incomplete_b = df[b_cols].isna().any(axis=1)
print(f"Rows with missing B features: {incomplete_b.sum()}")
```

---

### Step 6 — Drop rows with missing partner (B) features

```python
df = df.loc[~incomplete_b].reset_index(drop=True)
assert df[a_cols + b_cols].isna().sum().sum() == 0
```

**Why drop?** Excluded partners have near-empty surveys. Imputing 40+ `_B`
values would fabricate a generic profile and add noise. For content-based
pairwise modelling, both sides must have real preprocessed features.

Expected drop: **79 rows** (partners among the 7 excluded `iid`s), leaving
**8,210** usable directed pairs from an initial **8,289**.

---

### Step 7 — Save output

```python
df.to_csv('data/two_sided_pair_features.csv', index=False)
print(f"Saved {len(df)} rows with {len(df.columns)} columns")
```

Implemented in [`build_two_sided_data.py`](../build_two_sided_data.py):

```bash
python build_two_sided_data.py
```

---

## Expected Output Structure

Each row in `two_sided_pair_features.csv`:

```
iid | pid | wave | dec | match | [all features]_A | [all features]_B
```

- Columns ending in `_A` — A's pre-survey features
- Columns ending in `_B` — B's pre-survey features
- `dec` — A's decision toward B (0/1), training target
- `match` — mutual match outcome (0/1), evaluation target
- **No missing values** in any `_A` or `_B` feature column

---

## Important Notes

- Join uses `iid` for A's side and `pid` for B's side
- `pid` is the partner's participant ID in this dataset
- Rows where `pid` points to an excluded participant are **dropped** after the join
- This file is the input to all subsequent content-based modelling steps
