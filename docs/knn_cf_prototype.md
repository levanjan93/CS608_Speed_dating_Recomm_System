# Component 2 — KNN Neighbourhood Model: Baseline Prototype Notebook
## Instruction for Cursor Agent

---

## Context and Objective

This component is a **hybrid neighbourhood model** with two distinct steps:

- **Content-based step** — find neighbours of A from training participants
  using cosine similarity on pre-survey profile features
- **Collaborative step** — aggregate the actual yes/no decisions (`dec`)
  those neighbours made toward participants similar to B

The output is a preference score `score_cf(A→B)` per directed pair.
This score is used identically to Component 1 — for unilateral ranking
and reciprocal re-ranking, evaluated against `match` as ground truth.

The key distinction from Component 1:
- Component 1 scores pairs **directly** from features via a trained model
- Component 2 scores pairs **indirectly** through the aggregated behaviour
  of similar training participants — no model is trained

---

## Input File

```
two_sided_pair_features.csv
```

Columns used:
- `iid` — participant A's ID
- `pid` — participant B's ID
- `wave` — wave number
- `dec` — A's decision toward B (0/1), used for aggregation and intermediary eval
- `match` — mutual match outcome (0/1), final evaluation target
- `[feature]_A` — all pre-survey features of A
- `[feature]_B` — all pre-survey features of B

---

## Hyperparameters

Define these as constants at the top of the notebook.
Do not tune in this prototype — use defaults only.

```python
K_NEIGHBOURS    = 10      # number of similar training participants to find for A
K_PARTNERS      = 10      # number of similar training partners to find for B
FALLBACK_SCORE  = 0.0     # score to assign when no signal is available
                          # (no neighbour of A ever dated anyone similar to B)
                          # set as hyperparameter for future tuning
```

---

## Cross-Validation Setup

Same fixed 3-fold wave assignments as Component 1:

```python
FOLDS = {
    1: [1, 4, 5, 17, 19, 20, 21],
    2: [3, 6, 7, 11, 13, 15, 18],
    3: [2, 8, 9, 10, 14, 16]
}
```

For each fold: test on that fold's waves, train on all remaining waves.

---

## Step 1 — Build Participant Profile Table

Build a single unified profile table from all participants appearing
in training rows — regardless of whether they appear as A (iid) or
B (pid). This avoids any gap caused by dropped rows during preprocessing.

```python
profile_cols = [c for c in df.columns if c.endswith('_A')]
partner_cols = [c for c in df.columns if c.endswith('_B')]

# Profiles from A side — indexed by iid
profiles_from_A = (
    train_df.groupby('iid')[profile_cols]
    .first()
    .rename(columns={c: c.replace('_A', '') for c in profile_cols})
)

# Profiles from B side — indexed by pid
profiles_from_B = (
    train_df.groupby('pid')[partner_cols]
    .first()
    .rename(columns={c: c.replace('_B', '') for c in partner_cols})
)

# Merge into one unified profile table
# If a participant appears on both sides, A-side takes priority
train_profiles = profiles_from_B.combine_first(profiles_from_A)
```

Do the same for test participants:

```python
# Test participant profiles (as A)
test_profiles = (
    test_df.groupby('iid')[profile_cols]
    .first()
    .rename(columns={c: c.replace('_A', '') for c in profile_cols})
)

# Test partner profiles (as B)
test_partner_profiles = (
    test_df.groupby('pid')[partner_cols]
    .first()
    .rename(columns={c: c.replace('_B', '') for c in partner_cols})
)

# Unified test profiles
test_profiles_all = test_partner_profiles.combine_first(test_profiles)
```

---

## Step 2 — Handle Missing Values

Impute NaN with column mean computed from training profiles only.
Apply the same fitted imputer to test profiles.

```python
from sklearn.impute import SimpleImputer
import pandas as pd

feature_cols = train_profiles.columns.tolist()

imputer = SimpleImputer(strategy='mean')
train_profiles_imputed = pd.DataFrame(
    imputer.fit_transform(train_profiles),
    index=train_profiles.index,
    columns=feature_cols
)

test_profiles_imputed = pd.DataFrame(
    imputer.transform(test_profiles_all),
    index=test_profiles_all.index,
    columns=feature_cols
)
```

---

## Step 3 — Find Neighbours of A

For each test participant A, find their K_NEIGHBOURS most similar
training participants using cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_neighbours(A_profile, train_profiles_imputed, k):
    """
    A_profile             : 1D numpy array of A's profile features
    train_profiles_imputed: dataframe of training participant profiles
    k                     : number of neighbours to return

    Returns: list of (iid, similarity) tuples sorted by similarity descending
    """
    sims = cosine_similarity(
        A_profile.reshape(1, -1),
        train_profiles_imputed.values
    )[0]

    top_k_indices = np.argsort(sims)[::-1][:k]
    return [
        (train_profiles_imputed.index[i], sims[i])
        for i in top_k_indices
    ]
```

---

## Step 4 — Find Training Partners Similar to B

For each test partner B, find their K_PARTNERS most similar training
participants using cosine similarity on the same profile features:

```python
def find_similar_partners(B_profile, train_profiles_imputed, k):
    """
    B_profile             : 1D numpy array of B's profile features
    train_profiles_imputed: dataframe of training participant profiles
    k                     : number of similar partners to return

    Returns: list of (pid, similarity) tuples sorted by similarity descending
    """
    sims = cosine_similarity(
        B_profile.reshape(1, -1),
        train_profiles_imputed.values
    )[0]

    top_k_indices = np.argsort(sims)[::-1][:k]
    return [
        (train_profiles_imputed.index[i], sims[i])
        for i in top_k_indices
    ]
```

Note: both neighbour search and partner search use the same
`train_profiles_imputed` table and the same cosine similarity function.
The difference is only in which profile is used as the query vector.

---

## Step 5 — Precompute Decision Lookup

Before scoring, build a fast lookup of all training decisions to avoid
repeated dataframe filtering inside the scoring loop:

```python
# Build dict: {(iid, pid): dec} for all training interactions
train_decision_lookup = dict(
    zip(
        zip(train_df['iid'], train_df['pid']),
        train_df['dec']
    )
)
```

---

## Step 6 — Compute CF Score

For each directed test pair (A→B), aggregate neighbour decisions
toward partners similar to B:

```python
def compute_cf_score(A_neighbours, B_similar_partners,
                     train_decision_lookup, fallback_score):
    """
    A_neighbours          : list of (neighbour_iid, sim_A_C) tuples
    B_similar_partners    : list of (partner_pid, sim_B_Bprime) tuples
    train_decision_lookup : dict of {(iid, pid): dec}
    fallback_score        : score to return when no signal is available

    For each neighbour C of A:
        For each training partner B' similar to B:
            If C dated B' in training, get dec(C→B')
            Weighted contribution = sim(A,C) × sim(B,B') × dec(C→B')

    score_cf(A→B) = sum of weighted contributions / sum of weights
    Returns fallback_score if no signal found (denominator == 0)
    """
    numerator   = 0.0
    denominator = 0.0

    for (neighbour_iid, sim_ac) in A_neighbours:
        for (partner_pid, sim_bb) in B_similar_partners:
            dec_val = train_decision_lookup.get((neighbour_iid, partner_pid))
            if dec_val is not None:
                weight       = sim_ac * sim_bb
                numerator   += weight * dec_val
                denominator += weight

    if denominator == 0:
        return fallback_score

    return numerator / denominator
```

---

## Step 7 — Generate All Scores for Test Fold

For every directed pair (A→B) in the test fold, compute the CF score.
Use tqdm to monitor progress since this loop can be slow.

```python
from tqdm import tqdm

scores = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="Scoring test pairs"):
    iid = row['iid']
    pid = row['pid']

    # Get profile vectors
    A_profile = test_profiles_imputed.loc[iid].values
    B_profile = test_profiles_imputed.loc[pid].values

    # Find neighbours and similar partners
    A_neighbours       = find_neighbours(
        A_profile, train_profiles_imputed, K_NEIGHBOURS
    )
    B_similar_partners = find_similar_partners(
        B_profile, train_profiles_imputed, K_PARTNERS
    )

    # Compute score
    cf_score = compute_cf_score(
        A_neighbours,
        B_similar_partners,
        train_decision_lookup,
        FALLBACK_SCORE
    )

    scores.append({
        'iid':      iid,
        'pid':      pid,
        'wave':     row['wave'],
        'score_cf': cf_score,
        'match':    row['match'],
        'dec':      row['dec']
    })

scores_df = pd.DataFrame(scores)
```

After scoring, print:
```python
# Score distribution diagnostics
print(f"Score min:     {scores_df['score_cf'].min():.4f}")
print(f"Score max:     {scores_df['score_cf'].max():.4f}")
print(f"Score mean:    {scores_df['score_cf'].mean():.4f}")
print(f"Score std:     {scores_df['score_cf'].std():.4f}")

# How many pairs fell back to fallback score
fallback_count = (scores_df['score_cf'] == FALLBACK_SCORE).sum()
print(f"Pairs using fallback score: {fallback_count} / {len(scores_df)}")
```

The fallback count and score distribution will inform whether
FALLBACK_SCORE needs to be adjusted in future tuning.

---

## Step 8 — Intermediary Evaluation on dec

Since this model is non-parametric (no trained classifier), compute
AUC only as the intermediary metric:

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(scores_df['dec'], scores_df['score_cf'])
print(f"CF AUC on dec: {auc:.4f}")
```

If AUC < 0.5, flip all scores and note it in output:
```python
if auc < 0.5:
    print("AUC below 0.5 — flipping scores")
    scores_df['score_cf'] = 1 - scores_df['score_cf']
    auc = 1 - auc
    print(f"Flipped AUC: {auc:.4f}")
```

---

## Step 9 — Unilateral Ranking

For each participant A in the test fold, rank all wave-mates by
`score_cf` descending:

```python
def get_unilateral_ranking(scores_df):
    rankings = {}
    for iid, group in scores_df.groupby('iid'):
        ranked_pids = (
            group.sort_values('score_cf', ascending=False)['pid'].tolist()
        )
        rankings[iid] = ranked_pids
    return rankings
```

---

## Step 10 — Reciprocal Re-ranking

Combine both directed scores using geometric mean:

```python
def get_reciprocal_ranking(scores_df):
    score_lookup = dict(
        zip(
            zip(scores_df['iid'], scores_df['pid']),
            scores_df['score_cf']
        )
    )

    rankings = {}
    for iid, group in scores_df.groupby('iid'):
        mutual_scores = []
        for _, row in group.iterrows():
            pid      = row['pid']
            score_ab = row['score_cf']
            score_ba = score_lookup.get((pid, iid))

            if score_ba is None:
                print(f"Warning: missing reverse score for ({pid}, {iid})")
                continue

            mutual = np.sqrt(score_ab * score_ba)
            mutual_scores.append((pid, mutual))

        ranked_pids = [
            pid for pid, _ in
            sorted(mutual_scores, key=lambda x: x[1], reverse=True)
        ]
        rankings[iid] = ranked_pids

    return rankings
```

---

## Step 11 — Evaluation Metrics

Use the same metric functions as Component 1:

```python
def mutual_match_at_k(ranked_list, matches, k):
    if len(matches) == 0:
        return None
    top_k = ranked_list[:k]
    hits  = sum(1 for pid in top_k if pid in matches)
    return hits / min(k, len(matches))


def ndcg_at_k(ranked_list, matches, k):
    if len(matches) == 0:
        return None
    top_k = ranked_list[:k]
    dcg   = sum(
        1 / np.log2(i + 2)
        for i, pid in enumerate(top_k)
        if pid in matches
    )
    ideal_hits = min(k, len(matches))
    idcg       = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_ranking(rankings, scores_df, k_values=[3, 5]):
    results  = {f'MM@{k}': []    for k in k_values}
    results.update({f'NDCG@{k}': [] for k in k_values})
    excluded = 0
    total    = 0

    for iid, ranked_list in rankings.items():
        matches = set(
            scores_df[
                (scores_df['iid'] == iid) &
                (scores_df['match'] == 1)
            ]['pid']
        )
        total += 1
        if len(matches) == 0:
            excluded += 1
            continue

        for k in k_values:
            mm   = mutual_match_at_k(ranked_list, matches, k)
            ndcg = ndcg_at_k(ranked_list, matches, k)
            if mm   is not None: results[f'MM@{k}'].append(mm)
            if ndcg is not None: results[f'NDCG@{k}'].append(ndcg)

    averages = {key: np.mean(vals) for key, vals in results.items()}
    averages['excluded'] = excluded
    averages['total']    = total
    return averages
```

---

## Step 12 — Sanity Checks

Run before evaluating each fold:

```python
# No wave overlap between train and test
assert len(set(train_waves) & set(test_waves)) == 0, \
    "Data leakage: test waves found in training"

# Both directions exist for every test pair
test_pairs = set(zip(test_df['iid'], test_df['pid']))
for (a, b) in test_pairs:
    assert (b, a) in test_pairs, \
        f"Missing reverse pair ({b}, {a})"

# Score range check
assert scores_df['score_cf'].between(0, 1).all(), \
    "Scores outside [0, 1]"

# Print class distributions
print(f"dec=1 rate in train:  {train_df['dec'].mean():.3f}")
print(f"match=1 rate in test: {test_df['match'].mean():.3f}")
print(f"Training rows:        {len(train_df)}")
print(f"Test rows:            {len(test_df)}")
print(f"Training participants: {train_profiles_imputed.shape[0]}")
print(f"Test participants:     {test_profiles_imputed.shape[0]}")
```

---

## Step 13 — Results Table

Print the following summary averaged across three folds:

```
=== Hyperparameters ===
K_NEIGHBOURS:   10
K_PARTNERS:     10
FALLBACK_SCORE: 0.0

=== Score Distribution (average across folds) ===
Min:  x.xxxx
Max:  x.xxxx
Mean: x.xxxx
Std:  x.xxxx
Pairs using fallback score: X / Y (x.x%)

=== Intermediary Metric (predicting dec) ===
KNN CF    AUC: x.xxx

=== Final System Metrics (predicting match) ===
Participants with zero matches excluded: X of Y per fold (average)

Model     Condition     MM@3    MM@5    NDCG@3    NDCG@5
KNN CF    Unilateral    x.xxx   x.xxx   x.xxx     x.xxx
KNN CF    Reciprocal    x.xxx   x.xxx   x.xxx     x.xxx

=== Per-Fold Breakdown ===
[Print each fold's results separately so variance is visible]
```

---

## Notebook Structure

Organise the notebook with the following clearly labelled sections:

```
1.  Imports and Setup
2.  Hyperparameter Constants
3.  Load Data
4.  Cross-Validation Fold Definitions
5.  Profile Building Functions
6.  Missing Value Imputation
7.  Neighbour and Partner Search Functions
8.  Decision Lookup Builder
9.  CF Score Function
10. Scoring Loop (with tqdm)
11. Score Distribution Diagnostics
12. Intermediary Evaluation (AUC on dec)
13. Unilateral Ranking
14. Reciprocal Re-ranking
15. Evaluation Metric Functions
16. Main Loop — iterate over 3 folds
17. Aggregate and Print Final Results Table
```

---

## Dependencies

```
pandas
numpy
scikit-learn
tqdm
```

---

## Important Notes for Cursor

- This is a **baseline prototype** — use default hyperparameters only.
  K_NEIGHBOURS, K_PARTNERS, and FALLBACK_SCORE will be tuned later.
- No model is trained in this component — it is entirely non-parametric.
- The score distribution printout and fallback count are important
  diagnostics for deciding how to tune FALLBACK_SCORE later.
- Do not modify the input CSV file.
- The notebook should run end to end without errors before
  any results are interpreted.
