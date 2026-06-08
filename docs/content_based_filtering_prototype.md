# Content-Based Filtering — Baseline Prototype Notebook
## Instruction for Cursor Agent

---

## Context and Objective

This is a **recommender system** problem, not a standard classification problem.
The distinction is critical:

- In standard classification, the output is a class label (yes/no)
- In a recommender system, the output must be a **continuous preference score**
  per directed pair (A→B) so that wave-mates can be **ranked** by predicted
  compatibility

The preference score is the probability that A says yes to B:
```
score(A→B) = P(dec=1 | features of A and B)
```

This score is used in two ways:
1. **Unilateral ranking** — rank B by score(A→B) descending
2. **Reciprocal re-ranking** — combine score(A→B) and score(B→A) via
   geometric mean to surface mutual matches

The final evaluation target is `match` (mutual match = both said yes),
not `dec`. Models are trained on `dec` but evaluated on `match`.

---

## Input File

```
two_sided_pair_features.csv
```

Columns:
- `iid` — participant A's ID
- `pid` — participant B's ID
- `wave` — wave number
- `dec` — A's decision toward B (0/1), training target
- `match` — mutual match outcome (0/1), evaluation target
- `[feature]_A` — all pre-survey features of A
- `[feature]_B` — all pre-survey features of B

---

## Cross-Validation Setup

Use the following fixed 3-fold wave assignments. This respects the
wave-isolated structure of the dataset — participants in different
waves never interact, so wave-based splitting prevents data leakage.

```python
FOLDS = {
    1: [1, 4, 5, 17, 19, 20, 21],
    2: [3, 6, 7, 11, 13, 15, 18],
    3: [2, 8, 9, 10, 14, 16]
}
```

For each fold:
- **Test set** — rows where `wave` is in that fold's wave list
- **Training set** — all remaining rows

Average all metrics across three folds.

---

## Models to Implement

Implement three models. All use default hyperparameters — this is a
baseline notebook. No tuning yet.

### Model 1 — Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

### Model 2 — LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
```

LightGBM handles NaN natively — do not impute before passing to LightGBM.

### Model 3 — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

# Random Forest probabilities are poorly calibrated by default.
# Wrap with CalibratedClassifierCV before using scores for ranking.
model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
```

---

## Feature Preparation

### Missing value handling

- For Logistic Regression and Random Forest: impute NaN with column mean
  computed from training rows only. Apply same imputation to test rows.
- For LightGBM: pass features as-is. LightGBM handles NaN internally.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

### Scaling

Apply StandardScaler for Logistic Regression only.
LightGBM and Random Forest do not require scaling.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
```

### Feature columns

Use all columns ending in `_A` and `_B` as features.
Exclude `iid`, `pid`, `wave`, `dec`, `match` from feature matrix.

```python
feature_cols = [c for c in df.columns
                if c.endswith('_A') or c.endswith('_B')]
```

---

## Intermediary Evaluation — Predicting dec

For each model, after training, evaluate on test fold rows using `dec`
as ground truth:

```python
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score
)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = model.predict(X_test)

auc      = roc_auc_score(y_test_dec, y_pred_proba)
logloss  = log_loss(y_test_dec, y_pred_proba)
accuracy = accuracy_score(y_test_dec, y_pred_class)
```

Print per-fold and average across folds.

---

## Generate Preference Scores

For every directed pair (A→B) in the test fold, store:

```python
scores_df = pd.DataFrame({
    'iid':      test_df['iid'],
    'pid':      test_df['pid'],
    'wave':     test_df['wave'],
    'score':    y_pred_proba,
    'match':    test_df['match']
})
```

---

## Unilateral Ranking

For each participant A in the test fold:
- Filter rows where `iid == A`
- Sort by `score` descending
- Store as ordered list of `pid` values

```python
def get_unilateral_ranking(scores_df):
    rankings = {}
    for iid, group in scores_df.groupby('iid'):
        ranked_pids = group.sort_values('score', ascending=False)['pid'].tolist()
        rankings[iid] = ranked_pids
    return rankings
```

---

## Reciprocal Re-ranking

For each undirected pair (A, B), combine both directed scores:

```python
import numpy as np

def get_reciprocal_ranking(scores_df):
    # Build lookup: (iid, pid) -> score
    score_lookup = dict(zip(
        zip(scores_df['iid'], scores_df['pid']),
        scores_df['score']
    ))

    rankings = {}
    for iid, group in scores_df.groupby('iid'):
        mutual_scores = []
        for _, row in group.iterrows():
            pid = row['pid']
            score_ab = row['score']
            score_ba = score_lookup.get((pid, iid), None)

            if score_ba is None:
                # Missing reverse direction — skip this pair with warning
                print(f"Warning: missing reverse score for ({pid}, {iid})")
                continue

            mutual = np.sqrt(score_ab * score_ba)
            mutual_scores.append((pid, mutual))

        ranked_pids = [pid for pid, _ in
                       sorted(mutual_scores, key=lambda x: x[1], reverse=True)]
        rankings[iid] = ranked_pids

    return rankings
```

---

## Evaluation Metrics

Implement the following functions. Both use `match` as ground truth,
not `dec`.

### mutual-match@k

```python
def mutual_match_at_k(ranked_list, matches, k):
    """
    ranked_list : ordered list of pid values
    matches     : set of pids who are real mutual matches with A
    k           : cutoff
    Returns None if participant has zero matches (excluded from average)
    """
    if len(matches) == 0:
        return None
    top_k = ranked_list[:k]
    hits = sum(1 for pid in top_k if pid in matches)
    return hits / min(k, len(matches))
```

### NDCG@k

```python
def ndcg_at_k(ranked_list, matches, k):
    """
    ranked_list : ordered list of pid values
    matches     : set of pids who are real mutual matches with A
    k           : cutoff
    Returns None if participant has zero matches (excluded from average)
    """
    if len(matches) == 0:
        return None
    top_k = ranked_list[:k]
    dcg = sum(
        1 / np.log2(i + 2)
        for i, pid in enumerate(top_k)
        if pid in matches
    )
    ideal_hits = min(k, len(matches))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return None
    return dcg / idcg
```

### Compute all metrics for one ranking

```python
def evaluate_ranking(rankings, scores_df, k_values=[3, 5]):
    results = {f'MM@{k}': [] for k in k_values}
    results.update({f'NDCG@{k}': [] for k in k_values})
    excluded = 0
    total = 0

    for iid, ranked_list in rankings.items():
        matches = set(
            scores_df[(scores_df['iid'] == iid) &
                      (scores_df['match'] == 1)]['pid']
        )
        total += 1
        if len(matches) == 0:
            excluded += 1
            continue

        for k in k_values:
            mm = mutual_match_at_k(ranked_list, matches, k)
            ndcg = ndcg_at_k(ranked_list, matches, k)
            if mm is not None:
                results[f'MM@{k}'].append(mm)
            if ndcg is not None:
                results[f'NDCG@{k}'].append(ndcg)

    averages = {key: np.mean(vals) for key, vals in results.items()}
    averages['excluded'] = excluded
    averages['total'] = total
    return averages
```

---

## Sanity Checks

Run these before evaluating each fold:

```python
# No wave overlap between train and test
assert len(set(train_waves) & set(test_waves)) == 0, \
    "Data leakage: test waves in training set"

# Both directions exist for every test pair
test_pairs = set(zip(test_df['iid'], test_df['pid']))
for (a, b) in test_pairs:
    assert (b, a) in test_pairs, \
        f"Missing reverse pair ({b}, {a})"

# Scores are valid probabilities
assert scores_df['score'].between(0, 1).all(), \
    "Scores outside [0, 1]"

# No NaN in features passed to LR and RF
assert not np.isnan(X_train_imputed).any(), "NaN in training features"
assert not np.isnan(X_test_imputed).any(), "NaN in test features"

# Print class distributions
print(f"dec=1 rate in train: {y_train.mean():.3f}")
print(f"match=1 rate in test: {test_df['match'].mean():.3f}")
print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")
```

---

## Final Results Table

Print the following summary averaged across three folds:

```
=== Intermediary Metrics (predicting dec) ===

Model                 Accuracy    AUC-ROC    Log-loss
Logistic Regression   x.xxx       x.xxx      x.xxx
LightGBM              x.xxx       x.xxx      x.xxx
Random Forest         x.xxx       x.xxx      x.xxx

=== Final System Metrics (predicting match) ===
Note: participants with zero matches excluded from averages
      Report excluded count per fold

Model                 Condition     MM@3    MM@5    NDCG@3    NDCG@5
Logistic Regression   Unilateral    x.xxx   x.xxx   x.xxx     x.xxx
Logistic Regression   Reciprocal    x.xxx   x.xxx   x.xxx     x.xxx
LightGBM              Unilateral    x.xxx   x.xxx   x.xxx     x.xxx
LightGBM              Reciprocal    x.xxx   x.xxx   x.xxx     x.xxx
Random Forest         Unilateral    x.xxx   x.xxx   x.xxx     x.xxx
Random Forest         Reciprocal    x.xxx   x.xxx   x.xxx     x.xxx
```

Also print per-fold breakdown for each model so variance is visible.

---

## Notebook Structure

Organise the notebook with the following clearly labelled sections:

```
1. Imports and Setup
2. Load Data
3. Cross-Validation Fold Definitions
4. Feature Preparation Functions
5. Model Definitions
6. Evaluation Metric Functions
7. Main Loop — iterate over 3 folds
   7a. Split train/test by wave
   7b. Prepare features
   7c. Train each model
   7d. Compute intermediary metrics (dec)
   7e. Generate preference scores
   7f. Unilateral ranking and evaluation
   7g. Reciprocal re-ranking and evaluation
8. Aggregate and Print Final Results Table
```

---

## Dependencies

```
pandas
numpy
scikit-learn
lightgbm
```

Install LightGBM if not already available:
```bash
pip install lightgbm
```

---

## Important Notes for Cursor

- This is a **baseline prototype** — use default hyperparameters only.
  No grid search, no tuning. Tuning comes in a later notebook.
- Feature engineering (interaction features, difference features) also
  comes later. Use raw _A and _B columns as-is for now.
- Do not modify the input CSV files.
- The notebook should run end to end without errors before any
  results are interpreted.
