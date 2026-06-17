# Stage 1 — Content-Based Model (`score_cb`): LR baseline vs EBM contender

**Scope:** this document contains *only* Stage 1. It defines the two
candidate models behind one common interface, the S8 head-to-head that picks
the winner, the explainability artifacts, and the decision record.
**It supersedes Cell 8 of `PIPELINE_NOTEBOOK_CODE.md`** — when assembling the
notebook, use the cells below instead of the old Cell 8.

**Decision rule (from plan §4, recorded before running):**
EBM is promoted only if it beats LR by **≥ 0.005 mean AUC on S8's three CV
folds** *and* its explanations are stable across folds (min pairwise cosine of
fold-level mean theme vectors ≥ 0.90). Otherwise LR wins on simplicity.
Ties go to LR — it is the model named in the proposal deck (p.9).

---

## Prerequisites

1. Notebook already contains the foundation cells **0-1 … 0-7 from
   `STAGE0_FOUNDATION.md`** (config/paths, provenance utils, `DATA`/`DATA_SHA`,
   `S1_FEATS`/`THEME_MAP`/`THEMES`, `scheme_splits`, `process_split`,
   `ndcg_at5`) — run them top to bottom in a fresh kernel first.
2. One-time install (pure-Python wheels on Windows):

   ```
   pip install "interpret>=0.4"
   ```

---

## Cell S1-0 — Prerequisite check

Run this first; it fails loudly instead of half-working.

```python
# --- Stage 1 prerequisite check ---------------------------------------------
_required = ["DATA", "DATA_SHA", "S1_FEATS", "THEME_MAP", "THEMES",
             "scheme_splits", "process_split", "ndcg_at5", "ID_COLS",
             "RESULTS", "SEED", "CONFIG", "params_hash", "df_fingerprint"]
_missing = [n for n in _required if n not in globals()]
assert not _missing, f"Run the foundation cells first; missing: {_missing}"

import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

try:
    import interpret
    from interpret.glassbox import ExplainableBoostingClassifier
    assert hasattr(ExplainableBoostingClassifier, "eval_terms"), (
        "interpret too old — eval_terms() missing. Run: pip install -U 'interpret>=0.4'"
    )
    print(f"interpret {interpret.__version__} OK")
except ImportError as e:
    raise SystemExit("pip install 'interpret>=0.4' first") from e

print(f"S1_FEATS: {len(S1_FEATS)} features | themes: {len(THEMES)}")
```

**Verify:** prints interpret version and `S1_FEATS: 27`.

---

## Cell S1-1 — Common interface + both models

Design notes, so the verification is informed:

- **One contract for everything downstream.** Both classes expose
  `fit(train_df)`, `predict_proba(df)`, `contributions(df)` (per-feature
  signed logit contributions, columns = `S1_FEATS`), `theme_contributions(df)`,
  and `describe()` (exact params for the provenance record). The card code,
  the stacker, and the stability analysis never need to know which model won.
- **Exactness is asserted, not assumed.** For LR the identity is
  `Σⱼ βⱼxⱼ + β₀ == logit`. For EBM it is
  `σ(intercept + Σ terms) == predict_proba` — `eval_terms()` returns the
  per-term local scores, which for `interactions=0` are per-feature.
  Both asserts run on **every** `contributions()` call (they are cheap).
- **EBM is constrained to main effects** (`interactions=0`): Stage 3's FM
  owns interactions, and main-effects keep the card vocabulary identical to
  LR's. The engineered `align_*` features already carry the one interaction
  that matters (A's preference × B's self-rating).
- **EBM's internal bagging/early-stopping splits rows randomly *within* the
  training waves.** Eval waves are never seen — the wave-level leakage rule
  is intact.

```python
def _to_theme(contrib_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature contributions into theme columns (shared helper)."""
    out = pd.DataFrame(0.0, index=contrib_df.index, columns=THEMES)
    for f in contrib_df.columns:
        out[THEME_MAP[f]] += contrib_df[f]
    return out


class Stage1LR:
    """Deck-named baseline: L2 logistic regression on dec over S1_FEATS."""

    name = "lr"

    def __init__(self, C=1.0, seed=SEED):
        self.C = C
        self.model = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, train_df):
        self.model.fit(train_df[S1_FEATS].to_numpy(float), train_df["dec"])
        self.coef_ = pd.Series(self.model.coef_[0], index=S1_FEATS)
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(df[S1_FEATS].to_numpy(float))[:, 1]

    def contributions(self, df):
        X = df[S1_FEATS].to_numpy(float)
        C = X * self.model.coef_[0]
        logit = X @ self.model.coef_[0] + self.intercept_
        assert np.allclose(C.sum(1) + self.intercept_, logit, atol=1e-9), \
            "LR decomposition broke"
        return pd.DataFrame(C, columns=S1_FEATS, index=df.index)

    def theme_contributions(self, df):
        return _to_theme(self.contributions(df))

    def describe(self):
        return {"model": "LogisticRegression", "C": self.C, "penalty": "l2",
                "features": list(S1_FEATS)}


class Stage1EBM:
    """Contender: Explainable Boosting Machine, main effects only.
    logit = intercept + sum_j f_j(x_j); per-pair contribution of feature j
    is f_j(x_j) — same additive semantics as LR, learned shapes instead of
    straight lines."""

    name = "ebm"

    def __init__(self, seed=SEED, **overrides):
        self.params = dict(
            interactions=0,          # main effects ONLY (Stage 3 owns interactions)
            outer_bags=14,           # internal bagging stabilizes small-fold shapes
            learning_rate=0.01,
            max_bins=256,            # plenty for 8k rows; smoother than 1024
            random_state=seed,
            n_jobs=-1,
        )
        self.params.update(overrides)
        self.model = ExplainableBoostingClassifier(**self.params)

    def fit(self, train_df):
        self.model.fit(train_df[S1_FEATS], train_df["dec"])
        self.intercept_ = float(np.ravel(self.model.intercept_)[0])
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(df[S1_FEATS])[:, 1]

    def contributions(self, df):
        terms = self.model.eval_terms(df[S1_FEATS])          # n x n_terms
        C = pd.DataFrame(terms, columns=self.model.term_names_, index=df.index)
        assert set(C.columns) == set(S1_FEATS), \
            f"EBM term mismatch: {set(S1_FEATS) ^ set(C.columns)}"
        C = C[list(S1_FEATS)]
        # exactness: sigma(intercept + sum terms) must equal predict_proba
        p_from_terms = 1.0 / (1.0 + np.exp(-(C.sum(1).to_numpy() + self.intercept_)))
        assert np.allclose(p_from_terms, self.predict_proba(df), atol=1e-6), \
            "EBM decomposition broke"
        return C

    def theme_contributions(self, df):
        return _to_theme(self.contributions(df))

    def describe(self):
        return {"model": "ExplainableBoostingClassifier", **self.params,
                "features": list(S1_FEATS)}


STAGE1_MODELS = {"lr": Stage1LR, "ebm": Stage1EBM}


def make_stage1():
    """Factory the pipeline runner should call (model choice lives in CONFIG)."""
    name = CONFIG.get("s1_model", "lr")
    if name == "lr":
        return Stage1LR(C=CONFIG.get("s1_C", 1.0))
    return Stage1EBM()

print("Stage 1 models ready:", list(STAGE1_MODELS))
```

> **Integration note:** after the head-to-head verdict, the only change in
> `PIPELINE_NOTEBOOK_CODE.md` Cell 11 is `Stage1CB()` → `make_stage1()` inside
> `_fit_s12_predict`, plus the two new CONFIG keys
> (`"s1_model": "lr" | "ebm"`).

**Verify:** cell runs clean; `make_stage1()` returns an LR instance while
`CONFIG` has no `s1_model` key.

---

## Cell S1-2 — Smoke test on one S8 fold (both models)

```python
_sp = scheme_splits("S8", DATA["drop"])[0]            # rr-fold 0
_tr, _ev, _ = process_split(DATA["drop"], _sp["train_waves"], _sp["eval_waves"])

for name, cls in STAGE1_MODELS.items():
    t0 = time.time()
    m = cls().fit(_tr)
    p = m.predict_proba(_ev)
    s = _ev[ID_COLS].copy()
    s["p"] = p
    ndcg, n_users = ndcg_at5(s, "p")
    print(f"{name:>4s}: AUC={roc_auc_score(_ev['dec'], p):.4f}  "
          f"logloss={log_loss(_ev['dec'], p, labels=[0,1]):.4f}  "
          f"NDCG5_uni={ndcg:.3f} ({n_users} users)  [{time.time()-t0:.0f}s]")

# explanation parity preview: same pair, both vocabularies
_pair = _ev.iloc[[0]]
print(f"\nDemo pair iid={int(_pair['iid'].iloc[0])} -> pid={int(_pair['pid'].iloc[0])}")
for name, cls in STAGE1_MODELS.items():
    th = cls().fit(_tr).theme_contributions(_pair).iloc[0]
    top = th.sort_values(ascending=False)
    print(f"  {name}: top {list(top.index[:2])} {top.values[:2].round(3)}"
          f" | bottom {list(top.index[-2:])} {top.values[-2:].round(3)}")
```

**Verify:**
- both decomposition asserts stay silent (they run inside `contributions`);
- LR AUC ≳ 0.55; EBM AUC ≥ LR is *expected but not required* on a single fold;
- EBM fit time on ~5.4k rows ≈ 5–30 s (bagged boosting; one-time cost);
- the demo pair's top/bottom themes look sane under both models (they will
  not be identical — that is the point of the comparison).

---

## Cell S1-3 — The S8 head-to-head (decision cell)

Race on the three S8 CV folds only (the designated tuning scheme — the other
nine schemes stay untouched until the model is frozen). Includes the LR C-grid
so Stage-1 tuning happens here once, in one place.

```python
S8_CV = [sp for sp in scheme_splits("S8", DATA["drop"]) if sp["kind"] == "cv"]

CANDIDATES = {
    "lr_C0.1": lambda: Stage1LR(C=0.1),
    "lr_C1":   lambda: Stage1LR(C=1.0),
    "lr_C10":  lambda: Stage1LR(C=10.0),
    "ebm":     lambda: Stage1EBM(),
}

rows, theme_means = [], {}
for sp in S8_CV:
    tr, ev, _ = process_split(DATA["drop"], sp["train_waves"], sp["eval_waves"])
    for cand, ctor in CANDIDATES.items():
        t0 = time.time()
        m = ctor().fit(tr)
        p = m.predict_proba(ev)
        s = ev[ID_COLS].copy(); s["p"] = p
        ndcg, _n = ndcg_at5(s, "p")
        rows.append({"fold": sp["tag"], "cand": cand,
                     "auc": roc_auc_score(ev["dec"], p),
                     "logloss": log_loss(ev["dec"], p, labels=[0, 1]),
                     "ndcg5_uni": ndcg, "fit_s": round(time.time() - t0, 1)})
        theme_means[(cand, sp["tag"])] = m.theme_contributions(ev).mean()

h2h = pd.DataFrame(rows)
print(h2h.pivot(index="cand", columns="fold",
                values="auc").round(4).to_string())
summary = (h2h.groupby("cand")[["auc", "logloss", "ndcg5_uni"]]
              .agg(["mean", "std"]).round(4))
print("\n", summary.to_string())

# --- explanation stability: min pairwise cosine of fold-mean theme vectors ---
def _min_cosine(cand):
    V = np.stack([theme_means[(cand, sp["tag"])].to_numpy() for sp in S8_CV])
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = Vn @ Vn.T
    return cos[np.triu_indices_from(cos, k=1)].min()

stab = {c: round(float(_min_cosine(c)), 4) for c in CANDIDATES}
print("\nExplanation stability (min fold-pair cosine):", stab)

# --- decision rule (plan section 4, recorded before running) ------------------
best_lr = (h2h[h2h["cand"].str.startswith("lr")]
           .groupby("cand")["auc"].mean().idxmax())
lr_auc  = h2h[h2h["cand"] == best_lr]["auc"].mean()
ebm_auc = h2h[h2h["cand"] == "ebm"]["auc"].mean()
ebm_wins = (ebm_auc - lr_auc >= 0.005) and (stab["ebm"] >= 0.90)
winner = "ebm" if ebm_wins else best_lr
print(f"\nbest LR: {best_lr} (AUC {lr_auc:.4f}) | EBM AUC {ebm_auc:.4f} | "
      f"delta {ebm_auc - lr_auc:+.4f}")
print(f"WINNER -> {winner}"
      f"{'' if ebm_wins else '  (EBM did not clear +0.005 AUC with stability >= 0.90)'}")
```

**Verify:**
- 12 rows (3 folds × 4 candidates); EBM column slowest but < 1 min/fold;
- stability cosines near 1.0 for LR variants (sanity: LR is deterministic
  given the fold); EBM's value is the informative one;
- the verdict line prints one winner. **Do not override it ad hoc** — if you
  disagree with the rule, change the rule in the plan first, then re-run
  (that is the record-keeping discipline).

---

## Cell S1-4 — Explainability artifacts: shapes vs lines

One plot, both models, no `interpret` internals: x = feature value, y = that
feature's logit contribution on eval rows. LR appears as a straight line, EBM
as a step curve — the visual argument for (or against) the upgrade.

```python
import matplotlib.pyplot as plt

_sp = S8_CV[0]
tr, ev, _ = process_split(DATA["drop"], _sp["train_waves"], _sp["eval_waves"])
fits = {name: cls().fit(tr) for name, cls in STAGE1_MODELS.items()}
contribs = {name: m.contributions(ev) for name, m in fits.items()}

top_feats = (contribs["ebm"].abs().mean().sort_values(ascending=False)
             .head(6).index.tolist())

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), top_feats):
    for name, marker, alpha in [("lr", ".", 0.25), ("ebm", ".", 0.25)]:
        order = ev[f].argsort()
        ax.plot(ev[f].to_numpy()[order], contribs[name][f].to_numpy()[order],
                marker, ms=3, alpha=alpha, label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f, fontsize=10)
axes[0, 0].legend()
fig.suptitle("Stage 1 — per-feature logit contribution: LR (line) vs EBM (shape)")
fig.tight_layout()
(RESULTS / "figures" / "stage1").mkdir(exist_ok=True, parents=True)
fig.savefig(RESULTS / "figures" / "stage1" / "lr_vs_ebm_shapes.png", dpi=150)
plt.show()

coef_fig, ax = plt.subplots(figsize=(7, 6))
fits["lr"].coef_.sort_values().plot.barh(ax=ax)
ax.set_title("Stage 1 LR coefficients (logit per unit feature)")
ax.axvline(0, color="gray", lw=0.8)
coef_fig.tight_layout()
coef_fig.savefig(RESULTS / "figures" / "stage1" / "lr_coefficients.png", dpi=150)
plt.show()
```

**Verify:** in the shapes figure, LR points fall on straight lines through the
origin region while EBM shows curvature exactly where we predicted
non-linearity (`age_diff` flattening, `align_*` saturation). If EBM's curves
look like noise rather than shapes, that is evidence *for* LR — say so in the
decision record.

---

## Cell S1-5 — Freeze the decision + provenance record

```python
stage1_decision = {
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "decision_rule": "EBM needs >= +0.005 mean AUC over best LR on S8 CV "
                     "and min fold-pair theme-cosine >= 0.90; ties -> LR",
    "winner": winner,
    "candidates": {c: {"auc_mean": float(h2h[h2h['cand'] == c]['auc'].mean()),
                       "auc_std": float(h2h[h2h['cand'] == c]['auc'].std()),
                       "ndcg5_uni_mean": float(h2h[h2h['cand'] == c]['ndcg5_uni'].mean()),
                       "stability_min_cos": stab[c]}
                   for c in CANDIDATES},
    "winner_params": (Stage1EBM() if winner == "ebm"
                      else Stage1LR(C=float(winner.split("C")[1]))).describe(),
    "tuning_scheme": "S8 (drop-w12, holdout [8,13,14,19]) CV folds only",
    "data_sha": DATA_SHA, "data_fingerprint": df_fingerprint(DATA["drop"]),
    "config_hash": params_hash(CONFIG),
    "per_fold": h2h.to_dict("records"),
}
out = RESULTS / "runs" / f"stage1_decision_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps(stage1_decision, indent=2), encoding="utf-8")
print(f"Saved {out.name}")
print(f"\nNow set in Cell 1:  CONFIG['s1_model'] = '{'ebm' if winner=='ebm' else 'lr'}'"
      + (f"  and CONFIG['s1_C'] = {winner.split('C')[1]}" if winner != "ebm" else ""))
```

**Verify:** JSON lands in `results/runs/`; it contains everything needed to
reproduce or audit the choice (rule, per-fold numbers, params, data SHA).
Update `CONFIG` as printed — Stage 1 is now frozen; do not revisit it when
later stages underperform (that is what the ablation rows are for).

---

## What deliberately is NOT in this document

- No kNN, FM, fusion, or card-rendering code (Stages 2–4 get their own
  documents when we reach them).
- No runs on schemes other than S8 — model selection happens once, on the
  designated tuning scheme, per plan §2.2.
- No EBM interactions, no SHAP, no monotonicity constraints — main-effects
  shapes are the entire upgrade under test.
