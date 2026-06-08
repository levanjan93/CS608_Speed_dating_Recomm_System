# Speed Dating Recommender System

Recommender system prototypes for the Columbia Speed Dating dataset (CS608). The goal is to predict pairwise compatibility and rank wave-mates for mutual matches.

## Project structure

```
├── data/                          # Raw and preprocessed datasets
├── docs/                          # Design notes and study guides
├── notebooks/                     # EDA and model prototypes
├── build_two_sided_data.py        # Build directed pair feature matrix (A→B)
├── load_one_sided_data.py         # Build one-sided participant features
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Data pipeline

1. Place `Speed Dating Data.csv` in `data/` (included in this repo).
2. Build one-sided features:
   ```bash
   python load_one_sided_data.py
   ```
3. Build two-sided pair features:
   ```bash
   python build_two_sided_data.py
   ```

See `docs/preprocessing_and_feature_selection.md` and `docs/build_two_sided_features.md` for details.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/eda.ipynb` | Exploratory data analysis |
| `notebooks/content_based_filtering_prototype.ipynb` | Content-based filtering baseline (LightGBM, ranking metrics) |
| `notebooks/knn_cf_prototype.ipynb` | KNN collaborative filtering prototype |

## Evaluation

Models predict `dec` (whether A says yes to B) and are evaluated on ranking quality and mutual `match` outcomes. See `docs/content_based_filtering_prototype.md` for the cross-validation setup and metrics.
