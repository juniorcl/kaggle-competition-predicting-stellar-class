# Kaggle Competition: Predicting Stellar Classification

Multi-layer stacking pipeline for Kaggle's [Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) — stellar classification from astronomical measurements.

## Pipeline Architecture

```
                       ┌──────────────────┐
                       │  X_train_raw     │
                       │  X_test_raw      │
                       │  y_train         │
                       └────────┬─────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --layer      │
                    │  layer_N              │
                    │  (24 models)          │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  models/layer_N/*     │
                    │  .pkl files           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --stack      │
                    │  layer_N              │
                    │  (cross_val_predict)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ X_train_stacking_     │
                    │ layer_N.parquet       │
                    └───────────────────────┘
```

### Layers

| Layer | Models | Trained on | Stacking features |
|-------|--------|------------|-------------------|
| **layer_1** | 24 models (linear, tree, TruncatedSVD, neural network, naive bayes) | `X_train_raw.parquet` | Prediction probabilities from each model → 72 features |
| **layer_N** | Same 24 models | `X_train_stacking_layer_{N-1}.parquet` | Configurable via `NUM_LAYERS` in `src/config.py` |

All layers use the same unified model registry. Each model generates out-of-fold predictions (`cross_val_predict`) for training data and direct predictions for test data. These probability features feed into the next layer.

### Adding More Layers

Change `NUM_LAYERS` in `src/config.py`:

```python
NUM_LAYERS = 3  # adds layer_3
```

Then train individually:

```bash
python main.py --layer layer_3
python main.py --stack layer_3
```

## Project Structure

```
├── main.py                  # CLI: train models or generate stacking features
├── pyproject.toml           # Python dependencies
├── data/
│   ├── train.csv            # Original competition data
│   ├── test.csv             # Original competition data
│   ├── X_train_raw.parquet  # Feature-engineered training data
│   ├── X_test_raw.parquet   # Feature-engineered test data
│   ├── y_train.parquet      # Target (class_encoded: 0=GALAXY, 1=QSO, 2=STAR)
│   ├── X_train_stacking_layer_1.parquet  # Layer-1 stacking features
│   ├── X_test_stacking_layer_1.parquet
│   ├── X_train_stacking_layer_2.parquet  # Layer-2 stacking features
│   ├── X_test_stacking_layer_2.parquet
│   ├── submission_*.csv     # Kaggle submission files
│   └── sample_submission.csv
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Model registry, layer config, NUM_LAYERS
│   ├── *tuning.py            # One file per model (24 total)
│   ├── utils/
│   │   ├── preprocessing.py  # column_transformer (scalers + target encoder)
│   │   ├── dump_model.py     # pickle serializer
│   │   └── logging_setup.py  # Centralized logger factory
│   └── logs/                 # Per-model training logs
├── models/
│   ├── layer_1/              # Trained .pkl files (layer_1)
│   └── layer_2/              # Trained .pkl files (layer_2)
├── notebooks/
│   ├── 1.0_eda.ipynb
│   ├── 2.0_feature_engineering.ipynb
│   ├── 3.0_machine_learning.ipynb
│   ├── 4.0_stacking_data_layer_one.ipynb
│   ├── 5.0_stacking_submission.ipynb    # LogisticRegression meta-model
│   ├── 5.1_stacking_submission.ipynb    # LightGBM meta-model
│   └── utils/
│       ├── load_model.py
│       └── calibration.py   # MulticlassThresholdOptimizer
└── results/
    └── training_results.csv  # Metrics per training run
```

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13. Dependencies managed by `uv`. See `pyproject.toml` for full list.

## Usage

### 1. Train all models

```bash
python main.py
```

Trains all 24 models on `data/X_train_raw.parquet` (layer_1).

### 2. Train a specific layer

```bash
python main.py --layer layer_1
python main.py --layer layer_2
```

Layers >= 2 auto-resolve data from previous layer's stacking features. Each layer trains the same set of models.

### 3. Generate stacking features

```bash
python main.py --stack layer_1
python main.py --stack layer_2
```

Each call:
- Loads trained `.pkl` files from `models/<layer>/`
- Generates out-of-fold prediction probabilities for training data (`cross_val_predict`, 5-fold)
- Generates direct prediction probabilities for test data
- Saves parquet files to `data/X_train_stacking_<layer>.parquet` and `data/X_test_stacking_<layer>.parquet`

### 4. Generate submission

Use notebooks `5.0_stacking_submission.ipynb` or `5.1_stacking_submission.ipynb` with the generated stacking features to train a meta-model and produce `submission.csv`.

## Model Details

All layers use the same unified model registry (24 models). Layer 1 trains on raw features with `column_transformer` preprocessing. Layers 2+ train on stacking probabilities with `StandardScaler` only.

Models without n_trials are simple classifiers with no hyperparameter tuning.

| Model | n_trials | Pipeline |
|-------|----------|----------|
| XGBoost | 30 | `TargetEncoder → XGBClassifier` |
| CatBoost | 30 | `CatBoostEncoder → CatBoostClassifier` |
| LightGBM | 30 | `TargetEncoder → LGBMClassifier` |
| ExtraTrees | 30 | `TargetEncoder → ExtraTreesClassifier` |
| RandomForest | 30 | `TargetEncoder → RandomForestClassifier` |
| HistGradientBoosting | 30 | `TargetEncoder → HistGradientBoostingClassifier` |
| SGDClassifier | 60 | `column_transformer → SGDClassifier` |
| Ridge + Calibrated | 60 | `column_transformer → RidgeClassifier → Calibrated` |
| LogisticRegression | 60 | `column_transformer → LogisticRegression` |
| MLP | 30 | `column_transformer → MLPClassifier` |
| LinearSVC + Calibrated | 30 | `column_transformer → LinearSVC → Calibrated` |
| GaussianNB | — | `column_transformer → GaussianNB` |
| ComplementNB | — | `column_transformer → ComplementNB` |
| LDA | — | `column_transformer → LinearDiscriminantAnalysis` |
| TruncatedSVD + XGBoost | 30 | `column_transformer → TruncatedSVD → XGBClassifier` |
| TruncatedSVD + CatBoost | 30 | `column_transformer → TruncatedSVD → CatBoostClassifier` |
| TruncatedSVD + LightGBM | 30 | `column_transformer → TruncatedSVD → LGBMClassifier` |
| TruncatedSVD + ExtraTrees | 30 | `column_transformer → TruncatedSVD → ExtraTreesClassifier` |
| TruncatedSVD + RandomForest | 30 | `column_transformer → TruncatedSVD → RandomForestClassifier` |
| TruncatedSVD + HistGradientBoosting | 30 | `column_transformer → TruncatedSVD → HistGradientBoostingClassifier` |
| TruncatedSVD + SGD | 60 | `column_transformer → TruncatedSVD → Scaler → SGDClassifier` |
| TruncatedSVD + Ridge + Calibrated | 60 | `column_transformer → TruncatedSVD → Scaler → RidgeClassifier → Calibrated` |
| TruncatedSVD + LogisticRegression | 60 | `column_transformer → TruncatedSVD → Scaler → LogisticRegression` |
| TruncatedSVD + KNN | 60 | `column_transformer → TruncatedSVD → Scaler → KNeighborsClassifier` |

### Preprocessing (`column_transformer`)

```python
ColumnTransformer([
    ("target_encoder", TargetEncoder(), ['spectral_type', 'galaxy_population']),
    ("standard_scaler", StandardScaler(), ['alpha', 'delta']),
], remainder=RobustScaler())
```

Used for raw features (layer_1). Tree-based models use `TargetEncoder` or `CatBoostEncoder` directly instead of `column_transformer`. Layers 2+ receive probability features with `StandardScaler` only.

## Key Features

- **N-layer stacking**: Configurable number of stacking layers via `NUM_LAYERS` in `src/config.py`
- **Unified model registry**: Same 24 models across all layers (linear, tree, TruncatedSVD, neural network, naive bayes)
- **Optuna hyperparameter tuning** with 5-fold StratifiedKFold CV, MedianPruner, log loss minimization
- **Per-model logging** to `src/logs/<model_name>.log`
- **Training results** saved to `results/training_results.csv`
- **Stacking features** generated via `cross_val_predict` for clean out-of-fold predictions
- **Modular design**: Add new models by creating a tuning function and registering in `MODEL_REGISTRY`

## Adapting for a New Competition

1. Replace `data/train.csv`, `data/test.csv`
2. Run feature engineering notebook
3. Update `src/utils/preprocessing.py` if column names differ
4. Update target encoding columns in tuning files if column names differ
5. Run `python main.py` to train all models
6. Update submission notebook with new column names
