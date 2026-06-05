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
                    │  layer_one            │
                     │  (21 models)          │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  models/layer_one/*   │
                    │  .pkl files           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --stack      │
                    │  layer_one            │
                    │  (cross_val_predict)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ X_train_stacking_     │
                    │ layer_one.parquet     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --layer      │
                    │  layer_two            │
                    │  (6 models)           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  models/layer_two/*   │
                    │  .pkl files           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --stack      │
                    │  layer_two            │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ X_train_stacking_     │
                    │ layer_two.parquet     │
                    └───────────────────────┘
```

### Layers

| Layer | Models | Trained on | Stacking features |
|-------|--------|------------|-------------------|
| **layer_one** | 21 models (SGD, Ridge, LogisticRegression, 10 TruncatedSVD variants, 3 KNN variants, 6 kernel approximation variants) | `X_train_raw.parquet` | Prediction probabilities (class 0, 1) from each model → 42 features |
| **layer_two** | 6 tree-based models (XGBoost, CatBoost, LightGBM, ExtraTrees, RandomForest, HistGradientBoosting) | `X_train_stacking_layer_one.parquet` | Prediction probabilities → 12 features |

Each layer's models generate out-of-fold predictions (`cross_val_predict`) for training data and direct predictions for test data. These probability features feed into the next layer.

## Project Structure

```
├── main.py                  # CLI: train models or generate stacking features
├── pipeline.py              # End-to-end orchestrator (all layers)
├── pyproject.toml           # Python dependencies
├── data/
│   ├── train.csv            # Original competition data
│   ├── test.csv             # Original competition data
│   ├── X_train_raw.parquet  # Feature-engineered training data
│   ├── X_test_raw.parquet   # Feature-engineered test data
│   ├── y_train.parquet      # Target (class_encoded: 0=GALAXY, 1=QSO, 2=STAR)
│   ├── X_train_stacking_layer_one.parquet  # Layer-one stacking features
│   ├── X_test_stacking_layer_one.parquet
│   ├── submission_*.csv     # Kaggle submission files
│   └── sample_submission.csv
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # MODEL_REGISTRY, LAYERS config
│   ├── stacking.py           # generate_stacking_features()
│   ├── *tuning.py            # One file per model (27 total)
│   ├── utils/
│   │   ├── preprocessing.py  # column_transformer (scalers + target encoder)
│   │   ├── dump_model.py     # pickle serializer
│   │   └── logging_setup.py  # Centralized logger factory
│   └── logs/                 # Per-model training logs
├── models/
│   ├── layer_one/            # Trained .pkl files (layer_one models)
│   └── layer_two/            # Trained .pkl files (layer_two models)
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

### 1. Train all models (both layers)

```bash
python main.py
```

Trains all 27 models on `data/X_train_raw.parquet`.

### 2. Train a specific layer

```bash
python main.py --layer layer_one
python main.py --layer layer_two --data data/X_train_stacking_layer_one.parquet
```

### 3. Generate stacking features

```bash
python main.py --stack layer_one
python main.py --stack layer_two
```

Each call:
- Loads trained `.pkl` files from `models/<layer>/`
- Generates out-of-fold prediction probabilities for training data (`cross_val_predict`, 5-fold)
- Generates direct prediction probabilities for test data
- Saves parquet files to `data/X_train_stacking_<layer>.parquet` and `data/X_test_stacking_<layer>.parquet`

### 4. Full end-to-end pipeline

```bash
python pipeline.py
```

Equivalent to:
```
python main.py --layer layer_one
python main.py --stack layer_one
python main.py --layer layer_two --data data/X_train_stacking_layer_one.parquet
python main.py --stack layer_two
```

### 5. Generate submission

Use notebooks `5.0_stacking_submission.ipynb` or `5.1_stacking_submission.ipynb` with the generated stacking features to train a meta-model and produce `submission.csv`.

## Model Details

### Layer One (21 models)

| Model | n_trials | Pipeline |
|-------|----------|----------|
| SGDClassifier | 60 | `column_transformer → SGDClassifier` |
| Ridge + CalibratedClassifierCV | 60 | `column_transformer → Ridge → CalibratedClassifierCV` |
| LogisticRegression | 60 | `column_transformer → LogisticRegression` |
| TruncatedSVD + CatBoost | 30 | `column_transformer → TruncatedSVD → CatBoost` |
| TruncatedSVD + XGBoost | 30 | `column_transformer → TruncatedSVD → XGBoost` |
| TruncatedSVD + LightGBM | 30 | `column_transformer → TruncatedSVD → LightGBM` |
| TruncatedSVD + ExtraTrees | 30 | `column_transformer → TruncatedSVD → ExtraTrees` |
| TruncatedSVD + RandomForest | 30 | `column_transformer → TruncatedSVD → RandomForest` |
| TruncatedSVD + HistGradientBoosting | 30 | `column_transformer → TruncatedSVD → HistGradientBoosting` |
| TruncatedSVD + SGD | 60 | `column_transformer → TruncatedSVD → SGD` |
| TruncatedSVD + Ridge | 60 | `column_transformer → TruncatedSVD → Ridge → Calibrated` |
| TruncatedSVD + LogisticRegression | 60 | `column_transformer → TruncatedSVD → LogisticRegression` |
| TruncatedSVD + KNN | 60 | `column_transformer → TruncatedSVD → StandardScaler → KNN` |
| TruncatedSVD + Nystroem + KNN | 60 | `column_transformer → TruncatedSVD → StandardScaler → Nystroem → StandardScaler → KNN` |
| TruncatedSVD + RBFSampler + KNN | 60 | `column_transformer → TruncatedSVD → StandardScaler → RBFSampler → StandardScaler → KNN` |
| TruncatedSVD + Nystroem + SGD | 60 | `column_transformer → TruncatedSVD → StandardScaler → Nystroem → StandardScaler → SGD` |
| TruncatedSVD + RBFSampler + SGD | 60 | `column_transformer → TruncatedSVD → StandardScaler → RBFSampler → StandardScaler → SGD` |
| TruncatedSVD + Nystroem + LogisticRegression | 60 | `column_transformer → TruncatedSVD → StandardScaler → Nystroem → StandardScaler → LogisticRegression` |
| TruncatedSVD + RBFSampler + LogisticRegression | 60 | `column_transformer → TruncatedSVD → StandardScaler → RBFSampler → StandardScaler → LogisticRegression` |
| TruncatedSVD + Nystroem + Ridge | 60 | `column_transformer → TruncatedSVD → StandardScaler → Nystroem → StandardScaler → Ridge → Calibrated` |
| TruncatedSVD + RBFSampler + Ridge | 60 | `column_transformer → TruncatedSVD → StandardScaler → RBFSampler → StandardScaler → Ridge → Calibrated` |

### Layer Two (6 models)

| Model | n_trials | Pipeline |
|-------|----------|----------|
| XGBoost | 30 | `TargetEncoder → XGBClassifier` |
| CatBoost | 30 | `CatBoostEncoder → CatBoostClassifier` |
| LightGBM | 30 | `TargetEncoder → LGBMClassifier` |
| ExtraTrees | 30 | `TargetEncoder → ExtraTreesClassifier` |
| RandomForest | 30 | `TargetEncoder → RandomForestClassifier` |
| HistGradientBoosting | 30 | `TargetEncoder → HistGradientBoostingClassifier` |

### Preprocessing (`column_transformer`)

```python
ColumnTransformer([
    ("target_encoder", TargetEncoder(), ['spectral_type', 'galaxy_population']),
    ("standard_scaler", StandardScaler(), ['alpha', 'delta']),
], remainder=RobustScaler())
```

## Key Features

- **Optuna hyperparameter tuning** with 5-fold StratifiedKFold CV, MedianPruner, log loss minimization
- **Per-model logging** to `src/logs/<model_name>.log`
- **Training results** saved to `results/training_results.csv` with best loss, params, duration
- **Stacking features** generated via `cross_val_predict` for clean out-of-fold predictions
- **TruncatedSVD embeddings**: dimensionality reduction before classification (layer_one SVD variants)
- **Modular design**: add new models by creating a tuning function and adding to `src/config.py`

## Adapting for a New Competition

1. Replace `data/train.csv`, `data/test.csv`
2. Run feature engineering notebook
3. Update `src/utils/preprocessing.py` if column names differ
4. Update `target` parameter in `main.py` if target column name differs
5. Run `python pipeline.py`
6. Update submission notebook with new column names
