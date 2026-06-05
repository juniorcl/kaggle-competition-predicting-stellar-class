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
                    │  layer_1              │
                    │  (21 models)          │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  models/layer_1/*     │
                    │  .pkl files           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --stack      │
                    │  layer_1              │
                    │  (cross_val_predict)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ X_train_stacking_     │
                    │ layer_1.parquet       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --layer      │
                    │  layer_2              │
                    │  (27 models)          │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  models/layer_2/*     │
                    │  .pkl files           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  main.py --stack      │
                    │  layer_2              │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ X_train_stacking_     │
                    │ layer_2.parquet       │
                    └───────────────────────┘
```

### Layers

| Layer | Models | Trained on | Stacking features |
|-------|--------|------------|-------------------|
| **layer_1** | 21 models (linear, tree, TruncatedSVD, kernel approximation variants) | `X_train_raw.parquet` | Prediction probabilities (class 0, 1) from each model → 42 features |
| **layer_2** | 27 models (same types + native tree, no preprocessing) | `X_train_stacking_layer_1.parquet` | Prediction probabilities → 54 features |
| **layer_N** | Same 27 stacking models | `X_train_stacking_layer_{N-1}.parquet` | Configurable via `NUM_LAYERS` in `src/config.py` |

Each layer's models generate out-of-fold predictions (`cross_val_predict`) for training data and direct predictions for test data. These probability features feed into the next layer.

### Adding More Layers

Change `NUM_LAYERS` in `src/config.py`:

```python
NUM_LAYERS = 3  # adds layer_3
```

Then run `python pipeline.py` or train individually:

```bash
python main.py --layer layer_3
python main.py --stack layer_3
```

## Project Structure

```
├── main.py                  # CLI: train models or generate stacking features
├── pipeline.py              # End-to-end orchestrator (loops over N layers)
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
│   ├── config.py             # Model registries, layer config, NUM_LAYERS
│   ├── stacking.py           # generate_stacking_features()
│   ├── stacking_tuning.py    # 27 stacking tuners (layers 2+)
│   ├── *tuning.py            # One file per raw model (21 total)
│   ├── utils/
│   │   ├── preprocessing.py  # column_transformer (scalers + target encoder)
│   │   ├── dump_model.py     # pickle serializer
│   │   └── logging_setup.py  # Centralized logger factory
│   └── logs/                 # Per-model training logs
├── models/
│   ├── layer_1/              # Trained .pkl files (layer_1 raw models)
│   └── layer_2/              # Trained .pkl files (layer_2 stacking models)
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

### 1. Train all models (raw features)

```bash
python main.py
```

Trains all 21 raw models on `data/X_train_raw.parquet`.

### 2. Train a specific layer

```bash
python main.py --layer layer_1
python main.py --layer layer_2 --data data/X_train_stacking_layer_1.parquet
```

Layers >= 2 auto-resolve data from previous layer's stacking features.

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

### 4. Full end-to-end pipeline

```bash
python pipeline.py
```

Automatically loops over all layers (configured by `NUM_LAYERS` in `src/config.py`).

### 5. Generate submission

Use notebooks `5.0_stacking_submission.ipynb` or `5.1_stacking_submission.ipynb` with the generated stacking features to train a meta-model and produce `submission.csv`.

## Model Details

### Layer 1 — Raw Features (21 models)

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

### Layer 2 — Stacking Features (27 models)

All models train on stacking features (class probabilities) with `StandardScaler` but no other preprocessing.

| Model | n_trials | Pipeline |
|-------|----------|----------|
| Stacking SGD | 60 | `StandardScaler → SGDClassifier` |
| Stacking Ridge | 60 | `StandardScaler → Ridge → Calibrated` |
| Stacking LogisticRegression | 60 | `StandardScaler → LogisticRegression` |
| Stacking XGBoost | 30 | `XGBClassifier` |
| Stacking CatBoost | 30 | `CatBoostClassifier` |
| Stacking LightGBM | 30 | `LGBMClassifier` |
| Stacking ExtraTrees | 30 | `ExtraTreesClassifier` |
| Stacking RandomForest | 30 | `RandomForestClassifier` |
| Stacking HistGradientBoosting | 30 | `HistGradientBoostingClassifier` |
| Stacking TruncatedSVD + CatBoost | 30 | `Scaler → TruncatedSVD → CatBoost` |
| Stacking TruncatedSVD + XGBoost | 30 | `Scaler → TruncatedSVD → XGBoost` |
| Stacking TruncatedSVD + LightGBM | 30 | `Scaler → TruncatedSVD → LightGBM` |
| Stacking TruncatedSVD + ExtraTrees | 30 | `Scaler → TruncatedSVD → ExtraTrees` |
| Stacking TruncatedSVD + RandomForest | 30 | `Scaler → TruncatedSVD → RandomForest` |
| Stacking TruncatedSVD + HistGradientBoosting | 30 | `Scaler → TruncatedSVD → HistGradientBoosting` |
| Stacking TruncatedSVD + SGD | 60 | `Scaler → TruncatedSVD → SGD` |
| Stacking TruncatedSVD + Ridge | 60 | `Scaler → TruncatedSVD → Ridge → Calibrated` |
| Stacking TruncatedSVD + LogisticRegression | 60 | `Scaler → TruncatedSVD → LogisticRegression` |
| Stacking TruncatedSVD + KNN | 60 | `Scaler → TruncatedSVD → Scaler → KNN` |
| Stacking TruncatedSVD + Nystroem + KNN | 60 | `Scaler → TruncatedSVD → Scaler → Nystroem → Scaler → KNN` |
| Stacking TruncatedSVD + RBFSampler + KNN | 60 | `Scaler → TruncatedSVD → Scaler → RBFSampler → Scaler → KNN` |
| Stacking TruncatedSVD + Nystroem + SGD | 60 | `Scaler → TruncatedSVD → Scaler → Nystroem → Scaler → SGD` |
| Stacking TruncatedSVD + RBFSampler + SGD | 60 | `Scaler → TruncatedSVD → Scaler → RBFSampler → Scaler → SGD` |
| Stacking TruncatedSVD + Nystroem + LogisticRegression | 60 | `Scaler → TruncatedSVD → Scaler → Nystroem → Scaler → LogisticRegression` |
| Stacking TruncatedSVD + RBFSampler + LogisticRegression | 60 | `Scaler → TruncatedSVD → Scaler → RBFSampler → Scaler → LogisticRegression` |
| Stacking TruncatedSVD + Nystroem + Ridge | 60 | `Scaler → TruncatedSVD → Scaler → Nystroem → Scaler → Ridge → Calibrated` |
| Stacking TruncatedSVD + RBFSampler + Ridge | 60 | `Scaler → TruncatedSVD → Scaler → RBFSampler → Scaler → Ridge → Calibrated` |

### Preprocessing (`column_transformer`)

```python
ColumnTransformer([
    ("target_encoder", TargetEncoder(), ['spectral_type', 'galaxy_population']),
    ("standard_scaler", StandardScaler(), ['alpha', 'delta']),
], remainder=RobustScaler())
```

Used only for Layer 1 (raw features). Layers 2+ receive probability features with `StandardScaler` only.

## Key Features

- **N-layer stacking**: Configurable number of stacking layers via `NUM_LAYERS` in `src/config.py`
- **Optuna hyperparameter tuning** with 5-fold StratifiedKFold CV, MedianPruner, log loss minimization
- **Per-model logging** to `src/logs/<model_name>.log`
- **Training results** saved to `results/training_results.csv` with best loss, params, duration
- **Stacking features** generated via `cross_val_predict` for clean out-of-fold predictions
- **All model types per layer**: Linear, tree, TruncatedSVD, kernel approximation variants
- **Modular design**: Add new models by creating a tuning function and adding to config

## Adapting for a New Competition

1. Replace `data/train.csv`, `data/test.csv`
2. Run feature engineering notebook
3. Update `src/utils/preprocessing.py` if column names differ
4. Update `target` parameter in `main.py` if target column name differs
5. Run `python pipeline.py`
6. Update submission notebook with new column names
