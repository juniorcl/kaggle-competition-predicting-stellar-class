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
                     │  train_layer_1.py     │
                     │  src/layer_one/       │
                     │  (7 registered models)│
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  models/layer_1/*.pkl │
                     └───────────┬───────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │ notebooks/4.x_stacking*.ipynb│
                  │ cross_val_predict → parquet  │
                  └──────────────┬──────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ X_train_stacking_     │
                     │ layer_one.parquet     │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  train_layer_2.py     │
                     │  src/layer_stacking/  │
                     │  (same 7 models)      │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  models/layer_2/*.pkl │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │     ... layer N       │
                     └───────────────────────┘
```

### Layers

| Layer | Models | Trained on | Package |
|-------|--------|------------|---------|
| **layer_1** | 7 registered (14 tuning files) | `X_train_raw.parquet` | `src/layer_one/` |
| **layer_N** | Same 7 models | `X_train_stacking_layer_{N-1}.parquet` | `src/layer_stacking/` |

Layer 1 uses `column_transformer` preprocessing (TargetEncoder + scalers). Layers 2+ receive stacking probabilities; models train directly without TargetEncoder.

### Adding More Layers

Create `train_layer_N.py` pointing to the next stacking parquet:

```bash
# 1. Generate stacking features via notebook
# 2. Create train script like train_layer_2.py but read layer_{N-1} data
python train_layer_N.py
```

## Project Structure

```
├── train_layer_1.py         # Train 7 models on raw features → models/layer_1/
├── train_layer_2.py         # Train 7 models on layer-1 stacking feats → models/layer_2/
├── train_layer_3.py         # Train 7 models on layer-2 stacking feats → models/layer_3/
├── main.py                  # Stub (incomplete)
├── pyproject.toml           # Python dependencies (uv)
├── data/
│   ├── train.csv            # Original competition data
│   ├── test.csv             # Original competition data
│   ├── sample_submission.csv
│   ├── X_train_raw.parquet  # Feature-engineered training data
│   ├── X_test_raw.parquet   # Feature-engineered test data
│   ├── X_train_fe.parquet   # Additional FE output
│   ├── X_test_fe.parquet
│   ├── y_train.parquet      # Target (class_encoded: 0=GALAXY, 1=QSO, 2=STAR)
│   ├── X_train_stacking_layer_{one,two,three}.parquet
│   ├── X_test_stacking_layer_{one,two,three}.parquet
│   └── submission_stacking_*.csv
├── src/
│   ├── layer_one/           # Raw-feature training package
│   │   ├── config.py        # MODEL_REGISTRY (7 models)
│   │   ├── *tuning.py       # 14 Optuna tuning scripts
│   │   ├── logs/            # Per-model training logs
│   │   └── utils/           # preprocessing, dump_model, logging_setup
│   └── layer_stacking/      # Stacking-feature training package
│       ├── config.py        # MODEL_REGISTRY (same 7 models, no TargetEncoder)
│       ├── *tuning.py       # 14 Optuna tuning scripts
│       ├── logs/
│       └── utils/
├── models/
│   ├── layer_1/             # Trained .pkl files (gitignored)
│   ├── layer_2/
│   └── layer_3/
├── notebooks/
│   ├── 1.0_exploratory_data_analysis.ipynb
│   ├── 2.0_feature_engineering.ipynb
│   ├── 3.0_machine_learning.ipynb
│   ├── 4.0_stacking_data_layer_one.ipynb
│   ├── 4.1_stacking_data_layer_two.ipynb
│   ├── 4.2_stacking_data_layer_two.ipynb
│   ├── 5.0_stacking_submission.ipynb    # LogisticRegression meta-model
│   ├── 5.1_stacking_submission.ipynb    # LightGBM meta-model
│   ├── 5.2_stacking_submission.ipynb
│   ├── 5.3_stacking_submission.ipynb
│   └── utils/
│       ├── load_model.py
│       └── calibration.py   # MulticlassThresholdOptimizer
├── results/
│   └── training_results.csv
└── catboost_info/
```

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13. Dependencies managed by `uv`. See `pyproject.toml` for full list.

## Usage

### 1. Train layer 1 (raw features)

```bash
python train_layer_1.py
```

Trains 7 registered models on `data/X_train_raw.parquet`. Skips existing `.pkl` files. Saves to `models/layer_1/`.

### 2. Train stacking layers

```bash
python train_layer_2.py
python train_layer_3.py
```

Layer 2 reads `X_train_stacking_layer_one.parquet`, layer 3 reads `X_train_stacking_layer_two.parquet`. Models train without TargetEncoder (stacking features are already numeric probabilities).

### 3. Generate stacking features

Use notebooks `4.0_stacking_data_layer_one.ipynb` through `4.2_stacking_data_layer_two.ipynb`:

- Load trained `.pkl` files from `models/<layer>/`
- Generate out-of-fold prediction probabilities via `cross_val_predict` (5-fold)
- Generate direct prediction probabilities for test data
- Save parquet files to `data/X_train_stacking_<layer>.parquet` and `data/X_test_stacking_<layer>.parquet`

### 4. Generate submission

Use notebooks `5.0_stacking_submission.ipynb`–`5.3_stacking_submission.ipynb` with the generated stacking features to train a meta-model and produce `submission_stacking_*.csv`.

## Model Details

All layers use the same unified model registry (7 registered models, 14 tuning scripts available). Layer 1 trains with `column_transformer` preprocessing. Layers 2+ skip encoders and train directly on stacking probabilities.

All models use Optuna with 90 trials, PR-AUC macro maximize, 5-fold StratifiedKFold CV, and MedianPruner.

### Registered Models (MODEL_REGISTRY)

| Model | n_trials | Layer 1 Pipeline |
|-------|----------|------------------|
| XGBoost | 90 | `TargetEncoder → XGBClassifier` |
| CatBoost | 90 | `CatBoostEncoder → CatBoostClassifier` |
| LightGBM | 90 | `TargetEncoder → LGBMClassifier` |
| ExtraTrees | 90 | `TargetEncoder → ExtraTreesClassifier` |
| RandomForest | 90 | `TargetEncoder → RandomForestClassifier` |
| HistGradientBoosting | 90 | `TargetEncoder → HistGradientBoostingClassifier` |
| LogisticRegression | 90 | `column_transformer → LogisticRegression` |

### Additional Tuning Scripts Available

| Model | n_trials | Layer 1 Pipeline |
|-------|----------|------------------|
| Ridge + Calibrated | 90 | `column_transformer → RidgeClassifier → CalibratedClassifierCV` |
| SGDClassifier | 90 | `column_transformer → SGDClassifier` |
| LinearSVC + Calibrated | 90 | `column_transformer → LinearSVC → CalibratedClassifierCV` |
| MLP | 90 | `column_transformer → MLPClassifier` |
| LDA | 90 | `column_transformer → LinearDiscriminantAnalysis` |
| QDA | 90 | `column_transformer → QuadraticDiscriminantAnalysis` |
| TruncatedSVD + KNN | 90 | `column_transformer → TruncatedSVD → StandardScaler → KNeighborsClassifier` |

### Preprocessing (`column_transformer`)

```python
ColumnTransformer([
    ("target_encoder", TargetEncoder(), ['spectral_type', 'galaxy_population']),
    ("standard_scaler", StandardScaler(), ['alpha', 'delta']),
], remainder=RobustScaler())
```

Used for raw features (layer 1). Tree-based models use `TargetEncoder` or `CatBoostEncoder` directly instead of `column_transformer`. Layers 2+ receive probability features; models train directly with no encoder.

## Key Features

- **N-layer stacking**: Per-layer training scripts (`train_layer_1.py`, `train_layer_2.py`, ...)
- **Separate packages**: `src/layer_one/` for raw features, `src/layer_stacking/` for stacking features
- **14 Optuna tuning scripts** with 5-fold StratifiedKFold CV, MedianPruner, PR-AUC maximization
- **7 active models** in `MODEL_REGISTRY` (extensible — add to registry to activate more)
- **Per-model logging** to `src/<layer>/logs/<model_name>.log`
- **Training results** saved to `results/training_results.csv`
- **Stacking features** generated via notebooks using `cross_val_predict`
- **MulticlassThresholdOptimizer** in `notebooks/utils/calibration.py` for weighted probability calibration
- **Skip existing models**: Rerun safely — already-trained models are skipped unless removed

## Adapting for a New Competition

1. Replace `data/train.csv`, `data/test.csv`
2. Run feature engineering notebook (2.0) to produce `X_train_raw.parquet`, `X_test_raw.parquet`
3. Update `src/layer_one/utils/preprocessing.py` if column names differ
4. Update target encoding columns in tuning files if column names differ
5. Run `python train_layer_1.py` to train all models
6. Run stacking notebooks to generate features for deeper layers
7. Update submission notebooks with new column names
