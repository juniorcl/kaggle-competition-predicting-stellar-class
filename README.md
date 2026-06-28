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
                     │  (3 registered models)│
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
                     │  (5 registered models)│
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │     ... layer N       │
                     └───────────────────────┘
```

### Layers

| Layer | Models | Trained on | Package |
|-------|--------|------------|---------|
| **layer_1** | 3 registered (14 tuning scripts available) | `X_train_fe.parquet` | `src/layer_one/` |
| **layer_N** | 5 registered (14 tuning scripts available) | `X_train_stacking_layer_{N-1}.parquet` | `src/layer_stacking/` |

Each layer can use a **different set of models**. Edit `MODEL_REGISTRY` in the layer's `config.py` to select which models to train — register from 14 available tuning scripts per layer.

Layer 1 trains on raw feature-engineered data with category dtype handling. Layers 2+ receive stacking probabilities (already numeric); models train directly with no encoding.

### Adding More Layers

Create `train_layer_N.py` pointing to the next stacking parquet:

```bash
# 1. Generate stacking features via notebook
# 2. Choose which models to register in src/layer_stacking/config.py
# 3. Create train script like train_layer_2.py but read layer_{N-1} data
python train_layer_N.py
```

## Project Structure

```
├── train_layer_1.py         # Train registered models on raw features → models/layer_1/
├── train_layer_2.py         # Train registered models on layer-1 stacking feats → models/layer_2/
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
│   ├── X_train_stacking_layer_{one,two}.parquet
│   ├── X_test_stacking_layer_{one,two}.parquet
│   └── submission_stacking_*.csv
├── src/
│   ├── layer_one/           # Raw-feature training package
│   │   ├── config.py        # MODEL_REGISTRY (select which models to train)
│   │   ├── *tuning.py       # 14 Optuna tuning scripts
│   │   ├── logs/            # Per-model training logs
│   │   └── utils/           # preprocessing, dump_model, logging_setup
│   └── layer_stacking/      # Stacking-feature training package
│       ├── config.py        # MODEL_REGISTRY (select models per layer)
│       ├── *tuning.py       # 14 Optuna tuning scripts
│       ├── logs/
│       └── utils/
├── models/
│   ├── layer_1/             # Trained .pkl files (gitignored)
│   ├── layer_2/
│   └── ...
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

### 1. Select models per layer

Edit `src/layer_one/config.py` and `src/layer_stacking/config.py` to choose which models to train. Each `MODEL_REGISTRY` dict maps model names to tuning functions — add or remove entries freely.

### 2. Train layer 1 (raw features)

```bash
python train_layer_1.py
```

Trains registered models on `data/X_train_fe.parquet`. Skips existing `.pkl` files. Saves to `models/layer_1/`.

### 3. Train stacking layers

```bash
python train_layer_2.py
```

Layer 2 reads `X_train_stacking_layer_one.parquet`. Stacking features are already numeric probabilities — models train directly.

### 4. Generate stacking features

Use notebooks `4.0_stacking_data_layer_one.ipynb` through `4.2_stacking_data_layer_two.ipynb`:

- Load trained `.pkl` files from `models/<layer>/`
- Generate out-of-fold prediction probabilities via `cross_val_predict` (5-fold)
- Generate direct prediction probabilities for test data
- Save parquet files to `data/X_train_stacking_<layer>.parquet` and `data/X_test_stacking_<layer>.parquet`

### 5. Generate submission

Use notebooks `5.0_stacking_submission.ipynb`–`5.3_stacking_submission.ipynb` with the generated stacking features to train a meta-model and produce `submission_stacking_*.csv`.

## Model Details

Each layer defines its own `MODEL_REGISTRY` in `config.py`. Layer 1 activates 3 models; stacking layers activate 5. You can freely change which models are registered per layer.

All tuning scripts use Optuna with 5-fold StratifiedKFold CV, MedianPruner, and log-loss minimization.

### Layer 1 Registered Models (MODEL_REGISTRY)

| Model | n_trials | Pipeline |
|-------|----------|----------|
| XGBoost | 60 | `XGBClassifier` (multi:softprob) |
| CatBoost | 60 | `CatBoostClassifier` (MultiClass, category dtype) |
| LightGBM | 60 | `LGBMClassifier` (multiclass) |

Layer 1 uses categorical dtype (`spectral_type`, `galaxy_population` → `category`) — CatBoost natively handles it; other models receive encoded dummies internally.

### Stacking Layer Registered Models (MODEL_REGISTRY)

| Model | Pipeline |
|-------|----------|
| XGBoost | `XGBClassifier` |
| CatBoost | `CatBoostClassifier` (no cat_features) |
| LightGBM | `LGBMClassifier` |
| SGDClassifier | `SGDClassifier` (custom softmax for non-probabilistic losses) |
| LogisticRegression | `LogisticRegression` (saga solver, elasticnet) |

Stacking layers train directly on probability features — no encoding needed.

### Additional Tuning Scripts Available

All 14 scripts exist per layer and can be activated by adding them to `MODEL_REGISTRY` in `config.py`:

| Model | Tuning Script |
|-------|--------------|
| XGBoost | `xgboost_tuning.py` |
| CatBoost | `catboost_tuning.py` |
| LightGBM | `lightgbm_tuning.py` |
| ExtraTrees | `extra_tree_tuning.py` |
| RandomForest | `random_forest_tuning.py` |
| HistGradientBoosting | `hist_gradient_boosting_tuning.py` |
| LogisticRegression | `logistic_regression_tuning.py` |
| Ridge + Calibrated | `ridge_tuning.py` |
| SGDClassifier | `sgdclassifier_tuning.py` |
| LinearSVC + Calibrated | `linear_svc_tuning.py` |
| MLP | `mlp_tuning.py` |
| LDA | `lda_tuning.py` |
| QDA | `qda_tuning.py` |
| TruncatedSVD + KNN | `trunsvd_knn_tuning.py` |

### Per-Layer Configuration

Each layer package has its own `config.py` with a `MODEL_REGISTRY` dict. To change which models train in a given layer:

```python
# src/layer_stacking/config.py
MODEL_REGISTRY = {
    'xgboost': tune_xgboost,
    'catboost': tune_catboost,
    'lightgbm': tune_lightgbm,
    'sgdclassifier': tune_sgdclassifier,
    'logistic_regression': tune_logistic_regression,
    # add more: 'random_forest': tune_random_forest, ...
}
```

Stacking layers can use different model sets than the raw-feature layer — pick the best mix for each stage.

## Key Features

- **N-layer stacking**: Per-layer training scripts (`train_layer_1.py`, `train_layer_2.py`, ...)
- **Separate packages**: `src/layer_one/` for raw features, `src/layer_stacking/` for stacking features
- **14 Optuna tuning scripts** with 5-fold StratifiedKFold CV, MedianPruner
- **Per-layer model selection**: Each layer's `MODEL_REGISTRY` picks which subset of the 14 scripts to train. Layer 1 uses 3 models; stacking layers use 5 — customize freely
- **Per-model logging** to `src/<layer>/logs/<model_name>.log`
- **Training results** saved to `results/training_results.csv`
- **Stacking features** generated via notebooks using `cross_val_predict`
- **MulticlassThresholdOptimizer** in `notebooks/utils/calibration.py` for weighted probability calibration
- **Skip existing models**: Rerun safely — already-trained models are skipped unless removed

## Adapting for a New Competition

1. Replace `data/train.csv`, `data/test.csv`
2. Run feature engineering notebook (2.0) to produce `X_train_fe.parquet`, `X_test_fe.parquet`
3. Update column references in tuning files if column names differ
4. Register desired models in `src/layer_one/config.py`
5. Run `python train_layer_1.py` to train selected models
6. Run stacking notebooks to generate features for deeper layers
7. Register stacking models in `src/layer_stacking/config.py`
8. Run `python train_layer_2.py`
9. Update submission notebooks with new column names
