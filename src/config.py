from . import (
    tune_xgboost,
    tune_catboost,
    tune_lightgbm,
    tune_extra_tree,
    tune_random_forest,
    tune_hist_gradient_boosting,
    tune_sgdclassifier,
    tune_ridge,
    tune_logistic_regression,
    tune_trunsvd_catboost,
    tune_trunsvd_xgboost,
    tune_trunsvd_lightgbm,
    tune_trunsvd_extra_tree,
    tune_trunsvd_random_forest,
    tune_trunsvd_hist_gradient_boosting,
    tune_trunsvd_sgdclassifier,
    tune_trunsvd_ridge,
    tune_trunsvd_logistic_regression,
    tune_trunsvd_knn,
    tune_trunsvd_nystroem_knn,
    tune_trunsvd_rbfsampler_knn,
    tune_trunsvd_nystroem_sgd,
    tune_trunsvd_rbfsampler_sgd,
    tune_trunsvd_nystroem_logistic_regression,
    tune_trunsvd_rbfsampler_logistic_regression,
    tune_trunsvd_nystroem_ridge,
    tune_trunsvd_rbfsampler_ridge,

    tune_stacking_sgd,
    tune_stacking_ridge,
    tune_stacking_logistic_regression,
    tune_stacking_xgboost,
    tune_stacking_catboost,
    tune_stacking_lightgbm,
    tune_stacking_extra_tree,
    tune_stacking_random_forest,
    tune_stacking_hist_gradient_boosting,
    tune_stacking_trunsvd_catboost,
    tune_stacking_trunsvd_xgboost,
    tune_stacking_trunsvd_lightgbm,
    tune_stacking_trunsvd_extra_tree,
    tune_stacking_trunsvd_random_forest,
    tune_stacking_trunsvd_hist_gradient_boosting,
    tune_stacking_trunsvd_sgd,
    tune_stacking_trunsvd_ridge,
    tune_stacking_trunsvd_logistic_regression,
    tune_stacking_trunsvd_knn,
    tune_stacking_trunsvd_nystroem_knn,
    tune_stacking_trunsvd_rbfsampler_knn,
    tune_stacking_trunsvd_nystroem_sgd,
    tune_stacking_trunsvd_rbfsampler_sgd,
    tune_stacking_trunsvd_nystroem_logistic_regression,
    tune_stacking_trunsvd_rbfsampler_logistic_regression,
    tune_stacking_trunsvd_nystroem_ridge,
    tune_stacking_trunsvd_rbfsampler_ridge,
)

# ---------------------------------------------------------------------------
# Number of stacking layers
# ---------------------------------------------------------------------------
# Change this to add more layers (layer_3, layer_4, ...).
# Each layer beyond layer_1 trains on stacking features from the previous layer.
NUM_LAYERS = 2

# ---------------------------------------------------------------------------
# Layer 1: models that train on raw features (with preprocessing)
# ---------------------------------------------------------------------------
RAW_MODEL_REGISTRY = [
    {"name": "sgdclassifier", "func": tune_sgdclassifier, "model_path": "models/layer_1/model_sgdclassifier.pkl", "n_trials": 60},
    {"name": "ridge", "func": tune_ridge, "model_path": "models/layer_1/model_ridge.pkl", "n_trials": 60},
    {"name": "logistic_regression", "func": tune_logistic_regression, "model_path": "models/layer_1/model_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_catboost", "func": tune_trunsvd_catboost, "model_path": "models/layer_1/trunsvd_catboost.pkl", "n_trials": 30},
    {"name": "trunsvd_xgboost", "func": tune_trunsvd_xgboost, "model_path": "models/layer_1/trunsvd_xgboost.pkl", "n_trials": 30},
    {"name": "trunsvd_lightgbm", "func": tune_trunsvd_lightgbm, "model_path": "models/layer_1/trunsvd_lightgbm.pkl", "n_trials": 30},
    {"name": "trunsvd_extra_tree", "func": tune_trunsvd_extra_tree, "model_path": "models/layer_1/trunsvd_extra_tree.pkl", "n_trials": 30},
    {"name": "trunsvd_random_forest", "func": tune_trunsvd_random_forest, "model_path": "models/layer_1/trunsvd_random_forest.pkl", "n_trials": 30},
    {"name": "trunsvd_hist_gradient_boosting", "func": tune_trunsvd_hist_gradient_boosting, "model_path": "models/layer_1/trunsvd_hist_gradient_boosting.pkl", "n_trials": 30},
    {"name": "trunsvd_sgdclassifier", "func": tune_trunsvd_sgdclassifier, "model_path": "models/layer_1/trunsvd_sgdclassifier.pkl", "n_trials": 60},
    {"name": "trunsvd_ridge", "func": tune_trunsvd_ridge, "model_path": "models/layer_1/trunsvd_ridge.pkl", "n_trials": 60},
    {"name": "trunsvd_logistic_regression", "func": tune_trunsvd_logistic_regression, "model_path": "models/layer_1/trunsvd_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_knn", "func": tune_trunsvd_knn, "model_path": "models/layer_1/trunsvd_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_knn", "func": tune_trunsvd_nystroem_knn, "model_path": "models/layer_1/trunsvd_nystroem_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_knn", "func": tune_trunsvd_rbfsampler_knn, "model_path": "models/layer_1/trunsvd_rbfsampler_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_sgd", "func": tune_trunsvd_nystroem_sgd, "model_path": "models/layer_1/trunsvd_nystroem_sgd.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_sgd", "func": tune_trunsvd_rbfsampler_sgd, "model_path": "models/layer_1/trunsvd_rbfsampler_sgd.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_logistic_regression", "func": tune_trunsvd_nystroem_logistic_regression, "model_path": "models/layer_1/trunsvd_nystroem_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_logistic_regression", "func": tune_trunsvd_rbfsampler_logistic_regression, "model_path": "models/layer_1/trunsvd_rbfsampler_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_ridge", "func": tune_trunsvd_nystroem_ridge, "model_path": "models/layer_1/trunsvd_nystroem_ridge.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_ridge", "func": tune_trunsvd_rbfsampler_ridge, "model_path": "models/layer_1/trunsvd_rbfsampler_ridge.pkl", "n_trials": 60},
]

# ---------------------------------------------------------------------------
# Layers 2+: models that train on stacking features (no preprocessing)
# ---------------------------------------------------------------------------
STACKING_MODEL_REGISTRY = [
    {"name": "stacking_sgd", "func": tune_stacking_sgd, "model_path": "models/layer_2/stacking_sgd.pkl", "n_trials": 60},
    {"name": "stacking_ridge", "func": tune_stacking_ridge, "model_path": "models/layer_2/stacking_ridge.pkl", "n_trials": 60},
    {"name": "stacking_logistic_regression", "func": tune_stacking_logistic_regression, "model_path": "models/layer_2/stacking_logistic_regression.pkl", "n_trials": 60},
    {"name": "stacking_xgboost", "func": tune_stacking_xgboost, "model_path": "models/layer_2/stacking_xgboost.pkl", "n_trials": 30},
    {"name": "stacking_catboost", "func": tune_stacking_catboost, "model_path": "models/layer_2/stacking_catboost.pkl", "n_trials": 30},
    {"name": "stacking_lightgbm", "func": tune_stacking_lightgbm, "model_path": "models/layer_2/stacking_lightgbm.pkl", "n_trials": 30},
    {"name": "stacking_extra_tree", "func": tune_stacking_extra_tree, "model_path": "models/layer_2/stacking_extra_tree.pkl", "n_trials": 30},
    {"name": "stacking_random_forest", "func": tune_stacking_random_forest, "model_path": "models/layer_2/stacking_random_forest.pkl", "n_trials": 30},
    {"name": "stacking_hist_gradient_boosting", "func": tune_stacking_hist_gradient_boosting, "model_path": "models/layer_2/stacking_hist_gradient_boosting.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_catboost", "func": tune_stacking_trunsvd_catboost, "model_path": "models/layer_2/stacking_trunsvd_catboost.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_xgboost", "func": tune_stacking_trunsvd_xgboost, "model_path": "models/layer_2/stacking_trunsvd_xgboost.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_lightgbm", "func": tune_stacking_trunsvd_lightgbm, "model_path": "models/layer_2/stacking_trunsvd_lightgbm.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_extra_tree", "func": tune_stacking_trunsvd_extra_tree, "model_path": "models/layer_2/stacking_trunsvd_extra_tree.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_random_forest", "func": tune_stacking_trunsvd_random_forest, "model_path": "models/layer_2/stacking_trunsvd_random_forest.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_hist_gradient_boosting", "func": tune_stacking_trunsvd_hist_gradient_boosting, "model_path": "models/layer_2/stacking_trunsvd_hist_gradient_boosting.pkl", "n_trials": 30},
    {"name": "stacking_trunsvd_sgd", "func": tune_stacking_trunsvd_sgd, "model_path": "models/layer_2/stacking_trunsvd_sgd.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_ridge", "func": tune_stacking_trunsvd_ridge, "model_path": "models/layer_2/stacking_trunsvd_ridge.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_logistic_regression", "func": tune_stacking_trunsvd_logistic_regression, "model_path": "models/layer_2/stacking_trunsvd_logistic_regression.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_knn", "func": tune_stacking_trunsvd_knn, "model_path": "models/layer_2/stacking_trunsvd_knn.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_nystroem_knn", "func": tune_stacking_trunsvd_nystroem_knn, "model_path": "models/layer_2/stacking_trunsvd_nystroem_knn.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_rbfsampler_knn", "func": tune_stacking_trunsvd_rbfsampler_knn, "model_path": "models/layer_2/stacking_trunsvd_rbfsampler_knn.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_nystroem_sgd", "func": tune_stacking_trunsvd_nystroem_sgd, "model_path": "models/layer_2/stacking_trunsvd_nystroem_sgd.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_rbfsampler_sgd", "func": tune_stacking_trunsvd_rbfsampler_sgd, "model_path": "models/layer_2/stacking_trunsvd_rbfsampler_sgd.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_nystroem_logistic_regression", "func": tune_stacking_trunsvd_nystroem_logistic_regression, "model_path": "models/layer_2/stacking_trunsvd_nystroem_logistic_regression.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_rbfsampler_logistic_regression", "func": tune_stacking_trunsvd_rbfsampler_logistic_regression, "model_path": "models/layer_2/stacking_trunsvd_rbfsampler_logistic_regression.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_nystroem_ridge", "func": tune_stacking_trunsvd_nystroem_ridge, "model_path": "models/layer_2/stacking_trunsvd_nystroem_ridge.pkl", "n_trials": 60},
    {"name": "stacking_trunsvd_rbfsampler_ridge", "func": tune_stacking_trunsvd_rbfsampler_ridge, "model_path": "models/layer_2/stacking_trunsvd_rbfsampler_ridge.pkl", "n_trials": 60},
]

# ---------------------------------------------------------------------------
# Auto-generate layer definitions
# ---------------------------------------------------------------------------
# layer_1  → RAW_MODEL_REGISTRY (raw features with preprocessing)
# layer_2+ → STACKING_MODEL_REGISTRY (probabilities, no preprocessing)
LAYERS = {}
for i in range(1, NUM_LAYERS + 1):
    registry = STACKING_MODEL_REGISTRY if i > 1 else RAW_MODEL_REGISTRY
    LAYERS[f"layer_{i}"] = registry

MODEL_REGISTRY = RAW_MODEL_REGISTRY
