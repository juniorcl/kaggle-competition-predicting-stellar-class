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
)

LAYER_ONE_MODELS = [
    {"name": "sgdclassifier", "func": tune_sgdclassifier, "model_path": "models/layer_one/model_sgdclassifier.pkl", "n_trials": 60},
    {"name": "ridge", "func": tune_ridge, "model_path": "models/layer_one/model_ridge.pkl", "n_trials": 60},
    {"name": "logistic_regression", "func": tune_logistic_regression, "model_path": "models/layer_one/model_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_catboost", "func": tune_trunsvd_catboost, "model_path": "models/layer_one/trunsvd_catboost.pkl", "n_trials": 30},
    {"name": "trunsvd_xgboost", "func": tune_trunsvd_xgboost, "model_path": "models/layer_one/trunsvd_xgboost.pkl", "n_trials": 30},
    {"name": "trunsvd_lightgbm", "func": tune_trunsvd_lightgbm, "model_path": "models/layer_one/trunsvd_lightgbm.pkl", "n_trials": 30},
    {"name": "trunsvd_extra_tree", "func": tune_trunsvd_extra_tree, "model_path": "models/layer_one/trunsvd_extra_tree.pkl", "n_trials": 30},
    {"name": "trunsvd_random_forest", "func": tune_trunsvd_random_forest, "model_path": "models/layer_one/trunsvd_random_forest.pkl", "n_trials": 30},
    {"name": "trunsvd_hist_gradient_boosting", "func": tune_trunsvd_hist_gradient_boosting, "model_path": "models/layer_one/trunsvd_hist_gradient_boosting.pkl", "n_trials": 30},
    {"name": "trunsvd_sgdclassifier", "func": tune_trunsvd_sgdclassifier, "model_path": "models/layer_one/trunsvd_sgdclassifier.pkl", "n_trials": 60},
    {"name": "trunsvd_ridge", "func": tune_trunsvd_ridge, "model_path": "models/layer_one/trunsvd_ridge.pkl", "n_trials": 60},
    {"name": "trunsvd_logistic_regression", "func": tune_trunsvd_logistic_regression, "model_path": "models/layer_one/trunsvd_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_knn", "func": tune_trunsvd_knn, "model_path": "models/layer_one/trunsvd_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_knn", "func": tune_trunsvd_nystroem_knn, "model_path": "models/layer_one/trunsvd_nystroem_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_knn", "func": tune_trunsvd_rbfsampler_knn, "model_path": "models/layer_one/trunsvd_rbfsampler_knn.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_sgd", "func": tune_trunsvd_nystroem_sgd, "model_path": "models/layer_one/trunsvd_nystroem_sgd.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_sgd", "func": tune_trunsvd_rbfsampler_sgd, "model_path": "models/layer_one/trunsvd_rbfsampler_sgd.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_logistic_regression", "func": tune_trunsvd_nystroem_logistic_regression, "model_path": "models/layer_one/trunsvd_nystroem_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_logistic_regression", "func": tune_trunsvd_rbfsampler_logistic_regression, "model_path": "models/layer_one/trunsvd_rbfsampler_logistic_regression.pkl", "n_trials": 60},
    {"name": "trunsvd_nystroem_ridge", "func": tune_trunsvd_nystroem_ridge, "model_path": "models/layer_one/trunsvd_nystroem_ridge.pkl", "n_trials": 60},
    {"name": "trunsvd_rbfsampler_ridge", "func": tune_trunsvd_rbfsampler_ridge, "model_path": "models/layer_one/trunsvd_rbfsampler_ridge.pkl", "n_trials": 60},
]

LAYER_TWO_MODELS = [
    {"name": "xgboost", "func": tune_xgboost, "model_path": "models/layer_two/model_xgboost.pkl", "n_trials": 30},
    {"name": "catboost", "func": tune_catboost, "model_path": "models/layer_two/model_catboost.pkl", "n_trials": 30},
    {"name": "lightgbm", "func": tune_lightgbm, "model_path": "models/layer_two/model_lightgbm.pkl", "n_trials": 30},
    {"name": "extra_tree", "func": tune_extra_tree, "model_path": "models/layer_two/model_extra_tree.pkl", "n_trials": 30},
    {"name": "random_forest", "func": tune_random_forest, "model_path": "models/layer_two/model_random_forest.pkl", "n_trials": 30},
    {"name": "hist_gradient_boosting", "func": tune_hist_gradient_boosting, "model_path": "models/layer_two/model_hist_gradient_boosting.pkl", "n_trials": 30},
]

LAYERS = {
    "layer_one": LAYER_ONE_MODELS,
    "layer_two": LAYER_TWO_MODELS,
}

MODEL_REGISTRY = LAYER_ONE_MODELS + LAYER_TWO_MODELS
