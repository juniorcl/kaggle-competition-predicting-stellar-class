from .xgboost_tuning import tune_xgboost
from .catboost_tuning import tune_catboost
from .lightgbm_tuning import tune_lightgbm
from .extra_tree_tuning import tune_extra_tree
from .random_forest_tuning import tune_random_forest
from .hist_gradient_boosting_tuning import tune_hist_gradient_boosting

from .ridge_tuning import tune_ridge
from .sgdclassifier_tuning import tune_sgdclassifier
from .logistic_regression_tuning import tune_logistic_regression

from .trunsvd_catboost_tuning import tune_trunsvd_catboost
from .trunsvd_xgboost_tuning import tune_trunsvd_xgboost
from .trunsvd_lightgbm_tuning import tune_trunsvd_lightgbm
from .trunsvd_extra_tree_tuning import tune_trunsvd_extra_tree
from .trunsvd_random_forest_tuning import tune_trunsvd_random_forest
from .trunsvd_hist_gradient_boosting_tuning import tune_trunsvd_hist_gradient_boosting
from .trunsvd_sgdclassifier_tuning import tune_trunsvd_sgdclassifier
from .trunsvd_ridge_tuning import tune_trunsvd_ridge
from .trunsvd_logistic_regression_tuning import tune_trunsvd_logistic_regression
from .trunsvd_knn_tuning import tune_trunsvd_knn
from .trunsvd_nystroem_knn_tuning import tune_trunsvd_nystroem_knn
from .trunsvd_rbfsampler_knn_tuning import tune_trunsvd_rbfsampler_knn
from .trunsvd_nystroem_sgd_tuning import tune_trunsvd_nystroem_sgd
from .trunsvd_rbfsampler_sgd_tuning import tune_trunsvd_rbfsampler_sgd
from .trunsvd_nystroem_logistic_regression_tuning import tune_trunsvd_nystroem_logistic_regression
from .trunsvd_rbfsampler_logistic_regression_tuning import tune_trunsvd_rbfsampler_logistic_regression
from .trunsvd_nystroem_ridge_tuning import tune_trunsvd_nystroem_ridge
from .trunsvd_rbfsampler_ridge_tuning import tune_trunsvd_rbfsampler_ridge

from .stacking_tuning import (
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

from .stacking import generate_stacking_features

from .config import NUM_LAYERS, MODEL_REGISTRY, RAW_MODEL_REGISTRY, STACKING_MODEL_REGISTRY, LAYERS


__all__ = [
    "tune_xgboost",
    "tune_catboost",
    "tune_lightgbm",
    "tune_extra_tree",
    "tune_random_forest",
    "tune_hist_gradient_boosting",

    "tune_ridge",
    "tune_sgdclassifier",
    "tune_logistic_regression",

    "tune_trunsvd_catboost",
    "tune_trunsvd_xgboost",
    "tune_trunsvd_lightgbm",
    "tune_trunsvd_extra_tree",
    "tune_trunsvd_random_forest",
    "tune_trunsvd_hist_gradient_boosting",
    "tune_trunsvd_sgdclassifier",
    "tune_trunsvd_ridge",
    "tune_trunsvd_logistic_regression",
    "tune_trunsvd_knn",
    "tune_trunsvd_nystroem_knn",
    "tune_trunsvd_rbfsampler_knn",
    "tune_trunsvd_nystroem_sgd",
    "tune_trunsvd_rbfsampler_sgd",
    "tune_trunsvd_nystroem_logistic_regression",
    "tune_trunsvd_rbfsampler_logistic_regression",
    "tune_trunsvd_nystroem_ridge",
    "tune_trunsvd_rbfsampler_ridge",

    "tune_stacking_sgd",
    "tune_stacking_ridge",
    "tune_stacking_logistic_regression",
    "tune_stacking_xgboost",
    "tune_stacking_catboost",
    "tune_stacking_lightgbm",
    "tune_stacking_extra_tree",
    "tune_stacking_random_forest",
    "tune_stacking_hist_gradient_boosting",
    "tune_stacking_trunsvd_catboost",
    "tune_stacking_trunsvd_xgboost",
    "tune_stacking_trunsvd_lightgbm",
    "tune_stacking_trunsvd_extra_tree",
    "tune_stacking_trunsvd_random_forest",
    "tune_stacking_trunsvd_hist_gradient_boosting",
    "tune_stacking_trunsvd_sgd",
    "tune_stacking_trunsvd_ridge",
    "tune_stacking_trunsvd_logistic_regression",
    "tune_stacking_trunsvd_knn",
    "tune_stacking_trunsvd_nystroem_knn",
    "tune_stacking_trunsvd_rbfsampler_knn",
    "tune_stacking_trunsvd_nystroem_sgd",
    "tune_stacking_trunsvd_rbfsampler_sgd",
    "tune_stacking_trunsvd_nystroem_logistic_regression",
    "tune_stacking_trunsvd_rbfsampler_logistic_regression",
    "tune_stacking_trunsvd_nystroem_ridge",
    "tune_stacking_trunsvd_rbfsampler_ridge",

    "generate_stacking_features",

    "NUM_LAYERS",
    "MODEL_REGISTRY",
    "RAW_MODEL_REGISTRY",
    "STACKING_MODEL_REGISTRY",
    "LAYERS",
]
