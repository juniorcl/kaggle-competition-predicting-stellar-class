from .mlp_tuning import tune_mlp
from .lda_tuning import tune_lda
from .qda_tuning import tune_qda
from .ridge_tuning import tune_ridge
from .xgboost_tuning import tune_xgboost
from .catboost_tuning import tune_catboost
from .lightgbm_tuning import tune_lightgbm
from .extra_tree_tuning import tune_extra_tree
from .linear_svc_tuning import tune_linear_svc
from .trunsvd_knn_tuning import tune_trunsvd_knn
from .random_forest_tuning import tune_random_forest
from .sgdclassifier_tuning import tune_sgdclassifier
from .logistic_regression_tuning import tune_logistic_regression
from .hist_gradient_boosting_tuning import tune_hist_gradient_boosting


MODEL_REGISTRY = {
    'xgboost': tune_xgboost,
    'catboost': tune_catboost,
    'lightgbm': tune_lightgbm,
}
