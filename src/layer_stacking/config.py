
from .xgboost_tuning import tune_xgboost
from .catboost_tuning import tune_catboost
from .lightgbm_tuning import tune_lightgbm
from .sgdclassifier_tuning import tune_sgdclassifier
from .logistic_regression_tuning import tune_logistic_regression


MODEL_REGISTRY = {
    'xgboost': tune_xgboost,
    'catboost': tune_catboost,
    'lightgbm': tune_lightgbm,
    'sgdclassifier': tune_sgdclassifier,
    'logistic_regression': tune_logistic_regression,
}
