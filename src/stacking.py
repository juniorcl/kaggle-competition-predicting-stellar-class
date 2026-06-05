import os
import pickle
import logging

import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold


logger = logging.getLogger('model.stacking')


SHORT_NAMES = {
    "model_xgboost": "xgb",
    "model_catboost": "cat",
    "model_lightgbm": "lgbm",
    "model_extra_tree": "extra",
    "model_random_forest": "rf",
    "model_histgradientboosting": "hist",
    "model_sgdclassifier": "sgd",
    "model_logistic_regression": "lr",
    "model_ridge": "ridge",
    "trunsvd_xgboost": "tsvd_xgb",
    "trunsvd_catboost": "tsvd_cat",
    "trunsvd_lightgbm": "tsvd_lgbm",
    "trunsvd_extra_tree": "tsvd_extra",
    "trunsvd_random_forest": "tsvd_rf",
    "trunsvd_hist_gradient_boosting": "tsvd_hist",
    "trunsvd_sgdclassifier": "tsvd_sgd",
    "trunsvd_ridge": "tsvd_ridge",
    "trunsvd_logistic_regression": "tsvd_lr",
    "trunsvd_knn": "tsvd_knn",
    "trunsvd_nystroem_knn": "tsvd_ny_knn",
    "trunsvd_rbfsampler_knn": "tsvd_rbf_knn",
    "trunsvd_nystroem_sgd": "tsvd_ny_sgd",
    "trunsvd_rbfsampler_sgd": "tsvd_rbf_sgd",
    "trunsvd_nystroem_logistic_regression": "tsvd_ny_lr",
    "trunsvd_rbfsampler_logistic_regression": "tsvd_rbf_lr",
    "trunsvd_nystroem_ridge": "tsvd_ny_ridge",
    "trunsvd_rbfsampler_ridge": "tsvd_rbf_ridge",
    "stacking_sgd": "stk_sgd",
    "stacking_ridge": "stk_ridge",
    "stacking_logistic_regression": "stk_lr",
    "stacking_xgboost": "stk_xgb",
    "stacking_catboost": "stk_cat",
    "stacking_lightgbm": "stk_lgbm",
    "stacking_extra_tree": "stk_extra",
    "stacking_random_forest": "stk_rf",
    "stacking_hist_gradient_boosting": "stk_hist",
    "stacking_trunsvd_catboost": "stk_tsvd_cat",
    "stacking_trunsvd_xgboost": "stk_tsvd_xgb",
    "stacking_trunsvd_lightgbm": "stk_tsvd_lgbm",
    "stacking_trunsvd_extra_tree": "stk_tsvd_extra",
    "stacking_trunsvd_random_forest": "stk_tsvd_rf",
    "stacking_trunsvd_hist_gradient_boosting": "stk_tsvd_hist",
    "stacking_trunsvd_sgd": "stk_tsvd_sgd",
    "stacking_trunsvd_ridge": "stk_tsvd_ridge",
    "stacking_trunsvd_logistic_regression": "stk_tsvd_lr",
    "stacking_trunsvd_knn": "stk_tsvd_knn",
    "stacking_trunsvd_nystroem_knn": "stk_tsvd_ny_knn",
    "stacking_trunsvd_rbfsampler_knn": "stk_tsvd_rbf_knn",
    "stacking_trunsvd_nystroem_sgd": "stk_tsvd_ny_sgd",
    "stacking_trunsvd_rbfsampler_sgd": "stk_tsvd_rbf_sgd",
    "stacking_trunsvd_nystroem_logistic_regression": "stk_tsvd_ny_lr",
    "stacking_trunsvd_rbfsampler_logistic_regression": "stk_tsvd_rbf_lr",
    "stacking_trunsvd_nystroem_ridge": "stk_tsvd_ny_ridge",
    "stacking_trunsvd_rbfsampler_ridge": "stk_tsvd_rbf_ridge",
}


def _load_models(models_dir: str) -> list[tuple[str, object]]:
    models = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('.pkl'):
            continue
        if fname == 'label_encoder.pkl':
            continue
        stem = fname[:-4]
        fpath = os.path.join(models_dir, fname)
        try:
            with open(fpath, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            logger.warning("Failed loading %s: %s", fname, e)
            continue
        if not hasattr(model, 'predict_proba'):
            logger.warning("Skipping %s (no predict_proba)", fname)
            continue
        short_name = SHORT_NAMES.get(stem, stem)
        models.append((short_name, model))
        logger.info("Loaded %s -> %s (%s)", fname, short_name, type(model).__name__)
    if not models:
        raise ValueError(f"No valid models found in {models_dir}")
    return models


def generate_stacking_features(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    X_test: pd.DataFrame,
    models_dir: str,
    *,
    output_path_train: str | None = None,
    output_path_test: str | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    keep_classes: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if keep_classes is None:
        keep_classes = [0, 1]

    models = _load_models(models_dir)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    train_dfs, test_dfs = [], []

    for short_name, model in models:
        logger.info("Generating stacking features: %s", short_name)

        train_proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
        n_classes = train_proba.shape[1]

        test_proba = model.predict_proba(X_test)

        cols = [f"{short_name}_{i}" for i in keep_classes if i < n_classes]
        train_dfs.append(pd.DataFrame(train_proba[:, keep_classes], columns=cols, index=X_train.index))
        test_dfs.append(pd.DataFrame(test_proba[:, keep_classes], columns=cols, index=X_test.index))

    X_train_stack = pd.concat(train_dfs, axis=1)
    X_test_stack = pd.concat(test_dfs, axis=1)

    logger.info("Stacking features shape — train: %s, test: %s", X_train_stack.shape, X_test_stack.shape)

    if output_path_train:
        X_train_stack.to_parquet(output_path_train, index=True)
        logger.info("Saved: %s", output_path_train)
    if output_path_test:
        X_test_stack.to_parquet(output_path_test, index=True)
        logger.info("Saved: %s", output_path_test)

    return X_train_stack, X_test_stack
