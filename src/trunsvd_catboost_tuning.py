import optuna

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_trunsvd_catboost(X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, target: str, model_path: str) -> tuple:

    logger = setup_model_logger('trunsvd_catboost')
    setup_optuna_logger(logger)

    logger.info("----- Model Tuning -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                column_transformer,
                TruncatedSVD(
                    n_components=trial.suggest_int("n_components", 2, 10),
                    algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                    n_iter=trial.suggest_int("n_iter", 3, 15),
                    power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                    tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True)
                ),
                CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    iterations=3000,
                    od_type="Iter",
                    od_wait=150,
                    random_state=42,
                    verbose=0,
                    auto_class_weights=trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"]),
                    boosting_type=trial.suggest_categorical("boosting_type", ["Plain"]),
                    depth=trial.suggest_int("depth", 4, 10),
                    min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
                    random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    rsm=trial.suggest_float("rsm", 0.5, 1.0),
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)

            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train[target]), n_trials=30, n_jobs=2, show_progress_bar=True)

    logger.info(f"Best LogLoss: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    truncatedsvd_params = ["n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]

    best_truncatesvd_params = {k: v for k, v in study.best_params.items() if k in truncatedsvd_params}
    best_cat_params = {k: v for k, v in study.best_params.items() if k not in truncatedsvd_params}

    pipe_tuned = make_pipeline(
        column_transformer,
        TruncatedSVD(**best_truncatesvd_params),
        CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=3000,
            od_type="Iter",
            od_wait=150,
            random_state=42,
            verbose=0,
            **best_cat_params
        )
    ).fit(X_train, y_train[target])


    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params
