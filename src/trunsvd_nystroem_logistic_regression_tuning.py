import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["trunsvd_n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]
NYSTROEM_PARAMS = ["kernel", "gamma", "n_components"]
MODEL_PARAMS = ["C", "l1_ratio"]


def tune_trunsvd_nystroem_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, target: str, model_path: str) -> tuple:

    logger = setup_model_logger('trunsvd_nystroem_logistic_regression')
    setup_optuna_logger(logger)

    logger.info("----- Tuning TruncatedSVD + Nystroem + LogisticRegression -----")

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
                    n_components=trial.suggest_int("trunsvd_n_components", 2, 10),
                    algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                    n_iter=trial.suggest_int("n_iter", 3, 15),
                    power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                    tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True)
                ),
                StandardScaler(),
                Nystroem(
                    kernel=trial.suggest_categorical("kernel", ["rbf", "sigmoid", "cosine"]),
                    gamma=trial.suggest_float("gamma", 1e-5, 10.0, log=True),
                    n_components=trial.suggest_int("n_components", 64, 300),
                    random_state=42,
                ),
                StandardScaler(),
                LogisticRegression(
                    solver=trial.suggest_categorical("solver", ["saga"]),
                    C=trial.suggest_float("C", 1e-5, 100, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)

            score = log_loss(y_valid_fold, proba)
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train[target]), n_trials=60, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best LogLoss: {study.best_value} | Best params: {study.best_params}")

    logger.info("----- Saving Pipeline -----")

    best_svd = {k: v for k, v in study.best_params.items() if k in TRUNCATEDSVD_PARAMS}
    best_nystroem = {k: v for k, v in study.best_params.items() if k in NYSTROEM_PARAMS}
    best_model = {k: v for k, v in study.best_params.items() if k in MODEL_PARAMS}

    pipe_tuned = make_pipeline(
        column_transformer,
        TruncatedSVD(**best_svd),
        StandardScaler(),
        Nystroem(**best_nystroem, random_state=42),
        StandardScaler(),
        LogisticRegression(
            **best_model,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params
