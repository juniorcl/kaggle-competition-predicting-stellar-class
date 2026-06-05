import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["trunsvd_n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]
NYSTROEM_PARAMS = ["kernel", "gamma", "n_components"]
MODEL_PARAMS = ["loss", "penalty", "alpha", "learning_rate", "eta0", "l1_ratio", "average"]


def tune_trunsvd_nystroem_sgd(X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, target: str, model_path: str) -> tuple:

    logger = setup_model_logger('trunsvd_nystroem_sgd')
    setup_optuna_logger(logger)

    logger.info("----- Tuning TruncatedSVD + Nystroem + SGD -----")

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
                SGDClassifier(
                    loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                    penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                    alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                    learning_rate=trial.suggest_categorical("learning_rate", ["optimal", "adaptive"]),
                    eta0=trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    average=trial.suggest_categorical("average", [True, False]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    n_jobs=1
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)

            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train[target]), n_trials=60, n_jobs=-1, show_progress_bar=True, catch=(ValueError,))

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
        SGDClassifier(
            **best_model,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
            n_jobs=1
        )
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params
