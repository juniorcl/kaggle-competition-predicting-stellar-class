import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]


def tune_trunsvd_ridge(X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, target: str, model_path: str) -> tuple:

    logger = setup_model_logger('trunsvd_ridge')
    setup_optuna_logger(logger)

    logger.info("----- Tuning TruncatedSVD + Ridge -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        solver = trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag", "saga"])

        class_weight_selection = trial.suggest_categorical("class_weight", ["none", "balanced"])
        class_weight_value = None if class_weight_selection == "none" else "balanced"

        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 100.0, log=True),
            "solver": solver,
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "class_weight": class_weight_value,
            "random_state": 42
        }

        if solver in ["lsqr", "sag", "saga"]:
            params["max_iter"] = trial.suggest_int("max_iter", 500, 5000, step=500)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = CalibratedClassifierCV(
                estimator=make_pipeline(
                    column_transformer,
                    TruncatedSVD(
                        n_components=trial.suggest_int("n_components", 2, 10),
                        algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                        n_iter=trial.suggest_int("n_iter", 3, 15),
                        power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                        tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True)
                    ),
                    StandardScaler(),
                    RidgeClassifier(**params)
                ),
                method='sigmoid',
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
    best_model = {k: v for k, v in study.best_params.items() if k not in TRUNCATEDSVD_PARAMS}

    tuned_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            column_transformer,
            TruncatedSVD(**best_svd),
            StandardScaler(),
            RidgeClassifier(**best_model, random_state=42)
        ),
        method='isotonic',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ).fit(X_train, y_train)

    dump_pickle(tuned_model, model_path)
    return study.best_value, study.best_params
