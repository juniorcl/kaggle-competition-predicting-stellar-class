import optuna

import numpy as np
from optuna import trial
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .utils import column_transformer, dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('logistic_regression')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []

        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        C = trial.suggest_float("C", 1e-5, 100.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 5000)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = LogisticRegression(
                solver="saga",
                C=C,
                l1_ratio=l1_ratio,
                class_weight=class_weight,
                fit_intercept=fit_intercept,
                tol=tol,
                max_iter=max_iter,
                random_state=42,
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)

            score = log_loss(y_valid_fold, proba)
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best Log Loss: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    pipe_tuned = LogisticRegression(
        **study.best_trial.params,
        solver="saga",
        random_state=42,
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
