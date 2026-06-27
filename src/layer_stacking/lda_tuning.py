import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .utils import column_transformer, dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_lda(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('lda')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []

        solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
        shrinkage = None
        tol = 0.0001

        if solver in ["lsqr", "eigen"]:
            shrinkage = trial.suggest_float("shrinkage", 0.0, 1.0)

        if solver == "svd":
            tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)

        params = {
            "solver": solver,
            "shrinkage": shrinkage,
            "tol": tol
        }

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):

            X_train_fold = X_train.iloc[train_idx, :]
            X_valid_fold = X_train.iloc[valid_idx, :]

            y_train_fold = y_train.iloc[train_idx]
            y_valid_fold = y_train.iloc[valid_idx]

            model = make_pipeline(
                column_transformer,
                LinearDiscriminantAnalysis(**params)
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

    pipe_tuned = make_pipeline(
        column_transformer,
        LinearDiscriminantAnalysis(**study.best_trial.params)
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
