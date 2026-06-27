import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import column_transformer, dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_ridge(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90 ) -> None:

    logger = setup_model_logger('ridge')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = CalibratedClassifierCV(
                estimator=make_pipeline(
                    column_transformer,
                    RidgeClassifier(
                        alpha=trial.suggest_float("alpha", 1e-5, 100.0, log=True),
                        solver=trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag", "saga"]),
                        tol=trial.suggest_float("tol", 1e-5, 1e-1, log=True),
                        fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                        class_weight="balanced",
                        random_state=42
                    )
                ),
                method='isotonic',
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
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best Log Loss: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    tuned_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            column_transformer,
            RidgeClassifier(
                **study.best_params,
                class_weight="balanced",
                random_state=42
            )
        ),
        method='isotonic',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ).fit(X_train, y_train)


    dump_pickle(tuned_model, model_path)
