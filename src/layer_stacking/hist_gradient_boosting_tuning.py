import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('hist_gradient_boosting')
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

            model = HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 100),
                l2_regularization=trial.suggest_float("l2_regularization", 0.0, 10.0),
                max_bins=trial.suggest_int("max_bins", 64, 255),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=42,
                class_weight='balanced',
                verbose=0
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

    pipe_tuned = HistGradientBoostingClassifier(
        **study.best_params,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
        class_weight='balanced',
        verbose=0
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
