import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]


def tune_trunsvd_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, target: str, model_path: str) -> tuple:

    logger = setup_model_logger('trunsvd_random_forest')
    setup_optuna_logger(logger)

    logger.info("----- Tuning TruncatedSVD + Random Forest -----")

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
                RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                    max_depth=trial.suggest_int("max_depth", 8, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 50),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                    bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                    class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
                    random_state=42,
                    n_jobs=1,
                ),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)

            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train[target]), n_trials=30, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best LogLoss: {study.best_value} | Best params: {study.best_params}")

    logger.info("----- Saving Pipeline -----")

    best_svd = {k: v for k, v in study.best_params.items() if k in TRUNCATEDSVD_PARAMS}
    best_model = {k: v for k, v in study.best_params.items() if k not in TRUNCATEDSVD_PARAMS}

    pipe_tuned = make_pipeline(
        column_transformer,
        TruncatedSVD(**best_svd),
        RandomForestClassifier(random_state=42, n_jobs=-1, **best_model),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params
