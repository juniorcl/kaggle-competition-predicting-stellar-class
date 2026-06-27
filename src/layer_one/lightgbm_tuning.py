import optuna

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('lightgbm')
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

            model = LGBMClassifier(
                objective='multiclass',
                metric='multi_logloss',
                boosting_type='gbdt',
                verbosity=-1,
                n_estimators=2000,
                random_state=42,
                n_jobs=1,
                num_leaves=trial.suggest_int('num_leaves', 16, 256),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                lambda_l1=trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                feature_fraction=trial.suggest_float('feature_fraction', 0.6, 1.0),
                bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
                bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 100),
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

    pipe_tuned = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        boosting_type='gbdt',
        verbosity=-1,
        n_estimators=2000,
        random_state=42,
        n_jobs=1,
        **study.best_params
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
