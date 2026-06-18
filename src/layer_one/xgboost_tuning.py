import os
import optuna

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from category_encoders import TargetEncoder

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


os.environ["XGBOOST_VERBOSITY"] = "0"


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('xgboost')
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

            model = make_pipeline(
                TargetEncoder(cols=['spectral_type', 'galaxy_population']),
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    verbosity=0,
                    enable_categorical=True,
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 1500),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    gamma=trial.suggest_float("gamma", 0, 5),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)

            y_bin = label_binarize(y_valid_fold, classes=[0, 1, 2])
            score = average_precision_score(y_bin, proba, average='macro')
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)


    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best PR-AUC: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    pipe_tuned = make_pipeline(
        TargetEncoder(cols=['spectral_type', 'galaxy_population']),
        XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            enable_categorical=True,
            **study.best_params
        )
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
