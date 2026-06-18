import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle, column_transformer
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_sgdclassifier(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('sgdclassifier')
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
                column_transformer,
                SGDClassifier(
                    loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                    penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                    alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                    learning_rate=trial.suggest_categorical("learning_rate", ["optimal", "adaptive"]),
                    eta0=trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    tol=trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                    average=trial.suggest_categorical("average", [True, False]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    n_jobs=1
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)

            y_bin = label_binarize(y_valid_fold, classes=[0, 1, 2])
            score = average_precision_score(y_bin, proba, average='macro')
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=-1, show_progress_bar=True, catch=(ValueError,))

    logger.info(f"Best PR-AUC: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    pipe_tuned = make_pipeline(
        column_transformer,
        SGDClassifier(
            **study.best_trial.params,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
            n_jobs=1
        )
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
