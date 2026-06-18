import optuna

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from .utils import column_transformer, dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_linear_svc(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('linear_svc')
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
                    LinearSVC(
                        penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
                        loss=trial.suggest_categorical("loss", ["squared_hinge"]),
                        C=trial.suggest_float("C", 1e-3, 100, log=True),
                        tol=trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                        multi_class="ovr",
                        class_weight="balanced",
                        random_state=42,
                        max_iter=5000,
                        dual="auto",
                    ),
                ),
                method="isotonic",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
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

    tuned_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            column_transformer,
            LinearSVC(
                **study.best_params,
                multi_class="ovr",
                class_weight="balanced",
                random_state=42,
                max_iter=5000,
                dual="auto"
            ),
        ),
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ).fit(X_train, y_train)


    dump_pickle(tuned_model, model_path)
