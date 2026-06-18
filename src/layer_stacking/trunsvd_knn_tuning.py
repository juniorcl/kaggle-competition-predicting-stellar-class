import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]


def tune_trunsvd_knn(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('trunsvd_knn')
    setup_optuna_logger(logger)


    logger.info("----- Tuning TruncatedSVD + KNN -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                TruncatedSVD(
                    n_components=trial.suggest_int("n_components", 2, 10),
                    algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                    n_iter=trial.suggest_int("n_iter", 3, 15),
                    power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                    tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True)
                ),
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    leaf_size=trial.suggest_int("leaf_size", 10, 50),
                    p=trial.suggest_int("p", 1, 2),
                    metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
                    n_jobs=1
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

    best_svd = {k: v for k, v in study.best_params.items() if k in TRUNCATEDSVD_PARAMS}
    best_knn = {k: v for k, v in study.best_params.items() if k not in TRUNCATEDSVD_PARAMS}

    pipe_tuned = make_pipeline(
        TruncatedSVD(**best_svd),
        StandardScaler(),
        KNeighborsClassifier(n_jobs=1, **best_knn)
    ).fit(X_train, y_train)

    dump_pickle(pipe_tuned, model_path)
