import optuna

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import column_transformer, dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_mlp(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('mlp')
    setup_optuna_logger(logger)

    logger.info("----- Fine Tuning -----")

    def objective(trial, X, y):

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        n_layers = trial.suggest_int("n_layers", 1, 2)

        hidden = tuple(
            trial.suggest_int(f"hidden_{i}", 16, 128, step=16)
            for i in range(n_layers)
        )

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                column_transformer,
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
                    alpha=trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                    learning_rate_init=trial.suggest_float("learning_rate_init", 1e-3, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
                    tol=trial.suggest_float("tol", 1e-4, 1e-3, log=True),
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
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


    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=4, show_progress_bar=True, catch=(ValueError,))

    logger.info(f"Best PR-AUC: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    best_params = study.best_params
    mlp_params = {k: v for k, v in best_params.items() if k != "n_layers"}
    hidden_sizes = tuple(mlp_params.pop(f"hidden_{i}") for i in range(best_params["n_layers"]))

    pipe_tuned = make_pipeline(
        column_transformer,
        MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            **mlp_params,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
        )
    ).fit(X_train, y_train)

    dump_pickle(pipe_tuned, model_path)
