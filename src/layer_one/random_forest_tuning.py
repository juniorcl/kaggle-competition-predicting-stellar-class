import optuna

import numpy as np
import pandas as pd

from category_encoders import TargetEncoder

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('random_forest')
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
                TargetEncoder(cols=["spectral_type", "galaxy_population"]),
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
        TargetEncoder(cols=["spectral_type", "galaxy_population"]),
        RandomForestClassifier(random_state=42, n_jobs=-1, **study.best_params),
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
