import optuna

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: str,
    n_trials: int = 90,
) -> None:

    logger = setup_model_logger("catboost")
    setup_optuna_logger(logger)

    logger.info("----- Model Tuning -----")

    #########################################################
    # Cat features
    #########################################################

    cat_features = [
        X_train.columns.get_loc("spectral_type"),
        X_train.columns.get_loc("galaxy_population"),
    ]

    #########################################################
    # CV criado uma única vez
    #########################################################

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    #########################################################
    # Pools criados uma única vez
    #########################################################

    folds = []

    for train_idx, valid_idx in cv.split(X_train, y_train):

        train_pool = Pool(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            cat_features=cat_features,
        )

        valid_pool = Pool(
            X_train.iloc[valid_idx],
            y_train.iloc[valid_idx],
            cat_features=cat_features,
        )

        folds.append(
            (
                train_pool,
                valid_pool,
                y_train.iloc[valid_idx].to_numpy(),
            )
        )

    #########################################################
    # Objective
    #########################################################

    def objective(trial):

        params = {

            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",

            "iterations": 1500,
            "early_stopping_rounds": 100,

            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,

            "boosting_type": "Plain",

            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights",
                [None, "Balanced", "SqrtBalanced"],
            ),

            "depth": trial.suggest_int(
                "depth",
                4,
                10,
            ),

            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf",
                1,
                100,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-3,
                0.2,
                log=True,
            ),

            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                1e-3,
                20,
                log=True,
            ),

            "random_strength": trial.suggest_float(
                "random_strength",
                1e-3,
                10,
                log=True,
            ),

            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0,
                10,
            ),

            "rsm": trial.suggest_float(
                "rsm",
                0.5,
                1.0,
            ),
        }

        scores = []

        for fold, (train_pool, valid_pool, y_valid) in enumerate(folds):

            model = CatBoostClassifier(**params)

            model.fit(
                train_pool,
                eval_set=valid_pool,
            )

            proba = model.predict_proba(valid_pool)

            score = log_loss(y_valid, proba)

            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    #########################################################
    # Study
    #########################################################

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=1,
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    logger.info(
        f"Best Log Loss: {study.best_value:.6f} | "
        f"Best params: {study.best_params}"
    )

    #########################################################
    # Modelo final
    #########################################################

    logger.info("----- Saving Pipeline -----")

    final_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1500,
        "early_stopping_rounds": 100,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        **study.best_params,
    }

    pipe_tuned = CatBoostClassifier(**final_params)

    pipe_tuned.fit(
        X_train,
        y_train,
        cat_features=cat_features,
    )

    dump_pickle(pipe_tuned, model_path)