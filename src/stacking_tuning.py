import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


TRUNCATEDSVD_PARAMS = ["trunsvd_n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]
NYSTROEM_PARAMS = ["kernel", "gamma", "n_components"]
RBFSAMPLER_PARAMS = ["gamma", "n_components"]


def _run_stacking_study(name: str, objective_fn, X_train, y_train, *, n_trials: int = 60, n_jobs: int = -1):
    logger = setup_model_logger(name)
    setup_optuna_logger(logger)
    logger.info("----- Tuning %s -----", name)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train),
        n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True,
        catch=(ValueError,),
    )

    logger.info("Best LogLoss: %s | Best params: %s", study.best_value, study.best_params)
    logger.info("----- Saving Pipeline -----")
    return logger, study


# ---------------------------------------------------------------------------
# Native linear
# ---------------------------------------------------------------------------

def tune_stacking_sgd(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                    penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                    alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                    learning_rate=trial.suggest_categorical("learning_rate", ["optimal", "adaptive"]),
                    eta0=trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    average=trial.suggest_categorical("average", [True, False]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    n_jobs=1,
                )
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)

            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_sgd", objective, X_train, y_train[target], n_trials=60)

    pipe_tuned = make_pipeline(
        StandardScaler(),
        SGDClassifier(**study.best_params, class_weight="balanced", max_iter=5000, random_state=42, n_jobs=1)
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_ridge(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        solver = trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag", "saga"])
        class_weight_selection = trial.suggest_categorical("class_weight", ["none", "balanced"])
        class_weight_value = None if class_weight_selection == "none" else "balanced"

        ridge_params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 100.0, log=True),
            "solver": solver,
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "class_weight": class_weight_value,
            "random_state": 42,
        }
        if solver in ["lsqr", "sag", "saga"]:
            ridge_params["max_iter"] = trial.suggest_int("max_iter", 500, 5000, step=500)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = CalibratedClassifierCV(
                estimator=make_pipeline(StandardScaler(), RidgeClassifier(**ridge_params)),
                method="sigmoid",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_ridge", objective, X_train, y_train[target], n_trials=60)

    best_ridge = {k: v for k, v in study.best_params.items() if k != "max_iter"}
    tuned_model = CalibratedClassifierCV(
        estimator=make_pipeline(StandardScaler(), RidgeClassifier(**best_ridge, random_state=42)),
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ).fit(X_train, y_train)

    dump_pickle(tuned_model, model_path)
    return study.best_value, study.best_params


def tune_stacking_logistic_regression(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver=trial.suggest_categorical("solver", ["saga"]),
                    C=trial.suggest_float("C", 1e-5, 100, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_logistic_regression", objective, X_train, y_train[target], n_trials=60)

    pipe_tuned = make_pipeline(
        StandardScaler(),
        LogisticRegression(**study.best_params, class_weight="balanced", random_state=42, max_iter=1000),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


# ---------------------------------------------------------------------------
# Native tree
# ---------------------------------------------------------------------------

def tune_stacking_xgboost(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = XGBClassifier(
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
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_xgboost", objective, X_train, y_train[target], n_trials=30)

    pipe_tuned = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
        enable_categorical=True,
        **study.best_params,
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_catboost(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=3000,
                od_type="Iter",
                od_wait=150,
                random_state=42,
                verbose=0,
                auto_class_weights=trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"]),
                depth=trial.suggest_int("depth", 4, 10),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
                random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                rsm=trial.suggest_float("rsm", 0.5, 1.0),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_catboost", objective, X_train, y_train[target], n_trials=30, n_jobs=2)

    pipe_tuned = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=3000,
        od_type="Iter",
        od_wait=150,
        random_state=42,
        verbose=0,
        **study.best_params,
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_lightgbm(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = LGBMClassifier(
                objective="multiclass",
                metric="multi_logloss",
                boosting_type="gbdt",
                verbosity=-1,
                n_estimators=2000,
                random_state=42,
                n_jobs=1,
                num_leaves=trial.suggest_int("num_leaves", 16, 256),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_lightgbm", objective, X_train, y_train[target], n_trials=30)

    pipe_tuned = LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        boosting_type="gbdt",
        verbosity=-1,
        n_estimators=2000,
        random_state=42,
        n_jobs=1,
        **study.best_params,
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_extra_tree(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 5, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 150),
                max_features=trial.suggest_float("max_features", 0.1, 0.4),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_extra_tree", objective, X_train, y_train[target], n_trials=30)

    pipe_tuned = ExtraTreesClassifier(
        class_weight="balanced", random_state=42, n_jobs=1, **study.best_params
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_random_forest(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = RandomForestClassifier(
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
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_random_forest", objective, X_train, y_train[target], n_trials=30)

    pipe_tuned = RandomForestClassifier(random_state=42, n_jobs=-1, **study.best_params).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_hist_gradient_boosting(X_train, y_train, target, model_path):
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
                class_weight="balanced",
                verbose=0,
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_hist_gradient_boosting", objective, X_train, y_train[target], n_trials=30)

    pipe_tuned = HistGradientBoostingClassifier(
        **study.best_params,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
        class_weight="balanced",
        verbose=0,
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


# ---------------------------------------------------------------------------
# TruncatedSVD helpers
# ---------------------------------------------------------------------------

def _suggest_svd(trial):
    return {
        "n_components": trial.suggest_int("trunsvd_n_components", 2, 10),
        "algorithm": trial.suggest_categorical("algorithm", ["randomized"]),
        "n_iter": trial.suggest_int("n_iter", 3, 15),
        "power_iteration_normalizer": trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
    }


def _suggest_nystroem(trial):
    return {
        "kernel": trial.suggest_categorical("kernel", ["rbf", "sigmoid", "cosine"]),
        "gamma": trial.suggest_float("gamma", 1e-5, 10.0, log=True),
        "n_components": trial.suggest_int("n_components", 64, 300),
    }


def _suggest_rbfsampler(trial):
    return {
        "gamma": trial.suggest_float("gamma", 1e-5, 10.0, log=True),
        "n_components": trial.suggest_int("n_components", 64, 300),
    }


def _extract_svd(params):
    return {k: v for k, v in params.items() if k in TRUNCATEDSVD_PARAMS}


def _extract_nystroem(params):
    return {k: v for k, v in params.items() if k in NYSTROEM_PARAMS}


def _extract_rbfsampler(params):
    return {k: v for k, v in params.items() if k in RBFSAMPLER_PARAMS}


def _extract_rest(params, *keysets):
    excluded = set()
    for ks in keysets:
        excluded.update(ks)
    return {k: v for k, v in params.items() if k not in excluded}


# ---------------------------------------------------------------------------
# TruncatedSVD + tree
# ---------------------------------------------------------------------------

def tune_stacking_trunsvd_catboost(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                CatBoostClassifier(
                    loss_function="MultiClass", eval_metric="MultiClass",
                    iterations=3000, od_type="Iter", od_wait=150,
                    random_state=42, verbose=0,
                    auto_class_weights=trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"]),
                    depth=trial.suggest_int("depth", 4, 10),
                    min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
                    random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    rsm=trial.suggest_float("rsm", 0.5, 1.0),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_catboost", objective, X_train, y_train[target], n_trials=30, n_jobs=2)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        CatBoostClassifier(
            loss_function="MultiClass", eval_metric="MultiClass",
            iterations=3000, od_type="Iter", od_wait=150,
            random_state=42, verbose=0, **best_tree,
        ),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_xgboost(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                XGBClassifier(
                    objective="multi:softprob", eval_metric="mlogloss",
                    verbosity=0, enable_categorical=True,
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 1500),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    gamma=trial.suggest_float("gamma", 0, 5),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_xgboost", objective, X_train, y_train[target], n_trials=30)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            verbosity=0, enable_categorical=True, **best_tree,
        ),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_lightgbm(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                LGBMClassifier(
                    objective="multiclass", metric="multi_logloss",
                    boosting_type="gbdt", verbosity=-1, n_estimators=2000,
                    random_state=42, n_jobs=1,
                    num_leaves=trial.suggest_int("num_leaves", 16, 256),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                    lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                    feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
                    bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                    bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_lightgbm", objective, X_train, y_train[target], n_trials=30)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        LGBMClassifier(
            objective="multiclass", metric="multi_logloss",
            boosting_type="gbdt", verbosity=-1, n_estimators=2000,
            random_state=42, n_jobs=1, **best_tree,
        ),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_extra_tree(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                ExtraTreesClassifier(
                    class_weight="balanced", random_state=42, n_jobs=1,
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 5, 15),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 150),
                    max_features=trial.suggest_float("max_features", 0.1, 0.4),
                    criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_extra_tree", objective, X_train, y_train[target], n_trials=30)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=1, **best_tree),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_random_forest(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                RandomForestClassifier(
                    random_state=42, n_jobs=1,
                    n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                    max_depth=trial.suggest_int("max_depth", 8, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 50),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                    bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                    class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_random_forest", objective, X_train, y_train[target], n_trials=30)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        RandomForestClassifier(random_state=42, n_jobs=1, **best_tree),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_hist_gradient_boosting(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]
            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                HistGradientBoostingClassifier(
                    class_weight="balanced", max_iter=1000,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=50, random_state=42, verbose=0,
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 100),
                    l2_regularization=trial.suggest_float("l2_regularization", 0.0, 10.0),
                    max_bins=trial.suggest_int("max_bins", 64, 255),
                ),
            ).fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_hist_gradient_boosting", objective, X_train, y_train[target], n_trials=30)

    best_svd = _extract_svd(study.best_params)
    best_tree = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=1000,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, random_state=42, verbose=0, **best_tree,
        ),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


# ---------------------------------------------------------------------------
# TruncatedSVD + linear
# ---------------------------------------------------------------------------

def tune_stacking_trunsvd_sgd(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                SGDClassifier(
                    loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                    penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                    alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                    learning_rate=trial.suggest_categorical("learning_rate", ["optimal", "adaptive"]),
                    eta0=trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    average=trial.suggest_categorical("average", [True, False]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    n_jobs=1,
                ),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)
            score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_sgd", objective, X_train, y_train[target], n_trials=60)

    best_svd = _extract_svd(study.best_params)
    best_model = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        SGDClassifier(**best_model, class_weight="balanced", max_iter=5000, random_state=42, n_jobs=1),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_ridge(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)

        solver = trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag", "saga"])
        cw_selection = trial.suggest_categorical("class_weight", ["none", "balanced"])
        cw_value = None if cw_selection == "none" else "balanced"
        ridge_params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 100.0, log=True),
            "solver": solver,
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "class_weight": cw_value,
            "random_state": 42,
        }
        if solver in ["lsqr", "sag", "saga"]:
            ridge_params["max_iter"] = trial.suggest_int("max_iter", 500, 5000, step=500)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = CalibratedClassifierCV(
                estimator=make_pipeline(
                    StandardScaler(),
                    TruncatedSVD(**svd_params),
                    RidgeClassifier(**ridge_params),
                ),
                method="sigmoid",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_ridge", objective, X_train, y_train[target], n_trials=60)

    best_svd = _extract_svd(study.best_params)
    best_ridge = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    best_ridge.pop("max_iter", None)

    tuned_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            TruncatedSVD(**best_svd),
            RidgeClassifier(**best_ridge, random_state=42),
        ),
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    ).fit(X_train, y_train)

    dump_pickle(tuned_model, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_logistic_regression(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                LogisticRegression(
                    solver="saga",
                    C=trial.suggest_float("C", 1e-5, 100, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ).fit(X_train_fold, y_train_fold)

            proba = model.predict_proba(X_valid_fold)
            score = log_loss(y_valid_fold, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    logger, study = _run_stacking_study("stacking_trunsvd_logistic_regression", objective, X_train, y_train[target], n_trials=60)

    best_svd = _extract_svd(study.best_params)
    best_model = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        LogisticRegression(**best_model, solver="saga", class_weight="balanced", random_state=42, max_iter=1000),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


def tune_stacking_trunsvd_knn(X_train, y_train, target, model_path):
    def objective(trial, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        svd_params = _suggest_svd(trial)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]
            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = make_pipeline(
                StandardScaler(),
                TruncatedSVD(**svd_params),
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    leaf_size=trial.suggest_int("leaf_size", 10, 50),
                    p=trial.suggest_int("p", 1, 2),
                    metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
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

    logger, study = _run_stacking_study("stacking_trunsvd_knn", objective, X_train, y_train[target], n_trials=60)

    best_svd = _extract_svd(study.best_params)
    best_knn = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
    pipe_tuned = make_pipeline(
        StandardScaler(),
        TruncatedSVD(**best_svd),
        StandardScaler(),
        KNeighborsClassifier(n_jobs=1, **best_knn),
    ).fit(X_train, y_train[target])

    dump_pickle(pipe_tuned, model_path)
    return study.best_value, study.best_params


# ---------------------------------------------------------------------------
# Kernel approximation helpers
# ---------------------------------------------------------------------------

def _make_stacking_kernel_knn_tuner(name, kernel_class, suggest_kernel_fn, extract_kernel_fn, n_trials):
    def tuner(X_train, y_train, target, model_path):
        def objective(trial, X, y):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            svd_params = _suggest_svd(trial)
            kernel_params = suggest_kernel_fn(trial)
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X.iloc[train_idx, :]
                X_valid_fold = X.iloc[valid_idx, :]
                y_train_fold = y.iloc[train_idx]
                y_valid_fold = y.iloc[valid_idx]

                model = make_pipeline(
                    StandardScaler(),
                    TruncatedSVD(**svd_params),
                    StandardScaler(),
                    kernel_class(**kernel_params, random_state=42),
                    StandardScaler(),
                    KNeighborsClassifier(
                        n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
                        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                        leaf_size=trial.suggest_int("leaf_size", 10, 50),
                        p=trial.suggest_int("p", 1, 2),
                        metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
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

        logger, study = _run_stacking_study(name, objective, X_train, y_train[target], n_trials=n_trials)

        best_svd = _extract_svd(study.best_params)
        best_kernel = extract_kernel_fn(study.best_params)
        best_knn = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
        best_knn = _extract_rest(best_knn, NYSTROEM_PARAMS if kernel_class is Nystroem else RBFSAMPLER_PARAMS)

        pipe_tuned = make_pipeline(
            StandardScaler(),
            TruncatedSVD(**best_svd),
            StandardScaler(),
            kernel_class(**best_kernel, random_state=42),
            StandardScaler(),
            KNeighborsClassifier(n_jobs=1, **best_knn),
        ).fit(X_train, y_train[target])

        dump_pickle(pipe_tuned, model_path)
        return study.best_value, study.best_params

    return tuner


tune_stacking_trunsvd_nystroem_knn = _make_stacking_kernel_knn_tuner(
    "stacking_trunsvd_nystroem_knn", Nystroem, _suggest_nystroem, _extract_nystroem, n_trials=60,
)

tune_stacking_trunsvd_rbfsampler_knn = _make_stacking_kernel_knn_tuner(
    "stacking_trunsvd_rbfsampler_knn", RBFSampler, _suggest_rbfsampler, _extract_rbfsampler, n_trials=60,
)


def _make_stacking_kernel_linear_tuner(name, model_class, suggest_kernel_fn, extract_kernel_fn, model_fixed_params, n_trials):
    def tuner(X_train, y_train, target, model_path):
        def objective(trial, X, y):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            svd_params = _suggest_svd(trial)
            kernel_params = suggest_kernel_fn(trial)
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X.iloc[train_idx, :]
                X_valid_fold = X.iloc[valid_idx, :]
                y_train_fold = y.iloc[train_idx]
                y_valid_fold = y.iloc[valid_idx]

                model = make_pipeline(
                    StandardScaler(),
                    TruncatedSVD(**svd_params),
                    StandardScaler(),
                    kernel_class(**kernel_params, random_state=42),
                    StandardScaler(),
                    model_class(**model_fixed_params),
                ).fit(X_train_fold, y_train_fold)

                proba = model.predict_proba(X_valid_fold)
                proba = np.nan_to_num(proba, nan=1e-15, posinf=1.0 - 1e-15, neginf=1e-15)
                score = log_loss(y_valid_fold, proba, labels=[0, 1, 2])
                scores.append(score)
                trial.report(np.mean(scores), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return np.mean(scores)

        logger, study = _run_stacking_study(name, objective, X_train, y_train[target], n_trials=n_trials)

        best_svd = _extract_svd(study.best_params)
        best_kernel = extract_kernel_fn(study.best_params)
        pipe_tuned = make_pipeline(
            StandardScaler(),
            TruncatedSVD(**best_svd),
            StandardScaler(),
            kernel_class(**best_kernel, random_state=42),
            StandardScaler(),
            model_class(**model_fixed_params),
        ).fit(X_train, y_train[target])

        dump_pickle(pipe_tuned, model_path)
        return study.best_value, study.best_params

    return tuner


def _make_stacking_kernel_ridge_tuner(name, kernel_class, suggest_kernel_fn, extract_kernel_fn, n_trials):
    def tuner(X_train, y_train, target, model_path):
        def objective(trial, X, y):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            svd_params = _suggest_svd(trial)
            kernel_params = suggest_kernel_fn(trial)

            solver = trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag", "saga"])
            cw_selection = trial.suggest_categorical("class_weight", ["none", "balanced"])
            cw_value = None if cw_selection == "none" else "balanced"
            ridge_params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 100.0, log=True),
                "solver": solver,
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "class_weight": cw_value,
                "random_state": 42,
            }
            if solver in ["lsqr", "sag", "saga"]:
                ridge_params["max_iter"] = trial.suggest_int("max_iter", 500, 5000, step=500)

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X.iloc[train_idx, :]
                X_valid_fold = X.iloc[valid_idx, :]
                y_train_fold = y.iloc[train_idx]
                y_valid_fold = y.iloc[valid_idx]

                model = CalibratedClassifierCV(
                    estimator=make_pipeline(
                        StandardScaler(),
                        TruncatedSVD(**svd_params),
                        StandardScaler(),
                        kernel_class(**kernel_params, random_state=42),
                        StandardScaler(),
                        RidgeClassifier(**ridge_params),
                    ),
                    method="sigmoid",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                ).fit(X_train_fold, y_train_fold)

                proba = model.predict_proba(X_valid_fold)
                score = log_loss(y_valid_fold, proba)
                scores.append(score)
                trial.report(np.mean(scores), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return np.mean(scores)

        logger, study = _run_stacking_study(name, objective, X_train, y_train[target], n_trials=n_trials)

        best_svd = _extract_svd(study.best_params)
        best_kernel = extract_kernel_fn(study.best_params)
        best_ridge = _extract_rest(study.best_params, TRUNCATEDSVD_PARAMS)
        best_ridge = _extract_rest(best_ridge, NYSTROEM_PARAMS if kernel_class is Nystroem else RBFSAMPLER_PARAMS)
        best_ridge.pop("max_iter", None)

        tuned_model = CalibratedClassifierCV(
            estimator=make_pipeline(
                StandardScaler(),
                TruncatedSVD(**best_svd),
                StandardScaler(),
                kernel_class(**best_kernel, random_state=42),
                StandardScaler(),
                RidgeClassifier(**best_ridge, random_state=42),
            ),
            method="isotonic",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        ).fit(X_train, y_train)

        dump_pickle(tuned_model, model_path)
        return study.best_value, study.best_params

    return tuner


_linear_fixed = {
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 0.0001,
    "learning_rate": "optimal",
    "eta0": 0.01,
    "l1_ratio": 0.15,
    "average": False,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42,
    "n_jobs": 1,
}

tune_stacking_trunsvd_nystroem_sgd = _make_stacking_kernel_linear_tuner(
    "stacking_trunsvd_nystroem_sgd", SGDClassifier, _suggest_nystroem, _extract_nystroem, _linear_fixed, n_trials=60,
)

tune_stacking_trunsvd_rbfsampler_sgd = _make_stacking_kernel_linear_tuner(
    "stacking_trunsvd_rbfsampler_sgd", SGDClassifier, _suggest_rbfsampler, _extract_rbfsampler, _linear_fixed, n_trials=60,
)

_logistic_fixed = {
    "solver": "saga",
    "C": 1.0,
    "l1_ratio": 0.5,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}

tune_stacking_trunsvd_nystroem_logistic_regression = _make_stacking_kernel_linear_tuner(
    "stacking_trunsvd_nystroem_logistic_regression", LogisticRegression, _suggest_nystroem, _extract_nystroem, _logistic_fixed, n_trials=60,
)

tune_stacking_trunsvd_rbfsampler_logistic_regression = _make_stacking_kernel_linear_tuner(
    "stacking_trunsvd_rbfsampler_logistic_regression", LogisticRegression, _suggest_rbfsampler, _extract_rbfsampler, _logistic_fixed, n_trials=60,
)

tune_stacking_trunsvd_nystroem_ridge = _make_stacking_kernel_ridge_tuner(
    "stacking_trunsvd_nystroem_ridge", Nystroem, _suggest_nystroem, _extract_nystroem, n_trials=60,
)

tune_stacking_trunsvd_rbfsampler_ridge = _make_stacking_kernel_ridge_tuner(
    "stacking_trunsvd_rbfsampler_ridge", RBFSampler, _suggest_rbfsampler, _extract_rbfsampler, n_trials=60,
)
