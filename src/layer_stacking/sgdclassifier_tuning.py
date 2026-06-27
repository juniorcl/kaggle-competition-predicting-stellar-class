import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from .utils import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def softmax(x):
    """Calcula as probabilidades Softmax de forma numericamente estável."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def tune_sgdclassifier(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90) -> None:

    logger = setup_model_logger('sgdclassifier')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    def objective(trial, X, y):
        
        loss = trial.suggest_categorical("loss", ["log_loss", "modified_huber", "hinge", "perceptron"])
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        alpha = trial.suggest_float("alpha", 1e-9, 1e-1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        learning_rate = trial.suggest_categorical("learning_rate", ["optimal", "adaptive", "constant", "invscaling"])
        eta0 = trial.suggest_float("eta0", 1e-5, 1.0, log=True)
        power_t = trial.suggest_float("power_t", 0.1, 0.5) if learning_rate == "invscaling" else 0.5
        requires_epsilon = loss in ["modified_huber", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
        epsilon = trial.suggest_float("epsilon", 1e-3, 1e-1, log=True) if requires_epsilon else 0.1
        tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
        average = trial.suggest_categorical("average", [True, False, 1, 5, 10])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        max_iter = trial.suggest_int("max_iter", 1000, 5000)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

            X_train_fold = X.iloc[train_idx, :]
            X_valid_fold = X.iloc[valid_idx, :]

            y_train_fold = y.iloc[train_idx]
            y_valid_fold = y.iloc[valid_idx]

            model = SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                learning_rate=learning_rate,
                eta0=eta0,
                power_t=power_t,
                l1_ratio=l1_ratio,
                epsilon=epsilon,
                tol=tol,
                average=average,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=42,
                n_jobs=1
            ).fit(X_train_fold, y_train_fold)

            if hasattr(model, "predict_proba") and loss in ["log_loss", "modified_huber"]:
                proba = model.predict_proba(X_valid_fold)
            
            else:
                decision = model.decision_function(X_valid_fold)
                
                if len(decision.shape) == 1:
                    decision = np.vstack([1 - decision, decision]).T
                
                proba = softmax(decision)

            proba = np.clip(proba, 1e-15, 1.0 - 1e-15)

            score = log_loss(y_valid_fold, proba)
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42), 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=-1, 
        show_progress_bar=True, 
        catch=(ValueError,)
    )

    logger.info(f"Best Log Loss: {study.best_value} | Best params: {study.best_params}")

    logger.info("----- Saving Pipeline -----")

    best_params = study.best_params.copy()
    
    if "power_t" not in best_params:
        best_params["power_t"] = 0.5
    
    if "epsilon" not in best_params:
        best_params["epsilon"] = 0.1

    pipe_tuned = SGDClassifier(
        **best_params,
        random_state=42,
        n_jobs=1
    ).fit(X_train, y_train)


    dump_pickle(pipe_tuned, model_path)
