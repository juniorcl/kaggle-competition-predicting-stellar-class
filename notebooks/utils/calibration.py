import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MulticlassThresholdOptimizer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_splits=5, method='Nelder-Mead', maxiter=500, random_state=42):
        self.n_splits = n_splits
        self.method = method
        self.maxiter = maxiter
        self.random_state = random_state
        
    def _objective(self, weights, X, y):
        weighted_probs = X * weights
        predictions = np.argmax(weighted_probs, axis=1)
        score = balanced_accuracy_score(y, predictions)
        return -score

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True, ensure_all_finite=True)
        
        self.n_classes_ = X.shape[1]
        
        if self.n_splits is None or self.n_splits <= 1:
            initial_weights = np.ones(self.n_classes_)
            res = minimize(self._objective, initial_weights, args=(X, y), method=self.method, options={'maxiter': self.maxiter})
            self.weights_ = res.x
            return self
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_weights = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            
            initial_weights = np.ones(self.n_classes_)
            res = minimize(self._objective, initial_weights, args=(X_train, y_train), method=self.method, options={'maxiter': self.maxiter})
            
            fold_weights.append(res.x)
        
        self.weights_ = np.mean(fold_weights, axis=0)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, attributes=['weights_'])
        X = check_array(X, ensure_2d=True)
        
        weighted_probs = X * self.weights_
        
        sum_probs = np.sum(weighted_probs, axis=1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1, sum_probs) 
        
        return weighted_probs / sum_probs

    def predict(self, X):
        check_is_fitted(self, attributes=['weights_'])
        X = check_array(X, ensure_2d=True)
        
        weighted_probs = X * self.weights_
        return np.argmax(weighted_probs, axis=1)