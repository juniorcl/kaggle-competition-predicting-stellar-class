from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select a subset of columns from a DataFrame.

    Can receive:

    - list of column names
    - boolean mask (style get_support())

    Examples
    --------

    FeatureSelector(
        features=["u", "g", "r"]
    )

    or

    FeatureSelector(
        mask=[True, False, True, ...]
    )
    """

    def __init__(self, features: Iterable[str] | None = None, mask: Iterable[bool] | None = None, check_missing: bool = True):

        self.features = features
        self.mask = mask
        self.check_missing = check_missing

    def fit(self, X: pd.DataFrame, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector espera um pandas.DataFrame.")

        if self.features is None and self.mask is None:
            raise ValueError("Informe 'features' ou 'mask'.")

        if self.features is not None and self.mask is not None:
            raise ValueError("Informe apenas 'features' OU 'mask'.")

        if self.mask is not None:
            mask = np.asarray(self.mask)

            if mask.dtype != bool:
                raise TypeError("'mask' should be boolean.")

            if len(mask) != X.shape[1]:
                raise ValueError("Mask has different size from the number of columns.")

            self.selected_features_ = X.columns[mask].tolist()

        else:
            self.selected_features_ = list(self.features)

        if self.check_missing:
            missing = sorted(set(self.selected_features_) - set(X.columns))

            if missing:
                raise ValueError(f"Features does not exist: {missing}")

        self.n_features_in_ = X.shape[1]

        self.feature_names_in_ = np.asarray(X.columns)

        return self

    def transform(self, X):
        return X.loc[:, self.selected_features_]

    def inverse_transform(self, X):

        out = pd.DataFrame(index=X.index, columns=self.feature_names_in_)

        out[self.selected_features_] = X

        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)

    def get_support(self, indices=False):

        mask = np.isin(self.feature_names_in_, self.selected_features_)

        if indices:
            return np.where(mask)[0]

        return mask

    @property
    def features_(self):
        return self.selected_features_

    def __len__(self):
        return len(self.selected_features_)

    def __repr__(self):
        return f"FeatureSelector("f"n_features={len(self)})"