from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


class NaiveLastValueRegressor(BaseEstimator, RegressorMixin):
    """Predicts no change: tomorrow's close = today's close.

    The benchmark every other model has to beat. If a "real" model can't
    beat this, it isn't adding value over just reading today's price.
    """

    def __init__(self, anchor_column: str = "close_lag_0"):
        self.anchor_column = anchor_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveLastValueRegressor":
        return self

    def predict(self, X: pd.DataFrame):
        return X[self.anchor_column].to_numpy()


# name -> zero-arg factory returning a fresh, unfitted estimator. A factory
# (not a shared instance) so cross-validation folds never reuse fit state.
# Add a new model by registering another factory here -- train.py and the
# comparison report iterate this dict without any other changes.
MODEL_REGISTRY: dict[str, Callable[[], BaseEstimator]] = {
    "naive": lambda: NaiveLastValueRegressor(),
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
    "xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1
    ),
}
