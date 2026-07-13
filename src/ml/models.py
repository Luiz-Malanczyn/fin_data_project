from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


class NaiveFlatRegressor(BaseEstimator, RegressorMixin):
    """Predicts no change: the close `horizon_days` from now = today's close.

    The benchmark every other model has to beat. It gets easier to beat as
    the horizon grows -- a week or a month out, "nothing happens" is a much
    weaker guess than it is for tomorrow.
    """

    def __init__(self, anchor_column: str = "close_lag_0"):
        self.anchor_column = anchor_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveFlatRegressor":
        return self

    def predict(self, X: pd.DataFrame):
        return X[self.anchor_column].to_numpy()


class NaiveDriftRegressor(BaseEstimator, RegressorMixin):
    """Extrapolates the recent 5-day average daily return forward across
    the horizon, instead of guessing "no change". A second, slightly less
    naive baseline that actually moves -- useful because a completely flat
    prediction reads as broken even when it's the statistically safest bet.
    """

    def __init__(self, anchor_column: str = "close_lag_0", return_column: str = "return_5d", horizon_days: int = 1):
        self.anchor_column = anchor_column
        self.return_column = return_column
        self.horizon_days = horizon_days

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveDriftRegressor":
        return self

    def predict(self, X: pd.DataFrame):
        avg_daily_return = X[self.return_column].to_numpy() / 5
        return X[self.anchor_column].to_numpy() * (1 + avg_daily_return * self.horizon_days)


def build_model_registry(horizon_days: int) -> dict[str, Callable[[], BaseEstimator]]:
    """name -> zero-arg factory returning a fresh, unfitted estimator. A
    factory (not a shared instance) so cross-validation folds never reuse
    fit state. `horizon_days` is only used by NaiveDriftRegressor, which
    needs to know how many days of drift to extrapolate.

    Add a new model by registering another factory here -- train.py and the
    comparison report iterate this dict without any other changes.
    """
    return {
        "naive_flat": lambda: NaiveFlatRegressor(),
        "naive_drift": lambda: NaiveDriftRegressor(horizon_days=horizon_days),
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
