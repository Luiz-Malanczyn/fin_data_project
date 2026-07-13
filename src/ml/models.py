from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Fallbacks used when no tuned hyperparameters are supplied for a tree
# model (see tuning.py) -- reasonably conservative defaults, not the result
# of any search.
DEFAULT_TREE_PARAMS: dict[str, dict] = {
    "random_forest": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3},
    "gradient_boosting": {},
    "xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
}


class NaiveFlatRegressor(BaseEstimator, RegressorMixin):
    """Predicts zero return: no change over the horizon.

    In return space this is a constant 0.0 -- unlike predicting raw price,
    where a model could win just by copying today's close, a real model now
    has to commit to a nonzero delta if it wants to beat this.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveFlatRegressor":
        return self

    def predict(self, X: pd.DataFrame):
        return np.zeros(len(X))


class NaiveDriftRegressor(BaseEstimator, RegressorMixin):
    """Extrapolates the recent 5-day average daily return forward across
    the horizon, instead of guessing zero. A second, slightly less naive
    baseline that actually moves -- useful because a completely flat
    prediction reads as broken even when it's the statistically safest bet.
    """

    def __init__(self, return_column: str = "return_5d", horizon_days: int = 1):
        self.return_column = return_column
        self.horizon_days = horizon_days

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveDriftRegressor":
        return self

    def predict(self, X: pd.DataFrame):
        avg_daily_return = X[self.return_column].to_numpy() / 5
        return avg_daily_return * self.horizon_days


def build_model_registry(
    horizon_days: int, tuned_params: dict[str, dict] | None = None
) -> dict[str, Callable[[], BaseEstimator]]:
    """name -> zero-arg factory returning a fresh, unfitted estimator. A
    factory (not a shared instance) so cross-validation folds never reuse
    fit state.

    Linear models are wrapped in a StandardScaler pipeline: features mix
    wildly different scales (a price lag ~R$40 next to a return ~0.01),
    which was producing ill-conditioned-matrix warnings and unstable
    coefficients (Ridge/LinearRegression need standardized inputs to
    regularize sensibly). Tree models are scale-invariant and skip it.

    `tuned_params` (from tuning.py's search) overrides DEFAULT_TREE_PARAMS
    per tree model when provided.

    Add a new model by registering another factory here -- train.py and the
    comparison report iterate this dict without any other changes.
    """
    params = {**DEFAULT_TREE_PARAMS, **(tuned_params or {})}

    return {
        "naive_flat": lambda: NaiveFlatRegressor(),
        "naive_drift": lambda: NaiveDriftRegressor(horizon_days=horizon_days),
        "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42)),
        "random_forest": lambda: RandomForestRegressor(
            random_state=42, n_jobs=-1, **params["random_forest"]
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            random_state=42, **params["gradient_boosting"]
        ),
        "xgboost": lambda: XGBRegressor(random_state=42, n_jobs=-1, **params["xgboost"]),
    }
