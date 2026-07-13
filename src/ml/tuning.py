"""Hyperparameter search for the tree-based models.

Run once per (asset, horizon) before the main model comparison in
train.py, using the same time-respecting cross-validation discipline as
everywhere else in this module (TimeSeriesSplit, never shuffled). The
random_forest/gradient_boosting/xgboost hyperparameters used until now
were reasonable-looking defaults, not the result of any search.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

N_ITER = 8
N_SPLITS = 3

PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
}


def tune_tree_hyperparams(X, y) -> dict[str, dict]:
    """Returns {model_name: best_params} for the three tree models, chosen
    by a randomized search minimizing MAE on the (return-space) target.
    """
    base_estimators = {
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "xgboost": XGBRegressor(random_state=42, n_jobs=-1),
    }

    tuned: dict[str, dict] = {}
    for name, estimator in base_estimators.items():
        search = RandomizedSearchCV(
            estimator,
            PARAM_DISTRIBUTIONS[name],
            n_iter=N_ITER,
            cv=TimeSeriesSplit(n_splits=N_SPLITS),
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        tuned[name] = search.best_params_

    return tuned
