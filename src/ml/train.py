"""Train and compare every registered model, at every horizon, for one (or
all active) investments.

Comparison uses TimeSeriesSplit cross-validation (never shuffled -- each
fold trains only on the past and validates on the future, like the model
will actually be used). Once compared, every model is refit on the full
history and persisted, so today's "second-best" model stays available for
later comparisons once more features are added -- not just the winner.

Three horizons are trained per asset: daily (next trading day), weekly
(~5 trading days for stocks / 7 calendar days for crypto), and monthly
(~21 trading days / 30 calendar days). The target is the *return* over the
horizon, not the raw future price -- see features.build_feature_frame for
why that matters. Tree hyperparameters are tuned per (asset, horizon) via
tuning.tune_tree_hyperparams before the comparison runs, and directional
accuracy is checked against a 50/50 coin flip with a binomial test so a
"58% accuracy" claim isn't taken at face value without knowing if it's
statistically distinguishable from luck.

Usage:
    python -m src.ml.train PETR4
    python -m src.ml.train bitcoin crypto   # type only needed if not in the watchlist
    python -m src.ml.train                  # trains every active asset in the watchlist
"""
from __future__ import annotations

import logging
import sys

from scipy.stats import binomtest
from sklearn.model_selection import TimeSeriesSplit

from src.config.watchlist_loader import load_watchlist
from src.ml.data import load_price_history, lookup_investment_type
from src.ml.features import HORIZON_STEPS, build_feature_frame
from src.ml.metrics import compute_metrics
from src.ml.models import build_model_registry
from src.ml.storage import save_model
from src.ml.tuning import tune_tree_hyperparams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5
BASELINE_COLUMN = "close_lag_0"
HORIZONS = ("daily", "weekly", "monthly")


def _cross_validate(model_factory, X, y) -> dict:
    """Average out-of-fold metrics across TimeSeriesSplit folds -- the
    numbers used to compare models against each other -- plus a binomial
    test of directional accuracy against a 50/50 coin flip, pooled across
    all folds' correct-call counts (not averaged per-fold p-values, which
    wouldn't be statistically meaningful).
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: list[dict] = []
    total_correct = 0
    total_n = 0

    for train_idx, test_idx in tscv.split(X):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        fold = compute_metrics(
            y.iloc[test_idx], preds, baseline_close=X.iloc[test_idx][BASELINE_COLUMN]
        )
        fold_metrics.append(fold)
        total_correct += round(fold["directional_accuracy"] * fold["n_samples"])
        total_n += fold["n_samples"]

    aggregated = {
        key: sum(fold[key] for fold in fold_metrics) / len(fold_metrics)
        for key in fold_metrics[0]
    }

    test_result = binomtest(total_correct, total_n, p=0.5, alternative="greater")
    aggregated["directional_accuracy_pvalue"] = float(test_result.pvalue)
    aggregated["directional_accuracy_significant"] = bool(test_result.pvalue < 0.05)

    return aggregated


def train_horizon(investment_id: str, investment_type: str, horizon: str) -> list[dict]:
    horizon_days = HORIZON_STEPS[investment_type][horizon]
    logger.info("[%s/%s] loading price history...", investment_id, horizon)
    history = load_price_history(investment_id, investment_type)
    X, y, _live_row = build_feature_frame(history, horizon_days=horizon_days)
    logger.info("[%s/%s] %d labeled rows, %d features", investment_id, horizon, len(X), X.shape[1])

    logger.info("[%s/%s] tuning tree hyperparameters...", investment_id, horizon)
    tuned_params = tune_tree_hyperparams(X, y)

    model_registry = build_model_registry(horizon_days, tuned_params=tuned_params)
    results = []
    for model_name, model_factory in model_registry.items():
        cv_metrics = _cross_validate(model_factory, X, y)

        # Comparison uses the CV metrics (out-of-fold, not overfit to the
        # full history); the persisted model is then refit on everything so
        # it's as current as possible for live predictions.
        final_model = model_factory()
        final_model.fit(X, y)
        save_model(investment_id, horizon, model_name, final_model, cv_metrics, list(X.columns))

        results.append({"model_name": model_name, **cv_metrics})
        sig = "significant" if cv_metrics["directional_accuracy_significant"] else "not significant"
        logger.info(
            "[%s/%s] %s -> MAE=%.5f directional_accuracy=%.1f%% (p=%.3f, %s)",
            investment_id,
            horizon,
            model_name,
            cv_metrics["mae"],
            cv_metrics["directional_accuracy"] * 100,
            cv_metrics["directional_accuracy_pvalue"],
            sig,
        )

    results.sort(key=lambda r: r["mae"])
    logger.info("[%s/%s] best model: %s", investment_id, horizon, results[0]["model_name"])
    return results


def train_and_compare(investment_id: str, investment_type: str = "stock") -> dict[str, list[dict]]:
    return {horizon: train_horizon(investment_id, investment_type, horizon) for horizon in HORIZONS}


def train_all_active_assets() -> dict[str, dict[str, list[dict]]]:
    """Trains every active asset in the watchlist, at every horizon,
    regardless of investment type -- stocks and crypto go through the
    exact same pipeline.
    """
    assets = [a for a in load_watchlist() if a.active]
    return {asset.id: train_and_compare(asset.id, asset.type) for asset in assets}


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_all_active_assets()
    elif len(sys.argv) == 2:
        train_and_compare(sys.argv[1], lookup_investment_type(sys.argv[1]))
    else:
        train_and_compare(sys.argv[1], sys.argv[2])
