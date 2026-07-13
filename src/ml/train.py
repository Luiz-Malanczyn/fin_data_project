"""Train and compare every registered model, at every horizon, for one (or
all active) investments.

Comparison uses TimeSeriesSplit cross-validation (never shuffled -- each
fold trains only on the past and validates on the future, like the model
will actually be used). Once compared, every model is refit on the full
history and persisted, so today's "second-best" model stays available for
later comparisons once more features are added -- not just the winner.

Three horizons are trained per asset: daily (next trading day), weekly
(~5 trading days for stocks / 7 calendar days for crypto), and monthly
(~21 trading days / 30 calendar days). A single day's "no change" baseline
is very hard to beat -- real price movement is mostly noise at that
distance -- but it gets much easier to beat further out, where genuine
trend/momentum outweighs day-to-day noise.

Usage:
    python -m src.ml.train PETR4
    python -m src.ml.train bitcoin crypto   # type only needed if not in the watchlist
    python -m src.ml.train                  # trains every active asset in the watchlist
"""
from __future__ import annotations

import logging
import sys

from sklearn.model_selection import TimeSeriesSplit

from src.config.watchlist_loader import load_watchlist
from src.ml.data import load_price_history, lookup_investment_type
from src.ml.features import HORIZON_STEPS, build_feature_frame
from src.ml.metrics import compute_metrics
from src.ml.models import build_model_registry
from src.ml.storage import save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5
BASELINE_COLUMN = "close_lag_0"
HORIZONS = ("daily", "weekly", "monthly")


def _cross_validate(model_factory, X, y) -> dict:
    """Average out-of-fold metrics across TimeSeriesSplit folds -- the
    number used to compare models against each other.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: list[dict] = []

    for train_idx, test_idx in tscv.split(X):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        fold_metrics.append(
            compute_metrics(
                y.iloc[test_idx], preds, baseline=X.iloc[test_idx][BASELINE_COLUMN]
            )
        )

    return {
        key: sum(fold[key] for fold in fold_metrics) / len(fold_metrics)
        for key in fold_metrics[0]
    }


def train_horizon(investment_id: str, investment_type: str, horizon: str) -> list[dict]:
    horizon_days = HORIZON_STEPS[investment_type][horizon]
    logger.info("[%s/%s] loading price history...", investment_id, horizon)
    history = load_price_history(investment_id, investment_type)
    X, y, _live_row = build_feature_frame(history, horizon_days=horizon_days)
    logger.info("[%s/%s] %d labeled rows, %d features", investment_id, horizon, len(X), X.shape[1])

    model_registry = build_model_registry(horizon_days)
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
        logger.info(
            "[%s/%s] %s -> MAE=%.4f RMSE=%.4f directional_accuracy=%.1f%%",
            investment_id,
            horizon,
            model_name,
            cv_metrics["mae"],
            cv_metrics["rmse"],
            cv_metrics["directional_accuracy"] * 100,
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
