"""Train and compare every registered model for one (or all active) stocks.

Comparison uses TimeSeriesSplit cross-validation (never shuffled -- each
fold trains only on the past and validates on the future, like the model
will actually be used). Once compared, every model is refit on the full
history and persisted, so today's "second-best" model stays available for
later comparisons once more features are added -- not just the winner.

Usage:
    python -m src.ml.train PETR4
    python -m src.ml.train          # trains every active stock in the watchlist
"""
from __future__ import annotations

import logging
import sys

from sklearn.model_selection import TimeSeriesSplit

from src.config.watchlist_loader import filter_by_type, load_watchlist
from src.ml.data import load_price_history
from src.ml.features import build_feature_frame
from src.ml.metrics import compute_metrics
from src.ml.models import MODEL_REGISTRY
from src.ml.storage import save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5
BASELINE_COLUMN = "close_lag_0"


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


def train_and_compare(investment_id: str) -> list[dict]:
    logger.info("Loading price history for %s...", investment_id)
    history = load_price_history(investment_id)
    X, y, _live_row = build_feature_frame(history)
    logger.info("%d labeled rows, %d features", len(X), X.shape[1])

    results = []
    for model_name, model_factory in MODEL_REGISTRY.items():
        logger.info("[%s] cross-validating %s...", investment_id, model_name)
        cv_metrics = _cross_validate(model_factory, X, y)

        # Comparison uses the CV metrics (out-of-fold, not overfit to the
        # full history); the persisted model is then refit on everything so
        # it's as current as possible for live predictions.
        final_model = model_factory()
        final_model.fit(X, y)
        save_model(investment_id, model_name, final_model, cv_metrics, list(X.columns))

        results.append({"model_name": model_name, **cv_metrics})
        logger.info(
            "[%s] %s -> MAE=%.4f RMSE=%.4f directional_accuracy=%.1f%%",
            investment_id,
            model_name,
            cv_metrics["mae"],
            cv_metrics["rmse"],
            cv_metrics["directional_accuracy"] * 100,
        )

    results.sort(key=lambda r: r["mae"])
    logger.info("[%s] best model: %s", investment_id, results[0]["model_name"])
    return results


def train_all_active_stocks() -> dict[str, list[dict]]:
    assets = filter_by_type(load_watchlist(), "stock")
    return {asset.id: train_and_compare(asset.id) for asset in assets}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_and_compare(sys.argv[1])
    else:
        train_all_active_stocks()
