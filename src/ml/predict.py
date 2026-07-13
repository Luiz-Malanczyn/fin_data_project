"""Predict the close price at a given horizon using saved models.

Usage:
    python -m src.ml.predict PETR4                    # all 3 horizons, best saved model each
    python -m src.ml.predict PETR4 weekly              # just the weekly horizon
    python -m src.ml.predict PETR4 weekly random_forest # a specific model
"""
from __future__ import annotations

import logging
import sys

from src.ml.data import load_price_history, lookup_investment_type
from src.ml.features import HORIZON_STEPS, build_feature_frame, target_date_for_horizon
from src.ml.storage import best_model_name, load_all_metadata, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = ("daily", "weekly", "monthly")


def predict_horizon(investment_id: str, horizon: str, model_name: str | None = None) -> dict:
    investment_type = lookup_investment_type(investment_id)
    horizon_days = HORIZON_STEPS[investment_type][horizon]

    history = load_price_history(investment_id, investment_type)
    _X, _y, live_row = build_feature_frame(history, horizon_days=horizon_days)

    model_name = model_name or best_model_name(investment_id, horizon)
    if model_name is None:
        raise RuntimeError(
            f"No trained model found for {investment_id!r}/{horizon!r}. Run src.ml.train first."
        )

    model, metadata = load_model(investment_id, horizon, model_name)

    # Guards against silently predicting with a model trained on a
    # different feature set (e.g. after adding fundamentals/macro data
    # without retraining).
    expected_features = metadata["feature_names"]
    if list(live_row.columns) != expected_features:
        raise ValueError(
            f"Feature mismatch for model {model_name!r} ({horizon}): it expects "
            f"{expected_features}, but current data has {list(live_row.columns)}. "
            "Retrain with `python -m src.ml.train` after changing feature engineering."
        )

    predicted_close = float(model.predict(live_row)[0])
    last_known_date = history["event_date"].max().date()
    last_known_close = float(live_row["close_lag_0"].iloc[0])
    target_date = target_date_for_horizon(last_known_date, investment_type, horizon)

    result = {
        "investment_id": investment_id,
        "investment_type": investment_type,
        "horizon": horizon,
        "model_name": model_name,
        "last_known_date": last_known_date.isoformat(),
        "last_known_close": last_known_close,
        "target_date": target_date.isoformat(),
        "predicted_close": predicted_close,
        "predicted_change_pct": (predicted_close / last_known_close - 1) * 100,
    }
    logger.info(
        "[%s/%s] %s: %s close=%.2f -> predicted %s close=%.2f (%+.2f%%)",
        investment_id,
        horizon,
        model_name,
        result["last_known_date"],
        last_known_close,
        result["target_date"],
        predicted_close,
        result["predicted_change_pct"],
    )
    return result


def predict_all_horizons(investment_id: str) -> dict[str, dict]:
    return {horizon: predict_horizon(investment_id, horizon) for horizon in HORIZONS}


def predict_horizon_detail(investment_id: str, horizon: str) -> dict:
    """Three different lenses on the same horizon, for reporting: the most
    accurate model on average (lowest cross-validated MAE -- usually
    naive_flat, i.e. "no change"), the trend-following naive_drift model
    (extrapolates recent momentum, so it actually moves), and whichever
    non-naive model showed the best directional accuracy in cross-
    validation (did it call up/down correctly more than half the time).
    """
    all_metadata = load_all_metadata(investment_id, horizon)
    if not all_metadata:
        raise RuntimeError(f"No trained models for {investment_id!r}/{horizon!r}. Run src.ml.train first.")

    best_mae_name = min(all_metadata, key=lambda m: m["metrics"]["mae"])["model_name"]
    non_naive = [m for m in all_metadata if not m["model_name"].startswith("naive")]

    result = {
        "most_accurate": predict_horizon(investment_id, horizon, best_mae_name),
        "trend_following": predict_horizon(investment_id, horizon, "naive_drift"),
    }
    if non_naive:
        best_directional = max(non_naive, key=lambda m: m["metrics"]["directional_accuracy"])
        pred = predict_horizon(investment_id, horizon, best_directional["model_name"])
        pred["directional_accuracy"] = best_directional["metrics"]["directional_accuracy"]
        result["best_directional"] = pred
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.ml.predict <investment_id> [horizon] [model_name]")

    if len(sys.argv) == 2:
        predict_all_horizons(sys.argv[1])
    elif len(sys.argv) == 3:
        predict_horizon(sys.argv[1], sys.argv[2])
    else:
        predict_horizon(sys.argv[1], sys.argv[2], sys.argv[3])
