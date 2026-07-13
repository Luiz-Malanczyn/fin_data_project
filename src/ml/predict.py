"""Predict the next trading day's close price using a saved model.

Usage:
    python -m src.ml.predict PETR4                 # uses the best saved model (lowest MAE)
    python -m src.ml.predict PETR4 random_forest    # uses a specific model
"""
from __future__ import annotations

import logging
import sys

from src.ml.data import load_price_history
from src.ml.features import build_feature_frame, next_business_day
from src.ml.storage import best_model_name, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_next_close(investment_id: str, model_name: str | None = None) -> dict:
    history = load_price_history(investment_id)
    _X, _y, live_row = build_feature_frame(history)

    model_name = model_name or best_model_name(investment_id)
    if model_name is None:
        raise RuntimeError(f"No trained model found for {investment_id!r}. Run src.ml.train first.")

    model, metadata = load_model(investment_id, model_name)

    # Guards against silently predicting with a model trained on a
    # different feature set (e.g. after adding fundamentals/macro data
    # without retraining).
    expected_features = metadata["feature_names"]
    if list(live_row.columns) != expected_features:
        raise ValueError(
            f"Feature mismatch for model {model_name!r}: it expects "
            f"{expected_features}, but current data has {list(live_row.columns)}. "
            "Retrain with `python -m src.ml.train` after changing feature engineering."
        )

    predicted_close = float(model.predict(live_row)[0])
    last_known_date = history["event_date"].max().date()
    last_known_close = float(live_row["close_lag_0"].iloc[0])
    target_date = next_business_day(last_known_date)

    result = {
        "investment_id": investment_id,
        "model_name": model_name,
        "last_known_date": last_known_date.isoformat(),
        "last_known_close": last_known_close,
        "target_date": target_date.isoformat(),
        "predicted_close": predicted_close,
        "predicted_change_pct": (predicted_close / last_known_close - 1) * 100,
    }
    logger.info(
        "[%s] %s: %s close=%.2f -> predicted %s close=%.2f (%+.2f%%)",
        investment_id,
        model_name,
        result["last_known_date"],
        last_known_close,
        result["target_date"],
        predicted_close,
        result["predicted_change_pct"],
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.ml.predict <investment_id> [model_name]")
    predict_next_close(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
