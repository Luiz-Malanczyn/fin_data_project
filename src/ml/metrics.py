from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def compute_metrics(y_true, y_pred, baseline) -> dict:
    """Regression error metrics plus directional accuracy: did the model
    correctly call whether the price would go up or down from `baseline`
    (today's close)? For trading decisions that's often more useful than
    the raw price error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    actual_direction = np.sign(y_true - baseline)
    predicted_direction = np.sign(y_pred - baseline)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "n_samples": int(len(y_true)),
    }
