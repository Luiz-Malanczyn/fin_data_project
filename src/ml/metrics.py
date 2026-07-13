from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def compute_metrics(y_true_return, y_pred_return, baseline_close) -> dict:
    """Metrics for a model fit on returns, not raw price.

    `mae`/`rmse`/`r2` are computed in return space -- the space the model
    actually optimizes, and the fair way to compare models (a model can't
    win these just by copying today's close, the way it trivially could
    when the target was the raw price level). `mae_price`/`mape_price`
    convert back to price using each row's own `baseline_close`, purely for
    human-readable "off by about R$X" reporting.

    `directional_accuracy` is how often the model's predicted return has
    the same sign as the actual return.
    """
    y_true_return = np.asarray(y_true_return, dtype=float)
    y_pred_return = np.asarray(y_pred_return, dtype=float)
    baseline_close = np.asarray(baseline_close, dtype=float)

    true_price = baseline_close * (1 + y_true_return)
    pred_price = baseline_close * (1 + y_pred_return)

    actual_direction = np.sign(y_true_return)
    predicted_direction = np.sign(y_pred_return)

    return {
        "mae": float(mean_absolute_error(y_true_return, y_pred_return)),
        "rmse": float(np.sqrt(np.mean((y_true_return - y_pred_return) ** 2))),
        "r2": float(r2_score(y_true_return, y_pred_return)),
        "mae_price": float(mean_absolute_error(true_price, pred_price)),
        "mape_price": float(mean_absolute_percentage_error(true_price, pred_price)),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "n_samples": int(len(y_true_return)),
    }
