"""Walk-forward backtest: would the significant directional signal we found
actually have made money, after realistic transaction costs?

Statistically significant directional accuracy is necessary but not
sufficient -- a 52-58% edge can still lose money once trading costs eat
into it. This simulates a simple long/cash strategy (never shorts) using
only walk-forward, out-of-fold predictions -- the same discipline as
train.py's cross-validation, so nothing here has seen the future.

Usage:
    python -m src.ml.backtest PETR4 monthly
    python -m src.ml.backtest              # every active asset x horizon
"""
from __future__ import annotations

import logging
import sys

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.config.watchlist_loader import load_watchlist
from src.ml.data import load_price_history, lookup_investment_type
from src.ml.features import HORIZON_STEPS, build_feature_frame
from src.ml.storage import load_all_metadata, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5
HORIZONS = ("daily", "weekly", "monthly")

# Round-trip cost estimate for a B3 retail trade: brokerage is close to
# zero at most brokers now, but bid-ask spread + B3 emoluments/liquidation
# fees still apply on both the buy and the sell. 0.2% is a conservative
# (i.e. not flattering to the strategy) estimate.
ROUND_TRIP_COST = 0.002


def _choose_model_name(investment_id: str, horizon: str) -> str:
    """The model actually worth backtesting: the non-naive model with the
    best statistically significant directional accuracy, if one exists --
    otherwise there's no real signal to backtest, so fall back to whichever
    model had the lowest MAE (almost always naive_flat, which by
    construction never trades).
    """
    all_metadata = load_all_metadata(investment_id, horizon)
    significant = [
        m
        for m in all_metadata
        if not m["model_name"].startswith("naive") and m["metrics"]["directional_accuracy_significant"]
    ]
    if significant:
        return max(significant, key=lambda m: m["metrics"]["directional_accuracy"])["model_name"]
    return min(all_metadata, key=lambda m: m["metrics"]["mae"])["model_name"]


def backtest_horizon(investment_id: str, horizon: str, model_name: str | None = None) -> dict:
    investment_type = lookup_investment_type(investment_id)
    horizon_days = HORIZON_STEPS[investment_type][horizon]

    history = load_price_history(investment_id, investment_type)
    X, y, _live_row = build_feature_frame(
        history, horizon_days=horizon_days, investment_type=investment_type, investment_id=investment_id
    )

    model_name = model_name or _choose_model_name(investment_id, horizon)
    fitted_model, _metadata = load_model(investment_id, horizon, model_name)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    strategy_returns: list[float] = []
    buy_hold_returns: list[float] = []
    n_trades = 0

    for train_idx, test_idx in tscv.split(X):
        # A fresh copy refit only on this fold's training rows -- walk-
        # forward, so nothing in the backtest window was seen during fit.
        fold_model = clone(fitted_model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Non-overlapping horizon_days blocks: this is what a strategy that
        # actually rebalances every `horizon` would do, avoiding the
        # double-counting that comes from treating every overlapping row as
        # an independent trade.
        for block_start in range(0, len(test_idx), horizon_days):
            block_idx = test_idx[block_start : block_start + horizon_days]
            if len(block_idx) == 0:
                continue
            i = block_idx[0]
            predicted_return = float(fold_model.predict(X.iloc[[i]])[0])
            actual_return = float(y.iloc[i])

            buy_hold_returns.append(actual_return)
            if predicted_return > 0:
                strategy_returns.append(actual_return - ROUND_TRIP_COST)
                n_trades += 1
            else:
                strategy_returns.append(0.0)

    strategy_cumulative = float(np.prod([1 + r for r in strategy_returns]) - 1)
    buy_hold_cumulative = float(np.prod([1 + r for r in buy_hold_returns]) - 1)
    winning_trades = sum(1 for r in strategy_returns if r > 0)

    result = {
        "investment_id": investment_id,
        "horizon": horizon,
        "model_name": model_name,
        "n_blocks": len(strategy_returns),
        "n_trades": n_trades,
        "win_rate": winning_trades / n_trades if n_trades else 0.0,
        "strategy_cumulative_return": strategy_cumulative,
        "buy_hold_cumulative_return": buy_hold_cumulative,
        "beats_buy_hold": strategy_cumulative > buy_hold_cumulative,
        "round_trip_cost": ROUND_TRIP_COST,
    }
    logger.info(
        "[%s/%s] backtest (%s): strategy=%+.1f%% buy&hold=%+.1f%% over %d blocks (%d trades) -> %s",
        investment_id,
        horizon,
        model_name,
        strategy_cumulative * 100,
        buy_hold_cumulative * 100,
        result["n_blocks"],
        n_trades,
        "beats buy & hold" if result["beats_buy_hold"] else "does not beat buy & hold",
    )
    return result


def backtest_all_active_assets() -> list[dict]:
    assets = [a for a in load_watchlist() if a.active]
    results = []
    for asset in assets:
        for horizon in HORIZONS:
            results.append(backtest_horizon(asset.id, horizon))
    return results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        backtest_all_active_assets()
    elif len(sys.argv) == 3:
        backtest_horizon(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit("usage: python -m src.ml.backtest [<investment_id> <horizon>]")
