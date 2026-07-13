from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import pandas as pd

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def price_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Today's OHLCV plus lagged closes. `close_lag_0` is today's close --
    the anchor the naive baseline and the return/rolling features are built
    relative to.
    """
    out = pd.DataFrame(index=df.index)
    out["close_lag_0"] = df["close"]
    out["open_lag_0"] = df["open"]
    out["high_lag_0"] = df["high"]
    out["low_lag_0"] = df["low"]
    out["volume_lag_0"] = df["volume"]
    for lag in range(1, 6):
        out[f"close_lag_{lag}"] = df["close"].shift(lag)
    return out


def return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["return_1d"] = df["close"].pct_change(1)
    out["return_5d"] = df["close"].pct_change(5)
    return out


def rolling_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rolling_mean_5"] = df["close"].rolling(5).mean()
    out["rolling_mean_10"] = df["close"].rolling(10).mean()
    out["rolling_std_5"] = df["close"].rolling(5).std()
    out["high_low_spread"] = df["high"] - df["low"]
    return out


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["day_of_week"] = df["event_date"].dt.dayofweek
    return out


# Registered in order; each builder receives the raw OHLCV history frame
# (event_date, open, high, low, close, volume) and returns the columns it
# contributes. To bring in more variables later (fundamentals, macro data,
# ...), add a builder that joins its own source onto `event_date` and
# returns the extra columns here -- training/comparison code doesn't change.
FEATURE_BUILDERS: list[FeatureBuilder] = [
    price_lag_features,
    return_features,
    rolling_stat_features,
    calendar_features,
]


def build_feature_frame(price_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X, y, live_row) from raw OHLCV history.

    X/y cover every row with a known next-trading-day close, ready for
    training/evaluation. `live_row` is the single most recent row (features
    only -- no next-day close exists yet): feed it to a trained model to
    predict the next trading day.
    """
    df = price_history.sort_values("event_date").reset_index(drop=True)

    features = pd.concat([builder(df) for builder in FEATURE_BUILDERS], axis=1)
    target = df["close"].shift(-1)

    combined = pd.concat([features, target.rename("target")], axis=1)
    live_row = combined.iloc[[-1]].drop(columns=["target"])

    labeled = combined.iloc[:-1].dropna()
    X = labeled.drop(columns=["target"])
    y = labeled["target"]

    return X, y, live_row


def next_business_day(from_date: date) -> date:
    """The next weekday after `from_date` -- tomorrow, or Monday if
    `from_date` is a Friday (or weekend). Does not account for market
    holidays, only weekends.
    """
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += timedelta(days=1)
    return next_day
