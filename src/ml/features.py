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


def long_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Longer lookback than price_lag_features/rolling_stat_features (which
    only look 1-10 rows back): month-ish (20), quarter-ish (60) and
    year-ish (252 trading days) rolling stats, momentum, and where today's
    close sits relative to its own trailing 252-day range. Needs deep
    history to produce anything (the first ~252 rows of a young asset will
    be NaN and get dropped by build_feature_frame's dropna).
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    for window in (20, 60, 252):
        out[f"rolling_mean_{window}"] = close.rolling(window).mean()
        out[f"return_{window}d"] = close.pct_change(window)
    out["rolling_std_20"] = close.rolling(20).std()
    out["rolling_std_60"] = close.rolling(60).std()
    out["pct_from_252d_high"] = close / close.rolling(252).max() - 1
    out["pct_from_252d_low"] = close / close.rolling(252).min() - 1
    return out


def _nearest_close_years_ago(df: pd.DataFrame, years_back: int) -> pd.Series:
    """The close price on the trading day nearest to exactly `years_back`
    calendar years before each row's date -- i.e. "the same day last year",
    tolerant of that exact calendar date falling on a weekend/holiday.
    """
    target_dates = pd.DataFrame(
        {"_row": range(len(df)), "event_date": df["event_date"] - pd.DateOffset(years=years_back)}
    ).sort_values("event_date")
    lookup = df[["event_date", "close"]].sort_values("event_date")

    matched = pd.merge_asof(
        target_dates, lookup, on="event_date", direction="nearest", tolerance=pd.Timedelta(days=10)
    ).sort_values("_row")
    return pd.Series(matched["close"].to_numpy(), index=df.index)


def _same_month_last_year_avg_close(df: pd.DataFrame) -> pd.Series:
    """Average close during the same calendar month one year ago -- e.g.
    for a row in July 2026, the average close across all of July 2025.
    Always fully in the past relative to the row it's attached to.
    """
    dates = df["event_date"]
    monthly_avg = df.assign(_y=dates.dt.year, _m=dates.dt.month).groupby(["_y", "_m"])["close"].mean()
    keys = pd.MultiIndex.from_arrays([dates.dt.year - 1, dates.dt.month])
    return pd.Series(monthly_avg.reindex(keys).to_numpy(), index=df.index)


def seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year comparisons against the same calendar period in prior
    years -- "what was it doing around now, last year / two years ago".
    Only possible with multi-year history, which is exactly why full
    history (not a short recent window) matters for this asset.
    """
    out = pd.DataFrame(index=df.index)
    for years_back in (1, 2):
        close_years_ago = _nearest_close_years_ago(df, years_back)
        out[f"close_{years_back}y_ago"] = close_years_ago
        out[f"return_{years_back}y"] = df["close"] / close_years_ago - 1

    same_month_avg = _same_month_last_year_avg_close(df)
    out["same_month_last_year_avg_close"] = same_month_avg
    out["vs_same_month_last_year"] = df["close"] / same_month_avg - 1
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
    long_horizon_features,
    seasonal_features,
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
