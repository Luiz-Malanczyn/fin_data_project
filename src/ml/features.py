from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.config.company_dimension import get_company
from src.ml.dividend_data import get_dividend_history
from src.ml.fundamentals_data import get_fundamentals_history
from src.ml.macro_data import get_macro_data
from src.ml.news_data import get_news_sentiment_history

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


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume relative to its own recent average, and day-over-day change --
    raw volume_lag_0 alone doesn't say whether today's volume was high or
    low *for this asset*, only these ratio features do.
    """
    out = pd.DataFrame(index=df.index)
    volume = df["volume"]
    out["volume_ratio_5"] = volume / volume.rolling(5).mean() - 1
    out["volume_ratio_20"] = volume / volume.rolling(20).mean() - 1
    out["volume_change_1d"] = volume.pct_change(1)
    return out


def risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Realized volatility (annualized) and a Sharpe-like risk-adjusted
    momentum ratio -- pure price-derived, so unlike fundamentals these are
    exactly correct at every historical point, and apply to any asset
    (stock or crypto) since they don't depend on Ibovespa/Selic.
    """
    out = pd.DataFrame(index=df.index)
    daily_return = df["close"].pct_change(1)
    for window in (20, 60):
        vol = daily_return.rolling(window).std() * np.sqrt(252)
        out[f"volatility_{window}d"] = vol
        mean_return_annualized = daily_return.rolling(window).mean() * 252
        out[f"sharpe_{window}d"] = mean_return_annualized / vol
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


def _merge_asof_backward(dates: pd.Series, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """As-of join: for each date in `dates`, the latest known value of
    `columns` from `other` at or before that date. `direction="backward"`
    is what makes this leak-free for series that don't update every day
    (Selic only changes on COPOM dates, dividends are paid a few times a
    year) -- it carries the last known value forward instead of
    interpolating from the future.
    """
    # merge_asof requires identical datetime64 units on both sides; sources
    # differ (BigQuery DATE vs. parsed strings), so normalize explicitly
    # instead of relying on both frames happening to agree.
    left = pd.DataFrame(
        {"_row": range(len(dates)), "event_date": pd.to_datetime(dates).astype("datetime64[ns]")}
    ).sort_values("event_date")
    right = other[["event_date"] + columns].copy()
    right["event_date"] = pd.to_datetime(right["event_date"]).astype("datetime64[ns]")
    right = right.sort_values("event_date")
    merged = pd.merge_asof(left, right, on="event_date", direction="backward").sort_values("_row")
    return merged[columns].reset_index(drop=True)


def macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stock-only: Ibovespa (market-wide momentum + this stock's return
    relative to it), USD/BRL (currency moves matter a lot for exporters
    like PETR4/VALE3/SUZB3), and the Selic policy rate. Registered
    separately from FEATURE_BUILDERS in build_feature_frame -- crypto has
    no natural equivalent to "the Brazilian stock index" or "the Brazilian
    policy rate", so this builder never runs for it.
    """
    ibov, usd_brl, selic = get_macro_data()
    out = pd.DataFrame(index=df.index)
    dates = df["event_date"]

    ibov_cols = _merge_asof_backward(dates, ibov, ["ibov_return_1d", "ibov_return_5d", "ibov_return_20d"])
    out["ibov_return_1d"] = ibov_cols["ibov_return_1d"]
    out["ibov_return_5d"] = ibov_cols["ibov_return_5d"]
    out["ibov_return_20d"] = ibov_cols["ibov_return_20d"]
    out["relative_strength_1d"] = df["close"].pct_change(1).to_numpy() - out["ibov_return_1d"].to_numpy()

    usd_brl_cols = _merge_asof_backward(dates, usd_brl, ["usd_brl", "usd_brl_return_5d"])
    out["usd_brl"] = usd_brl_cols["usd_brl"]
    out["usd_brl_return_5d"] = usd_brl_cols["usd_brl_return_5d"]

    selic_cols = _merge_asof_backward(dates, selic, ["selic_rate"])
    out["selic_rate"] = selic_cols["selic_rate"]

    return out


def beta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 60-day beta vs. the Ibovespa: how much this stock amplifies
    (beta > 1) or dampens (beta < 1) the market's moves -- a risk measure
    that "will it go up" alone doesn't capture. Stock-only (needs the
    index), computed from the same Ibovespa series as macro_features.
    """
    ibov, _usd_brl, _selic = get_macro_data()
    out = pd.DataFrame(index=df.index)

    ibov_close = _merge_asof_backward(df["event_date"], ibov, ["ibov_close"])["ibov_close"]
    stock_return = df["close"].pct_change(1)
    ibov_return = ibov_close.pct_change(1)

    rolling_cov = stock_return.rolling(60).cov(ibov_return)
    rolling_var = ibov_return.rolling(60).var()
    out["beta_60d"] = rolling_cov / rolling_var
    return out


def dividend_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Trailing-twelve-month dividend yield, from real historical payment
    dates (see dividend_data.py) -- not a current snapshot applied
    retroactively. Needs the specific stock's own dividend history, so
    unlike the other builders this one takes an extra argument and is
    wired in per-asset by build_feature_frame rather than living in
    FEATURE_BUILDERS/STOCK_ONLY_FEATURE_BUILDERS.
    """
    dividends = get_dividend_history(symbol)
    out = pd.DataFrame(index=df.index)
    if dividends.empty:
        out["dividend_yield_ttm"] = 0.0
        return out

    dividends = dividends.copy()
    dividends["cum_dividends"] = dividends["dividend_amount"].cumsum()

    dates = df["event_date"]
    window_start_dates = dates - pd.Timedelta(days=365)

    cum_now = _merge_asof_backward(dates, dividends, ["cum_dividends"])["cum_dividends"]
    cum_before = _merge_asof_backward(window_start_dates, dividends, ["cum_dividends"])["cum_dividends"]

    ttm_dividends = (cum_now.fillna(0) - cum_before.fillna(0)).clip(lower=0)
    out["dividend_yield_ttm"] = ttm_dividends.to_numpy() / df["close"].to_numpy()
    return out


def fundamentals_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """P/L, ROE, debt/equity, and net margin from CVM's quarterly filings
    (see fundamentals_data.py), as-of joined on each statement's real
    public disclosure date rather than the quarter it covers -- a row never
    sees a number before it actually existed. Needs the specific stock's
    CVM company code, so wired in per-asset by build_feature_frame like
    dividend_features rather than living in STOCK_ONLY_FEATURE_BUILDERS.

    Earlier version left rows with no known disclosure yet (CVM's
    standardized filings only go back to 2011) as NaN, relying on
    build_feature_frame's global dropna() to drop them -- but that dropna()
    requires *every* column to be non-null, so a stock's entire pre-2011
    price history (otherwise perfectly usable) got discarded just because
    fundamentals weren't available yet. Measured effect: retraining all 39
    (asset, horizon) combos this way lost 38-88% of training rows per
    asset (worst for the two banks) and made results *worse* than not
    having fundamentals at all (28 vs 45 significant findings, 6/39 vs
    12/39 backtests beating buy-and-hold).

    Fix: fill undisclosed/undefined ratios with a fixed neutral 0.0 (not a
    computed mean/median -- that would leak information from later,
    already-known quarters into earlier rows) and add `has_fundamentals`
    so models can learn to disregard the ratio columns when it's 0, instead
    of losing the row entirely.
    """
    out = pd.DataFrame(index=df.index)
    fundamentals = get_fundamentals_history(ticker)
    if fundamentals.empty:
        out["pe_ratio"] = 0.0
        out["roe_ltm"] = 0.0
        out["debt_to_equity"] = 0.0
        out["net_margin_ltm"] = 0.0
        out["has_fundamentals"] = 0.0
        return out

    cols = _merge_asof_backward(
        df["event_date"],
        fundamentals,
        ["revenue_ltm", "net_income_ltm", "eps_ltm", "equity", "total_liabilities"],
    )
    out["has_fundamentals"] = cols["equity"].notna().astype(float).to_numpy()
    out["pe_ratio"] = df["close"].to_numpy() / cols["eps_ltm"].to_numpy()
    out["roe_ltm"] = cols["net_income_ltm"].to_numpy() / cols["equity"].to_numpy()
    out["debt_to_equity"] = cols["total_liabilities"].to_numpy() / cols["equity"].to_numpy()
    out["net_margin_ltm"] = cols["net_income_ltm"].to_numpy() / cols["revenue_ltm"].to_numpy()
    for ratio_col in ("pe_ratio", "roe_ltm", "debt_to_equity", "net_margin_ltm"):
        out[ratio_col] = out[ratio_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def news_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Daily news sentiment from GDELT articles scored by an LLM for
    relevance and financial impact (see news_data.py). Exact same-day join
    on event_date, not an as-of/forward-fill like dividend_features or
    fundamentals_features: a fundamentals ratio is still true weeks after
    the filing, but a news event from last week isn't "today's sentiment"
    -- carrying it forward would manufacture a signal that isn't there.

    Coverage is sparse by construction (real relevant articles found per
    ticker range from ~40 to ~220 over 11 years of trading days), so a
    same-day-only feature would be zero on well over 95% of rows. Also
    add a 7-day trailing rolling average (article-count-weighted, so one
    strong day isn't diluted the same as a quiet one) to give the model
    a less starved signal, same rationale as the has_fundamentals flag:
    missing coverage gets a neutral 0.0 plus an explicit flag rather than
    dropping the row.
    """
    out = pd.DataFrame(index=df.index)
    news = get_news_sentiment_history(ticker)
    if news.empty:
        out["news_sentiment_score"] = 0.0
        out["news_article_count"] = 0.0
        out["news_sentiment_7d"] = 0.0
        out["has_news_coverage"] = 0.0
        return out

    merged = df[["event_date"]].merge(news, on="event_date", how="left")
    sentiment = merged["news_sentiment_score"].fillna(0.0)
    count = merged["n_articles"].fillna(0.0)

    out["news_sentiment_score"] = sentiment.to_numpy()
    out["news_article_count"] = count.to_numpy()
    weighted_sum = (sentiment * count).rolling(7, min_periods=1).sum()
    count_sum = count.rolling(7, min_periods=1).sum()
    out["news_sentiment_7d"] = (weighted_sum / count_sum.replace(0, np.nan)).fillna(0.0).to_numpy()
    out["has_news_coverage"] = merged["n_articles"].notna().astype(float).to_numpy()
    return out


# Registered in order; each builder receives the raw OHLCV history frame
# (event_date, open, high, low, close, volume) and returns the columns it
# contributes. To bring in more variables later (fundamentals, ...), add a
# builder that joins its own source onto `event_date` and returns the extra
# columns here -- training/comparison code doesn't change.
FEATURE_BUILDERS: list[FeatureBuilder] = [
    price_lag_features,
    return_features,
    rolling_stat_features,
    calendar_features,
    volume_features,
    risk_features,
    long_horizon_features,
    seasonal_features,
]

# Stock-only builders, appended on top of FEATURE_BUILDERS. Kept separate
# because they fetch external data with no equivalent for other investment
# types (see macro_features).
STOCK_ONLY_FEATURE_BUILDERS: list[FeatureBuilder] = [
    macro_features,
    beta_features,
]


def build_feature_frame(
    price_history: pd.DataFrame,
    horizon_days: int = 1,
    investment_type: str = "stock",
    investment_id: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X, y, live_row) from raw OHLCV history.

    `horizon_days` is how many rows ahead the target looks -- 1 for "next
    row", 5 for "5 rows from now", etc. The feature set itself never
    changes; only how far forward `target` is shifted does, so the same
    features can be compared at daily/weekly/monthly horizons.

    `investment_type="stock"` also appends STOCK_ONLY_FEATURE_BUILDERS
    (Ibovespa/USD-BRL/Selic/beta) -- skipped for crypto and anything else,
    which has no natural equivalent. `investment_id` additionally enables
    dividend_features, which needs to know *which* stock's own dividend
    history to fetch (unlike Ibovespa/Selic, dividends aren't shared
    across assets) -- omit it to build a feature frame without dividends
    (e.g. for a quick check where fetching dividend history isn't worth it).

    `y` is the **return** over the horizon (future_close / close - 1), not
    the raw future price. Today's close is already ~all of the information
    in tomorrow's close, so fitting raw price lets a model win just by
    copying `close_lag_0` -- that's the actual reason "no change" was so
    hard to beat. Fitting the return instead forces every model to commit
    to an actual delta; predictions are converted back to price at
    display time (predicted_close = close_lag_0 * (1 + predicted_return)).

    X/y cover every row with a known target that many rows ahead, ready for
    training/evaluation. `live_row` is always the single most recent row
    (features only -- its target is in the future, unknown): feed it to a
    trained model to predict `horizon_days` ahead.
    """
    df = price_history.sort_values("event_date").reset_index(drop=True)

    feature_frames = [builder(df) for builder in FEATURE_BUILDERS]
    if investment_type == "stock":
        feature_frames += [builder(df) for builder in STOCK_ONLY_FEATURE_BUILDERS]
        if investment_id:
            try:
                yahoo_symbol = get_company(investment_id)["yahoo_symbol"]
            except KeyError:
                # Not in company_dimension yet -- fall back to the default
                # convention rather than skipping dividends outright.
                yahoo_symbol = f"{investment_id}.SA"
            feature_frames.append(dividend_features(df, yahoo_symbol))
            feature_frames.append(fundamentals_features(df, investment_id))
            feature_frames.append(news_sentiment_features(df, investment_id))
    features = pd.concat(feature_frames, axis=1)
    # A handful of features are ratios (volume_change_1d, macro returns);
    # a zero-volume illiquid day or similar upstream glitch can turn one of
    # those into +-inf, which every sklearn estimator rejects outright. NaN
    # already funnels through the same dropna() below, so infinities do too.
    features = features.replace([np.inf, -np.inf], np.nan)
    future_close = df["close"].shift(-horizon_days)
    target = future_close / df["close"] - 1

    live_row = features.iloc[[-1]]

    combined = pd.concat([features, target.rename("target")], axis=1).dropna()
    X = combined.drop(columns=["target"])
    y = combined["target"]

    return X, y, live_row


# How many rows ahead each horizon means, per investment type. A stock
# history has one row per trading day (~252/year); crypto has one row per
# calendar day (it trades every day), so the row counts differ even though
# the calendar meaning ("about a week", "about a month") is the same.
HORIZON_STEPS: dict[str, dict[str, int]] = {
    "stock": {"daily": 1, "weekly": 5, "monthly": 21},
    "crypto": {"daily": 1, "weekly": 7, "monthly": 30},
}


def add_business_days(from_date: date, n: int) -> date:
    """`n` weekdays after `from_date` (n=1 behaves like next_business_day)."""
    current = from_date
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 .. Friday=4
            added += 1
    return current


def next_business_day(from_date: date) -> date:
    """The next weekday after `from_date` -- tomorrow, or Monday if
    `from_date` is a Friday (or weekend). Does not account for market
    holidays, only weekends.
    """
    return add_business_days(from_date, 1)


def target_date_for_horizon(last_known_date: date, investment_type: str, horizon: str) -> date:
    """The calendar date a prediction for `horizon` actually refers to,
    given the most recent date there's data for. Crypto markets never
    close, so horizons are plain calendar days; stock exchanges do, so
    horizons skip weekends.
    """
    steps = HORIZON_STEPS[investment_type][horizon]
    if investment_type == "crypto":
        return last_known_date + timedelta(days=steps)
    return add_business_days(last_known_date, steps)
