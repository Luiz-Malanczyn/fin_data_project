"""Historical dividend payments -- fetched once per symbol and cached, then
used to compute trailing-twelve-month dividend yield (see dividend_features
in features.py). This is real point-in-time data: Yahoo's chart API
`events=div` parameter returns actual historical payment dates and amounts,
not a current snapshot applied retroactively -- brapi.dev's live P/L and
Yahoo's quoteSummary earnings history were both explored for valuation
multiples (P/L, ROE) but neither exposes free historical time series
(quoteSummary now requires an auth "crumb"), so this project doesn't
fabricate fake historical fundamentals -- dividends are the one fundamental
signal available for free with genuine history.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
HISTORY_START = datetime(2000, 1, 1, tzinfo=timezone.utc)

_cache: dict[str, pd.DataFrame] = {}


def fetch_dividend_history(symbol: str) -> pd.DataFrame:
    """event_date, dividend_amount -- one row per historical payment."""
    now = datetime.now(timezone.utc)
    response = requests.get(
        f"{CHART_URL}/{symbol}",
        params={
            "period1": int(HISTORY_START.timestamp()),
            "period2": int(now.timestamp()),
            "interval": "1mo",  # only the dividend events are used, not the bars
            "events": "div",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    results = (response.json().get("chart") or {}).get("result") or []
    if not results:
        return pd.DataFrame(columns=["event_date", "dividend_amount"])

    dividends = (results[0].get("events") or {}).get("dividends") or {}
    if not dividends:
        return pd.DataFrame(columns=["event_date", "dividend_amount"])

    rows = [
        {
            "event_date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date(),
            "dividend_amount": payload["amount"],
        }
        for ts, payload in dividends.items()
    ]
    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df.sort_values("event_date").reset_index(drop=True)


def get_dividend_history(symbol: str) -> pd.DataFrame:
    if symbol not in _cache:
        try:
            _cache[symbol] = fetch_dividend_history(symbol)
        except requests.exceptions.RequestException:
            _cache[symbol] = pd.DataFrame(columns=["event_date", "dividend_amount"])
    return _cache[symbol]
