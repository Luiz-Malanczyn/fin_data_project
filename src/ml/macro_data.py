"""External market/macro data sources -- fetched once per process and
cached in memory, then joined onto each stock's feature frame (see
macro_features in features.py). Not part of investment_history: these
describe the wider market, not a single trackable asset, so they're
fetched fresh at training/prediction time rather than stored in BigQuery.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

MAX_RETRIES = 3

BCB_SERIES_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
IBOVESPA_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"

# BCB's daily-series API rejects date windows over ~10 years, so full
# history is fetched in fixed-size chunks from this start date to today.
BCB_HISTORY_START = date(2000, 1, 1)
BCB_CHUNK_YEARS = 9


def _fetch_bcb_series(code: int, value_column: str) -> pd.DataFrame:
    frames = []
    chunk_start = BCB_HISTORY_START
    today = date.today()

    while chunk_start <= today:
        chunk_end = min(
            date(chunk_start.year + BCB_CHUNK_YEARS, chunk_start.month, chunk_start.day), today
        )
        # BCB's API occasionally times out on a chunk with no apparent
        # pattern; a couple of retries reliably clears it.
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    BCB_SERIES_URL.format(code=code),
                    params={
                        "formato": "json",
                        "dataInicial": chunk_start.strftime("%d/%m/%Y"),
                        "dataFinal": chunk_end.strftime("%d/%m/%Y"),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2 * attempt)
        raw = response.json()
        if raw:
            frames.append(pd.DataFrame(raw))
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["event_date", value_column])

    df = pd.concat(frames, ignore_index=True)
    df["event_date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df[value_column] = df["valor"].astype(float)
    return (
        df[["event_date", value_column]]
        .sort_values("event_date")
        .drop_duplicates("event_date")
        .reset_index(drop=True)
    )


def fetch_usd_brl_history() -> pd.DataFrame:
    """event_date, usd_brl, usd_brl_return_5d -- BCB SGS series 1 (dolar comercial venda)."""
    df = _fetch_bcb_series(1, "usd_brl")
    df["usd_brl_return_5d"] = df["usd_brl"].pct_change(5)
    return df


def fetch_selic_history() -> pd.DataFrame:
    """event_date, selic_rate -- BCB SGS series 432 (meta Selic)."""
    return _fetch_bcb_series(432, "selic_rate")


def fetch_ibovespa_history() -> pd.DataFrame:
    """event_date, ibov_close, ibov_return_1d/5d/20d -- via Yahoo Finance,
    same chart endpoint used by YahooFinanceStockExtractor.

    Yahoo silently serves monthly bars instead of daily ones when asked for
    interval=1d over range=max on this index (fine on individual stocks,
    not on ^BVSP) -- explicit period1/period2 unix-timestamp windows keep
    it honest, so history is paginated the same ~9-year-chunk way as BCB.
    """
    frames = []
    chunk_start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    while chunk_start <= now:
        chunk_end = min(
            chunk_start.replace(year=chunk_start.year + BCB_CHUNK_YEARS), now
        )
        response = requests.get(
            IBOVESPA_CHART_URL,
            params={
                "period1": int(chunk_start.timestamp()),
                "period2": int(chunk_end.timestamp()),
                "interval": "1d",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        closes = result["indicators"]["quote"][0]["close"]
        if timestamps:
            frames.append(
                pd.DataFrame(
                    {
                        "event_date": [
                            datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in timestamps
                        ],
                        "ibov_close": closes,
                    }
                )
            )
        chunk_start = chunk_end + pd.Timedelta(days=1)

    df = pd.concat(frames, ignore_index=True).dropna()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").drop_duplicates("event_date").reset_index(drop=True)
    for window in (1, 5, 20):
        df[f"ibov_return_{window}d"] = df["ibov_close"].pct_change(window)
    return df


_cache: dict[str, pd.DataFrame] = {}


def get_macro_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ibovespa, usd_brl, selic) history, fetched once and cached for the
    life of the process -- every stock's feature build reuses the same
    three frames instead of re-fetching per asset.

    Populated into a local dict first and only copied into the module-level
    cache once all three fetches succeed -- if e.g. the Selic fetch fails
    after Ibovespa already succeeded, a naive "cache as you go" approach
    would leave `_cache` holding ibov but missing selic, so every
    subsequent call in the same process would `if "ibov" not in _cache`
    (true) and skip re-fetching, then KeyError on `_cache["selic"]".
    """
    if "ibov" not in _cache:
        fetched = {
            "ibov": fetch_ibovespa_history(),
            "usd_brl": fetch_usd_brl_history(),
            "selic": fetch_selic_history(),
        }
        _cache.update(fetched)
    return _cache["ibov"], _cache["usd_brl"], _cache["selic"]
