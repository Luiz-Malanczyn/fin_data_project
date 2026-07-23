from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from src.config.settings import settings
from src.config.watchlist_loader import load_watchlist

_client: bigquery.Client | None = None


def _get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=settings.gcp_project or None)
    return _client


def load_price_history(investment_id: str, investment_type: str = "stock") -> pd.DataFrame:
    """Load the full OHLCV history for one asset from investment_history,
    sorted chronologically. This is the single source of truth every
    feature builder and model works from -- new data sources (fundamentals,
    macro indicators) should join onto this frame by event_date rather than
    replacing it.
    """
    client = _get_client()
    table = f"{client.project}.{settings.bq_dataset}.{settings.bq_history_table}"
    query = f"""
        SELECT event_date, open, high, low, close, volume
        FROM `{table}`
        WHERE investment_id = @investment_id AND investment_type = @investment_type
        ORDER BY event_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("investment_id", "STRING", investment_id),
            bigquery.ScalarQueryParameter("investment_type", "STRING", investment_type),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Real bug hit in production: Yahoo's chart API occasionally returns
    # open/high/low as null (loaded here as 0.0) for the most recent bar
    # while still providing a real close -- seen simultaneously across all
    # 8 Yahoo-sourced tickers for the same date, so it's a source quirk,
    # not one ticker's bad luck. A live prediction reads exactly this most
    # recent row, and price_lag_features feeds open/high/low straight into
    # the model -- a linear model given close=37 but open=high=low=0 (a
    # feature magnitude it never saw in training) produced a predicted
    # -53% single-day return, nonsense on its face. Repairing by pinning
    # open/high/low to that row's own close rather than dropping the row:
    # a flat/no-information bar is a far smaller distortion than a
    # fabricated price, and it doesn't shift every other row's alignment.
    broken = (df["open"] == 0) | (df["high"] == 0) | (df["low"] == 0)
    broken &= df["close"] != 0
    if broken.any():
        for col in ("open", "high", "low"):
            df.loc[broken, col] = df.loc[broken, "close"]

    return df.reset_index(drop=True)


def lookup_investment_type(investment_id: str, default: str = "stock") -> str:
    """Look up an asset's type from the watchlist, so callers that only
    have an id (e.g. a CLI arg) don't have to also pass the type by hand.
    """
    for asset in load_watchlist():
        if asset.id == investment_id:
            return asset.type
    return default
