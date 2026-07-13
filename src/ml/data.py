from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from src.config.settings import settings

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
    return df.reset_index(drop=True)
