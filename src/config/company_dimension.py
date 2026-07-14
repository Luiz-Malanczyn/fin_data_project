"""Unified company/asset dimension table (BigQuery: fin_data.company_dimension).

Consolidates attributes that used to be scattered across separate
module-level dicts -- each one keyed by the same ticker, maintained
independently, with no guarantee they stayed in sync:
  - news_data.py: COMPANY_SEARCH_TERMS, COMPANY_URL_TERMS, COMPANY_DISPLAY_NAMES
  - fundamentals_data.py: COMPANY_CODES (cnpj) + a ticker-suffix rule for share_class
  - yahoo_extractor.py / features.py: the ".SA" Yahoo suffix convention,
    hardcoded independently in two places -- features.py's copy had no way
    to pick up a watchlist.yaml `params.symbol` override the extractor
    itself already supported, a latent bug this table also fixes by
    making yahoo_symbol the single source of truth for both.

One row per tracked investment (stocks and crypto alike); columns that
don't apply to a given investment_type (cnpj/share_class/yahoo_symbol for
crypto) are simply NULL.
"""
from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from src.config.settings import settings

TABLE = "company_dimension"


def _table_ref() -> str:
    return f"{settings.gcp_project}.{settings.bq_dataset}.{TABLE}"


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project or None)


SCHEMA = [
    bigquery.SchemaField("investment_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("investment_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("display_name", "STRING"),
    bigquery.SchemaField("cnpj", "STRING"),
    bigquery.SchemaField("share_class", "STRING"),
    bigquery.SchemaField("yahoo_symbol", "STRING"),
    bigquery.SchemaField("news_search_term", "STRING"),
    bigquery.SchemaField("news_url_terms", "STRING", mode="REPEATED"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("active", "BOOL"),
]


def ensure_table() -> None:
    table = bigquery.Table(_table_ref(), schema=SCHEMA)
    _bq_client().create_table(table, exists_ok=True)


_cache: pd.DataFrame | None = None


def get_company_dimension() -> pd.DataFrame:
    """The full dimension table, fetched once per process and cached --
    it's small (13 rows) and changes rarely, so every caller sharing one
    read avoids re-querying BigQuery per asset per feature build."""
    global _cache
    if _cache is None:
        query = f"SELECT * FROM `{_table_ref()}`"
        _cache = _bq_client().query(query).to_dataframe()
    return _cache


def get_company(investment_id: str) -> dict:
    """Single-row lookup as a plain dict. Raises KeyError if the ticker
    isn't in the dimension table -- callers should add it there rather
    than falling back to a guessed convention, so the table stays the one
    source of truth."""
    df = get_company_dimension()
    match = df[df["investment_id"] == investment_id]
    if match.empty:
        raise KeyError(f"{investment_id!r} not found in company_dimension")
    return match.iloc[0].to_dict()
