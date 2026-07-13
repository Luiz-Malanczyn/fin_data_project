from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class AssetConfig(BaseModel):
    """A watchlist entry: what to extract and where from."""

    id: str
    type: str  # 'stock', 'crypto', 'fixed_income', ...
    source: str  # 'brapi', 'coingecko', ...
    exchange: str | None = None
    active: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class InvestmentHistoryRecord(BaseModel):
    """A row of the unified `investment_history` table in BigQuery.

    event_date + investment_type + investment_id + source form the
    deduplication key used by the MERGE load.
    """

    event_date: date
    investment_type: str
    investment_id: str
    source: str
    currency: str | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    ingestion_ts: datetime
    raw_uri: str | None = None
