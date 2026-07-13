from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.config.settings import settings
from src.extractors.stocks.base_stock_extractor import BaseStockExtractor
from src.models.schemas import AssetConfig, InvestmentHistoryRecord

BRAPI_BASE_URL = "https://brapi.dev/api/quote"

# Valid values for the `range` param accepted by brapi.dev's quote endpoint.
VALID_RANGES = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


class BrapiStockExtractor(BaseStockExtractor):
    """Extracts B3 stock history via brapi.dev.

    Docs: https://brapi.dev/docs
    """

    source = "brapi"

    def fetch_raw(self, asset: AssetConfig) -> dict:
        params = {
            "range": asset.params.get("range", "3mo"),
            "interval": asset.params.get("interval", "1d"),
        }
        if settings.brapi_token:
            params["token"] = settings.brapi_token

        response = requests.get(f"{BRAPI_BASE_URL}/{asset.id}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def normalize(self, raw: dict, asset: AssetConfig) -> list[InvestmentHistoryRecord]:
        results = raw.get("results") or []
        if not results:
            return []

        result = results[0]
        currency = result.get("currency", "BRL")
        history = result.get("historicalDataPrice") or []
        ingestion_ts = datetime.now(timezone.utc)

        records = []
        for point in history:
            ts = point.get("date")
            if ts is None:
                continue
            records.append(
                InvestmentHistoryRecord(
                    event_date=datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                    investment_type=self.investment_type,
                    investment_id=asset.id,
                    source=self.source,
                    currency=currency,
                    open=point.get("open"),
                    high=point.get("high"),
                    low=point.get("low"),
                    close=point.get("close"),
                    volume=point.get("volume"),
                    extra={
                        "adjusted_close": point.get("adjustedClose"),
                        "exchange": asset.exchange,
                    },
                    ingestion_ts=ingestion_ts,
                )
            )
        return records
