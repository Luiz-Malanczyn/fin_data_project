from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.extractors.crypto.base_crypto_extractor import BaseCryptoExtractor
from src.models.schemas import AssetConfig, InvestmentHistoryRecord

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/coins"


class CoinGeckoCryptoExtractor(BaseCryptoExtractor):
    """Extracts cryptocurrency history via CoinGecko (free tier, no key required).

    Docs: https://www.coingecko.com/en/api/documentation
    """

    source = "coingecko"

    def fetch_raw(self, asset: AssetConfig) -> dict:
        params = {
            "vs_currency": asset.params.get("vs_currency", "usd"),
            "days": asset.params.get("days", 90),
            "interval": "daily",
        }
        response = requests.get(
            f"{COINGECKO_BASE_URL}/{asset.id}/market_chart", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def normalize(self, raw: dict, asset: AssetConfig) -> list[InvestmentHistoryRecord]:
        prices = raw.get("prices") or []
        volumes = dict((p[0], p[1]) for p in raw.get("total_volumes") or [])
        market_caps = dict((p[0], p[1]) for p in raw.get("market_caps") or [])
        ingestion_ts = datetime.now(timezone.utc)
        currency = asset.params.get("vs_currency", "usd")

        records = []
        for ts_ms, price in prices:
            event_date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            records.append(
                InvestmentHistoryRecord(
                    event_date=event_date,
                    investment_type=self.investment_type,
                    investment_id=asset.id,
                    source=self.source,
                    currency=currency,
                    close=price,
                    volume=volumes.get(ts_ms),
                    extra={"market_cap": market_caps.get(ts_ms)},
                    ingestion_ts=ingestion_ts,
                )
            )
        return records
