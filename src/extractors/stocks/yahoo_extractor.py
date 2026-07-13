from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.extractors.stocks.base_stock_extractor import BaseStockExtractor
from src.models.schemas import AssetConfig, InvestmentHistoryRecord

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


class YahooFinanceStockExtractor(BaseStockExtractor):
    """Extracts stock history via Yahoo Finance's public chart endpoint.

    Registered as a redundant alternative to BrapiStockExtractor, not a
    replacement: both independently bottom out around 2000-01-03 for B3
    tickers like PETR4, so switching doesn't unlock more history. Useful as
    a fallback if brapi.dev is unavailable or changes its free tier.

    B3 tickers need the ".SA" suffix on Yahoo (e.g. PETR4 -> PETR4.SA);
    override via asset.params["symbol"] for tickers that don't follow the
    `{id}.SA` convention.
    """

    source = "yahoo"

    def fetch_raw(self, asset: AssetConfig) -> dict:
        symbol = asset.params.get("symbol", f"{asset.id}.SA")
        params = {
            "range": asset.params.get("range", "5d"),
            "interval": asset.params.get("interval", "1d"),
        }
        response = requests.get(
            f"{CHART_URL}/{symbol}",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def normalize(self, raw: dict, asset: AssetConfig) -> list[InvestmentHistoryRecord]:
        results = (raw.get("chart") or {}).get("result") or []
        if not results:
            return []

        result = results[0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote = (indicators.get("quote") or [{}])[0]
        adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []
        currency = (result.get("meta") or {}).get("currency", "BRL")
        ingestion_ts = datetime.now(timezone.utc)

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        records = []
        for i, ts in enumerate(timestamps):
            close = closes[i] if i < len(closes) else None
            if close is None:
                # Yahoo emits a null row for non-trading days within the range
                continue
            records.append(
                InvestmentHistoryRecord(
                    event_date=datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                    investment_type=self.investment_type,
                    investment_id=asset.id,
                    source=self.source,
                    currency=currency,
                    open=opens[i] if i < len(opens) else None,
                    high=highs[i] if i < len(highs) else None,
                    low=lows[i] if i < len(lows) else None,
                    close=close,
                    volume=volumes[i] if i < len(volumes) else None,
                    extra={
                        "adjusted_close": adjclose[i] if i < len(adjclose) else None,
                        "exchange": asset.exchange,
                    },
                    ingestion_ts=ingestion_ts,
                )
            )
        return records
