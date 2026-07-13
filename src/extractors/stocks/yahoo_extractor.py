from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

from src.extractors.stocks.base_stock_extractor import BaseStockExtractor
from src.models.schemas import AssetConfig, InvestmentHistoryRecord

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
FULL_HISTORY_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
FULL_HISTORY_CHUNK_YEARS = 9
MAX_CHUNK_RETRIES = 3


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
        interval = asset.params.get("interval", "1d")
        range_param = asset.params.get("range", "5d")

        if range_param == "max":
            # Yahoo silently downgrades range=max&interval=1d to monthly
            # bars on long histories (confirmed on both ^BVSP and individual
            # B3 stocks) -- explicit period1/period2 windows keep it honest.
            return self._fetch_full_history(symbol, interval)

        response = requests.get(
            f"{CHART_URL}/{symbol}",
            params={"range": range_param, "interval": interval},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_full_history(self, symbol: str, interval: str) -> dict:
        chunk_start = FULL_HISTORY_START
        now = datetime.now(timezone.utc)

        timestamps: list[int] = []
        quote = {"open": [], "high": [], "low": [], "close": [], "volume": []}
        adjclose: list[float | None] = []
        meta: dict = {}

        while chunk_start <= now:
            chunk_end = min(chunk_start.replace(year=chunk_start.year + FULL_HISTORY_CHUNK_YEARS), now)

            # Batch-fetching many chunks/symbols back to back occasionally
            # gets a soft-throttled 200 response with an empty `result` --
            # no exception to catch, just missing data -- so retry on that
            # too, not only on HTTP errors.
            results: list = []
            for attempt in range(1, MAX_CHUNK_RETRIES + 1):
                response = requests.get(
                    f"{CHART_URL}/{symbol}",
                    params={
                        "period1": int(chunk_start.timestamp()),
                        "period2": int(chunk_end.timestamp()),
                        "interval": interval,
                    },
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=30,
                )
                response.raise_for_status()
                results = (response.json().get("chart") or {}).get("result") or []
                if results and results[0].get("timestamp"):
                    break
                if attempt < MAX_CHUNK_RETRIES:
                    time.sleep(1.5 * attempt)

            if results:
                result = results[0]
                meta = meta or (result.get("meta") or {})
                chunk_timestamps = result.get("timestamp") or []
                chunk_quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
                chunk_adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get(
                    "adjclose"
                ) or []

                timestamps.extend(chunk_timestamps)
                for key in quote:
                    values = chunk_quote.get(key) or [None] * len(chunk_timestamps)
                    quote[key].extend(values)
                adjclose.extend(chunk_adjclose or [None] * len(chunk_timestamps))

            chunk_start = chunk_end + timedelta(days=1)
            if chunk_start <= now:
                time.sleep(0.3)  # light pacing between chunks/symbols to avoid soft throttling

        return {
            "chart": {
                "result": [
                    {
                        "meta": meta,
                        "timestamp": timestamps,
                        "indicators": {"quote": [quote], "adjclose": [{"adjclose": adjclose}]},
                    }
                ]
            }
        }

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
