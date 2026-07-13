from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.extractors.crypto.base_crypto_extractor import BaseCryptoExtractor
from src.models.schemas import AssetConfig, InvestmentHistoryRecord

KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000


class BinanceCryptoExtractor(BaseCryptoExtractor):
    """Extracts crypto history via Binance's public klines (candlestick) endpoint.

    Unlike CoinGecko's free tier (capped at 365 days of history), Binance
    serves full daily history back to a symbol's listing date at no cost and
    without an API key, so it's the source used for full-history backfills.

    Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """

    source = "binance"

    def fetch_raw(self, asset: AssetConfig) -> dict:
        symbol = asset.params.get("symbol", f"{asset.id.upper()}USDT")
        interval = asset.params.get("interval", "1d")

        if asset.params.get("full_history"):
            klines = self._fetch_full_history(symbol, interval)
        else:
            klines = self._fetch_recent(symbol, interval, asset.params.get("limit", 3))

        return {"symbol": symbol, "interval": interval, "klines": klines}

    def _fetch_recent(self, symbol: str, interval: str, limit: int) -> list:
        response = requests.get(
            KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_full_history(self, symbol: str, interval: str) -> list:
        all_klines: list = []
        start_time = 0  # Binance clamps this to the symbol's actual listing date
        while True:
            response = requests.get(
                KLINES_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": MAX_KLINES_PER_REQUEST,
                    "startTime": start_time,
                },
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            all_klines.extend(batch)
            if len(batch) < MAX_KLINES_PER_REQUEST:
                break
            start_time = batch[-1][0] + 1  # next page starts right after the last candle's open time
        return all_klines

    def normalize(self, raw: dict, asset: AssetConfig) -> list[InvestmentHistoryRecord]:
        ingestion_ts = datetime.now(timezone.utc)
        records = []
        for kline in raw.get("klines") or []:
            open_time_ms = kline[0]
            records.append(
                InvestmentHistoryRecord(
                    event_date=datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).date(),
                    investment_type=self.investment_type,
                    investment_id=asset.id,
                    source=self.source,
                    currency="usdt",
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    extra={
                        "symbol": raw.get("symbol"),
                        "quote_asset_volume": float(kline[7]),
                        "num_trades": kline[8],
                    },
                    ingestion_ts=ingestion_ts,
                )
            )
        return records
