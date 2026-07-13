from __future__ import annotations

from src.extractors.base import BaseExtractor
from src.extractors.crypto.binance_extractor import BinanceCryptoExtractor
from src.extractors.crypto.coingecko_extractor import CoinGeckoCryptoExtractor
from src.extractors.stocks.brapi_extractor import BrapiStockExtractor
from src.extractors.stocks.yahoo_extractor import YahooFinanceStockExtractor

# (investment_type, source) -> extractor class.
# Registering a new source only requires adding an entry here, no pipeline changes.
EXTRACTOR_REGISTRY: dict[tuple[str, str], type[BaseExtractor]] = {
    ("stock", "brapi"): BrapiStockExtractor,
    ("stock", "yahoo"): YahooFinanceStockExtractor,
    ("crypto", "coingecko"): CoinGeckoCryptoExtractor,
    ("crypto", "binance"): BinanceCryptoExtractor,
}


def get_extractor(investment_type: str, source: str) -> BaseExtractor:
    key = (investment_type, source)
    if key not in EXTRACTOR_REGISTRY:
        raise ValueError(f"No extractor registered for type={investment_type!r} source={source!r}")
    return EXTRACTOR_REGISTRY[key]()
