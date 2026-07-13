from __future__ import annotations

from src.extractors.base import BaseExtractor


class BaseCryptoExtractor(BaseExtractor):
    """Behavior shared by every crypto extractor.

    Fixes investment_type='crypto'. Different sources (coingecko, binance, ...)
    inherit from this class and implement fetch_raw/normalize.
    """

    investment_type = "crypto"
