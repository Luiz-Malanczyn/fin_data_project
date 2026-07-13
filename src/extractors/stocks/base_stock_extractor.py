from __future__ import annotations

from src.extractors.base import BaseExtractor


class BaseStockExtractor(BaseExtractor):
    """Behavior shared by every stock extractor.

    Fixes investment_type='stock'. Different sources (brapi, yfinance, ...)
    inherit from this class and implement fetch_raw/normalize.
    """

    investment_type = "stock"
