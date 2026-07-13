from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.schemas import AssetConfig, InvestmentHistoryRecord
from src.storage.raw_storage import RawStorage, get_raw_storage


class BaseExtractor(ABC):
    """Contract shared by every extractor, regardless of investment type.

    Type-level subclasses (BaseStockExtractor, BaseCryptoExtractor, ...) fix
    `investment_type` and rules common to that type; source-level subclasses
    (BrapiStockExtractor, CoinGeckoCryptoExtractor, ...) implement the
    integration with a specific API.
    """

    investment_type: str
    source: str

    def __init__(self, raw_storage: RawStorage | None = None):
        self.raw_storage = raw_storage or get_raw_storage()

    @abstractmethod
    def fetch_raw(self, asset: AssetConfig) -> dict:
        """Call the external API and return the raw payload (already as dict/JSON)."""
        ...

    @abstractmethod
    def normalize(self, raw: dict, asset: AssetConfig) -> list[InvestmentHistoryRecord]:
        """Convert the raw payload into the common InvestmentHistoryRecord contract."""
        ...

    def extract(
        self, asset: AssetConfig, param_overrides: dict | None = None
    ) -> list[InvestmentHistoryRecord]:
        """Run fetch_raw + normalize for a single asset.

        `param_overrides` merges on top of the asset's configured params for
        this call only (e.g. a backfill run asking for range="max" instead
        of the watchlist's usual short incremental range), without mutating
        the original AssetConfig.
        """
        effective_asset = asset
        if param_overrides:
            effective_asset = asset.model_copy(
                update={"params": {**asset.params, **param_overrides}}
            )

        raw = self.fetch_raw(effective_asset)
        raw_uri = self.raw_storage.persist(raw, effective_asset, self.investment_type)
        records = self.normalize(raw, effective_asset)
        for record in records:
            record.raw_uri = raw_uri
        return records
