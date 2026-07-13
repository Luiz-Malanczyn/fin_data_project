from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from src.config.settings import settings
from src.models.schemas import AssetConfig


class RawStorage(ABC):
    """Landing zone (bronze) for the raw payload of each API call.

    Keeping the raw payload allows reprocessing/normalizing again without
    re-calling the API (important given the tight rate limits of free
    tiers) and provides traceability (each BigQuery row points back to the
    raw_uri that produced it).
    """

    @abstractmethod
    def persist(self, raw: dict, asset: AssetConfig, investment_type: str) -> str:
        """Persist the raw payload and return a URI/identifier for the saved object."""
        ...


def _object_path(asset: AssetConfig, investment_type: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{investment_type}/{asset.source}/{asset.id}/{ts}.json"


class LocalRawStorage(RawStorage):
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or settings.local_raw_dir

    def persist(self, raw: dict, asset: AssetConfig, investment_type: str) -> str:
        rel_path = _object_path(asset, investment_type)
        full_path = self.base_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return f"file://{full_path.as_posix()}"


class GCSRawStorage(RawStorage):
    def __init__(self, bucket_name=None):
        from google.cloud import storage  # lazy import: no credentials needed for local dev

        self.bucket_name = bucket_name or settings.raw_bucket
        self._client = storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)

    def persist(self, raw: dict, asset: AssetConfig, investment_type: str) -> str:
        rel_path = _object_path(asset, investment_type)
        blob = self._bucket.blob(rel_path)
        blob.upload_from_string(json.dumps(raw, ensure_ascii=False), content_type="application/json")
        return f"gs://{self.bucket_name}/{rel_path}"


def get_raw_storage() -> RawStorage:
    if settings.storage_backend == "gcs":
        return GCSRawStorage()
    return LocalRawStorage()
