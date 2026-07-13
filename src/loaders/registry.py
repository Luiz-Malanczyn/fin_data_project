from __future__ import annotations

from src.config.settings import settings
from src.loaders.base_loader import BaseLoader


def get_loader() -> BaseLoader:
    if settings.load_backend == "bigquery":
        from src.loaders.bigquery_loader import BigQueryLoader

        return BigQueryLoader()

    from src.loaders.local_loader import LocalCsvLoader

    return LocalCsvLoader()
