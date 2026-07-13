from __future__ import annotations

import pandas as pd

from src.config.settings import settings
from src.loaders.base_loader import BaseLoader
from src.models.schemas import InvestmentHistoryRecord


class LocalCsvLoader(BaseLoader):
    """Validation loader: writes to a local CSV instead of BigQuery.

    Used during development to visually check normalized data (e.g. PETR4,
    MGLU3, bitcoin) before pointing the pipeline at a real GCP environment.
    Deduplicates on (event_date, investment_type, investment_id, source),
    the same key used by BigQueryLoader's MERGE.
    """

    KEY_COLUMNS = ["event_date", "investment_type", "investment_id", "source"]

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or settings.local_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, records: list[InvestmentHistoryRecord]) -> int:
        if not records:
            return 0

        df_new = pd.DataFrame([r.model_dump(mode="json") for r in records])
        out_path = self.output_dir / f"{settings.bq_history_table}.csv"

        if out_path.exists():
            df_existing = pd.read_csv(out_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=self.KEY_COLUMNS, keep="last")
        else:
            df_combined = df_new

        df_combined.to_csv(out_path, index=False)
        return len(df_new)
