from __future__ import annotations

import json
import uuid

import pandas as pd
from google.cloud import bigquery

from src.config.settings import settings
from src.loaders.base_loader import BaseLoader
from src.models.schemas import InvestmentHistoryRecord

MERGE_KEY_COLUMNS = ["event_date", "investment_type", "investment_id", "source"]

TARGET_SCHEMA = [
    bigquery.SchemaField("event_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("investment_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("investment_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("open", "FLOAT64"),
    bigquery.SchemaField("high", "FLOAT64"),
    bigquery.SchemaField("low", "FLOAT64"),
    bigquery.SchemaField("close", "FLOAT64"),
    bigquery.SchemaField("volume", "FLOAT64"),
    bigquery.SchemaField("extra", "JSON"),
    bigquery.SchemaField("ingestion_ts", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("raw_uri", "STRING"),
]

# The BigQuery client loads dataframes as Parquet under the hood, which does
# not support the JSON column type directly ("Unsupported field type: JSON").
# The staging table instead holds `extra` as a JSON-encoded STRING, and the
# MERGE casts it to JSON with PARSE_JSON() when writing into the target table.
STAGING_SCHEMA = [
    field if field.name != "extra" else bigquery.SchemaField("extra", "STRING")
    for field in TARGET_SCHEMA
]


class BigQueryLoader(BaseLoader):
    """Loads records into the unified `investment_history` table via MERGE.

    Uses a temporary staging table + MERGE on the key
    (event_date, investment_type, investment_id, source), which guarantees
    that running the same extraction twice (reprocessing, retries) never
    duplicates history rows.
    """

    def __init__(self, client: bigquery.Client | None = None):
        self.client = client or bigquery.Client(project=settings.gcp_project or None)
        self.dataset = settings.bq_dataset
        self.table = settings.bq_history_table

    def load(self, records: list[InvestmentHistoryRecord]) -> int:
        if not records:
            return 0

        # `investment_history` is DAY-partitioned on event_date, and BigQuery
        # rejects a single DML job that touches more than 4000 partitions.
        # A full-history backfill of a single stock can easily span more than
        # 4000 distinct trading days, so records are merged one calendar year
        # at a time (at most ~366 partitions per batch) instead of all at once.
        total = 0
        for year_records in self._group_by_year(records).values():
            total += self._load_batch(year_records)
        return total

    def _group_by_year(
        self, records: list[InvestmentHistoryRecord]
    ) -> dict[int, list[InvestmentHistoryRecord]]:
        groups: dict[int, list[InvestmentHistoryRecord]] = {}
        for record in records:
            groups.setdefault(record.event_date.year, []).append(record)
        return groups

    def _load_batch(self, records: list[InvestmentHistoryRecord]) -> int:
        df = self._to_dataframe(records)
        # A source can report the same day twice under one merge key (e.g.
        # CoinGecko returning both a rounded and a "right now" timestamp for
        # the current day). BigQuery's MERGE rejects a staging table with
        # duplicate keys ("must match at most one source row for each target
        # row"), so duplicates are collapsed here, keeping the latest one.
        df = df.drop_duplicates(subset=MERGE_KEY_COLUMNS, keep="last")
        staging_table_id = (
            f"{self.client.project}.{self.dataset}._staging_{self.table}_{uuid.uuid4().hex[:8]}"
        )

        job_config = bigquery.LoadJobConfig(schema=STAGING_SCHEMA, write_disposition="WRITE_TRUNCATE")
        load_job = self.client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
        load_job.result()

        try:
            self._merge(staging_table_id)
        finally:
            self.client.delete_table(staging_table_id, not_found_ok=True)

        return len(df)

    def _to_dataframe(self, records: list[InvestmentHistoryRecord]) -> pd.DataFrame:
        rows = []
        for record in records:
            row = record.model_dump(mode="json")
            row["extra"] = json.dumps(row.get("extra") or {})
            rows.append(row)
        return pd.DataFrame(rows)

    def _merge(self, staging_table_id: str) -> None:
        target = f"{self.client.project}.{self.dataset}.{self.table}"
        on_clause = " AND ".join(f"T.{c} = S.{c}" for c in MERGE_KEY_COLUMNS)

        def source_expr(column: str) -> str:
            # wide_number_mode='round' avoids PARSE_JSON erroring on large
            # floats (e.g. crypto market cap in the trillions) that can't
            # round-trip exactly through its default number handling.
            return (
                "PARSE_JSON(S.extra, wide_number_mode => 'round')"
                if column == "extra"
                else f"S.{column}"
            )

        update_columns = [f.name for f in TARGET_SCHEMA if f.name not in MERGE_KEY_COLUMNS]
        update_clause = ", ".join(f"{c} = {source_expr(c)}" for c in update_columns)
        insert_columns = [f.name for f in TARGET_SCHEMA]

        query = f"""
        MERGE `{target}` T
        USING `{staging_table_id}` S
        ON {on_clause}
        WHEN MATCHED THEN
          UPDATE SET {update_clause}
        WHEN NOT MATCHED THEN
          INSERT ({", ".join(insert_columns)})
          VALUES ({", ".join(source_expr(c) for c in insert_columns)})
        """
        self.client.query(query).result()
