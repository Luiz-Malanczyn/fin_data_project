"""Central configuration for the framework, read from environment variables.

Defaults point to local execution (no GCP), so extraction/normalization can
be validated before pointing at a real BigQuery/GCS environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Watchlist
    watchlist_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("WATCHLIST_PATH", REPO_ROOT / "src" / "config" / "watchlist.yaml")
        )
    )

    # Storage (bronze / raw payloads)
    storage_backend: str = field(default_factory=lambda: os.environ.get("STORAGE_BACKEND", "local"))
    raw_bucket: str = field(default_factory=lambda: os.environ.get("RAW_BUCKET", "fin-data-raw"))
    local_raw_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("LOCAL_RAW_DIR", REPO_ROOT / "data" / "raw"))
    )

    # Load (gold / BigQuery)
    load_backend: str = field(default_factory=lambda: os.environ.get("LOAD_BACKEND", "local"))
    gcp_project: str = field(default_factory=lambda: os.environ.get("GCP_PROJECT", ""))
    bq_dataset: str = field(default_factory=lambda: os.environ.get("BQ_DATASET", "fin_data"))
    bq_history_table: str = field(default_factory=lambda: os.environ.get("BQ_HISTORY_TABLE", "investment_history"))
    local_output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("LOCAL_OUTPUT_DIR", REPO_ROOT / "data" / "local_output"))
    )

    # External API credentials
    brapi_token: str = field(default_factory=lambda: os.environ.get("BRAPI_TOKEN", ""))


settings = Settings()
