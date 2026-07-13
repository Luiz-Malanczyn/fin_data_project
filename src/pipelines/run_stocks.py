"""Stocks pipeline entrypoint. Triggered by Cloud Scheduler after market
close (e.g. once a day), via a Cloud Run Job. Pulls a short incremental
range (see watchlist.yaml) -- use backfill_stocks.py for full history.
"""
from __future__ import annotations

from src.pipelines.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline("stock")
