"""One-off entrypoint: pull the full available price history for every
active stock in the watchlist and load it into BigQuery. Run this once to
seed investment_history before switching a stock to the regular short
incremental schedule (run_stocks.py).

Usage:
    python -m src.pipelines.backfill_stocks
"""
from __future__ import annotations

from src.pipelines.backfill import run_backfill

if __name__ == "__main__":
    run_backfill("stock")
