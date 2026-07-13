"""Single entrypoint for the Docker image. Each Cloud Run Job passes the
investment type as an argument (e.g. `python -m src.pipelines.main stock`),
reusing the same image for every pipeline. Pass `--backfill` to pull full
history instead of the regular short incremental window.

Usage:
    python -m src.pipelines.main <investment_type> [--backfill]
"""
from __future__ import annotations

import sys

from src.pipelines.backfill import run_backfill
from src.pipelines.runner import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python -m src.pipelines.main <investment_type> [--backfill]")

    investment_type = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "--backfill":
        run_backfill(investment_type)
    else:
        run_pipeline(investment_type)
