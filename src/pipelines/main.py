"""Single entrypoint for the Docker image. Each Cloud Run Job passes the
investment type as an argument (e.g. `python -m src.pipelines.main stock`),
reusing the same image for every pipeline. Pass `--backfill` to pull full
history instead of the regular short incremental window.

`news` is a different kind of job -- not an investment type, but the daily
news-sentiment backfill (see src/ml/news_data.py). It shares the image and
entrypoint pattern for deployment simplicity even though it isn't a price
pipeline.

Usage:
    python -m src.pipelines.main <investment_type> [--backfill]
    python -m src.pipelines.main news
"""
from __future__ import annotations

import os
import sys

from src.pipelines.backfill import run_backfill
from src.pipelines.runner import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python -m src.pipelines.main <investment_type|news> [--backfill]")

    mode = sys.argv[1]
    if mode == "news":
        from src.ml.news_data import run_daily_backfill

        api_key = os.environ["GEMINI_API_KEY"]
        result = run_daily_backfill(api_key)
        print(result)
    elif len(sys.argv) == 3 and sys.argv[2] == "--backfill":
        run_backfill(mode)
    else:
        run_pipeline(mode)
