"""One-off full-history backfill.

Regular scheduled runs (run_stocks.py, run_crypto.py) only pull a short,
cheap incremental window -- see the `range`/`days` params in watchlist.yaml.
Backfill overrides those params to pull the entire history a source can
provide, for every active asset of the given type, and loads it through the
same idempotent MERGE used by incremental runs (so it's safe to run once to
seed the table, or again later to re-pull full history).
"""
from __future__ import annotations

from src.pipelines.runner import run_pipeline

# Per-type param override that asks each source for its full available
# history instead of the watchlist's usual short incremental window.
FULL_HISTORY_OVERRIDES: dict[str, dict] = {
    "stock": {"range": "max"},   # brapi.dev: max daily history available, free tier
    "crypto": {"days": 365},     # CoinGecko: days="max" needs a paid plan; 365 is the free-tier ceiling
}


def run_backfill(investment_type: str) -> int:
    overrides = FULL_HISTORY_OVERRIDES.get(investment_type)
    if overrides is None:
        raise ValueError(f"No full-history override configured for type={investment_type!r}")
    return run_pipeline(investment_type, param_overrides=overrides)
