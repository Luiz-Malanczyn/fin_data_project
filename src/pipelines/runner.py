from __future__ import annotations

import logging

from src.config.watchlist_loader import filter_by_type, load_watchlist
from src.extractors.registry import get_extractor
from src.loaders.registry import get_loader
from src.models.schemas import InvestmentHistoryRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(investment_type: str, param_overrides: dict | None = None) -> int:
    """Run the pipeline for one investment type: read the watchlist, filter
    the active assets of that type, extract each one with its configured
    source and load everything into the destination (BigQuery or local CSV,
    depending on LOAD_BACKEND).

    `param_overrides` is forwarded to each extractor call, letting a caller
    (e.g. a backfill run) request a wider date range than what the watchlist
    normally uses for incremental runs.
    """
    assets = filter_by_type(load_watchlist(), investment_type)
    if not assets:
        logger.warning("No active asset of type '%s' in the watchlist.", investment_type)
        return 0

    all_records: list[InvestmentHistoryRecord] = []
    for asset in assets:
        logger.info("Extracting %s (%s/%s)...", asset.id, asset.type, asset.source)
        extractor = get_extractor(asset.type, asset.source)
        try:
            records = extractor.extract(asset, param_overrides=param_overrides)
        except Exception:
            logger.exception("Failed to extract %s via %s", asset.id, asset.source)
            continue
        logger.info("  -> %d normalized records", len(records))
        all_records.extend(records)

    loader = get_loader()
    loaded = loader.load(all_records)
    logger.info("Load complete: %d records written via %s", loaded, type(loader).__name__)
    return loaded
