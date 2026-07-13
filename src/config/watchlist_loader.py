from __future__ import annotations

import yaml

from src.config.settings import settings
from src.models.schemas import AssetConfig


def load_watchlist(path=None) -> list[AssetConfig]:
    """Read watchlist.yaml and return the list of configured assets."""
    path = path or settings.watchlist_path
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [AssetConfig(**item) for item in raw.get("assets", [])]


def filter_by_type(assets: list[AssetConfig], investment_type: str, active_only: bool = True) -> list[AssetConfig]:
    return [
        a for a in assets
        if a.type == investment_type and (not active_only or a.active)
    ]
