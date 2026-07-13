from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config.settings import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


def _asset_dir(investment_id: str) -> Path:
    asset_dir = MODELS_DIR / investment_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    return asset_dir


def save_model(investment_id: str, model_name: str, model, metrics: dict, feature_names: list[str]) -> Path:
    """Persist a fitted model alongside a metadata sidecar recording the
    metrics it scored and the exact feature columns it was trained on --
    so a future retrain with more variables can never be silently loaded
    against the wrong feature set.
    """
    asset_dir = _asset_dir(investment_id)

    model_path = asset_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "investment_id": investment_id,
        "model_name": model_name,
        "metrics": metrics,
        "feature_names": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (asset_dir / f"{model_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(investment_id: str, model_name: str):
    asset_dir = _asset_dir(investment_id)
    model = joblib.load(asset_dir / f"{model_name}.joblib")
    metadata = json.loads((asset_dir / f"{model_name}.json").read_text(encoding="utf-8"))
    return model, metadata


def load_all_metadata(investment_id: str) -> list[dict]:
    asset_dir = _asset_dir(investment_id)
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(asset_dir.glob("*.json"))]


def best_model_name(investment_id: str, metric: str = "mae") -> str | None:
    metadata_list = load_all_metadata(investment_id)
    if not metadata_list:
        return None

    lower_is_better = metric in {"mae", "rmse", "mape"}
    key = (lambda m: m["metrics"][metric])
    best = min(metadata_list, key=key) if lower_is_better else max(metadata_list, key=key)
    return best["model_name"]
