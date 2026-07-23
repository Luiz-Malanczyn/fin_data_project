from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config.settings import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


def _horizon_dir(investment_id: str, horizon: str) -> Path:
    horizon_dir = MODELS_DIR / investment_id / horizon
    horizon_dir.mkdir(parents=True, exist_ok=True)
    return horizon_dir


def save_model(
    investment_id: str, horizon: str, model_name: str, model, metrics: dict, feature_names: list[str]
) -> Path:
    """Persist a fitted model alongside a metadata sidecar recording the
    metrics it scored and the exact feature columns it was trained on --
    so a future retrain with more variables can never be silently loaded
    against the wrong feature set. Namespaced by horizon (daily/weekly/
    monthly) since each is trained on a different target and isn't
    comparable to the others.
    """
    horizon_dir = _horizon_dir(investment_id, horizon)

    model_path = horizon_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "investment_id": investment_id,
        "horizon": horizon,
        "model_name": model_name,
        "metrics": metrics,
        "feature_names": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (horizon_dir / f"{model_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(investment_id: str, horizon: str, model_name: str):
    horizon_dir = _horizon_dir(investment_id, horizon)
    model = joblib.load(horizon_dir / f"{model_name}.joblib")
    metadata = json.loads((horizon_dir / f"{model_name}.json").read_text(encoding="utf-8"))
    return model, metadata


def load_all_metadata(investment_id: str, horizon: str) -> list[dict]:
    horizon_dir = _horizon_dir(investment_id, horizon)
    # Excludes *.backtest.json sidecars (see save_backtest_result) -- both
    # live in the same directory and both end in .json, so a bare "*.json"
    # glob here would silently pull backtest-result dicts (no "metrics"
    # key) into what every caller assumes is a list of training metadata.
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(horizon_dir.glob("*.json"))
        if not path.name.endswith(".backtest.json")
    ]


def best_model_name(investment_id: str, horizon: str, metric: str = "mae") -> str | None:
    metadata_list = load_all_metadata(investment_id, horizon)
    if not metadata_list:
        return None

    lower_is_better = metric in {"mae", "rmse", "mape"}
    key = (lambda m: m["metrics"][metric])
    best = min(metadata_list, key=key) if lower_is_better else max(metadata_list, key=key)
    return best["model_name"]


def save_backtest_result(investment_id: str, horizon: str, model_name: str, result: dict) -> None:
    """Persists a walk-forward backtest result alongside the model it
    tested -- previously computed by backtest.py and only ever printed,
    never saved, so there was no durable record of which (asset, horizon)
    combos actually showed a real, tradeable edge versus just a
    statistically significant one. Selection logic (src/ml/selection.py)
    reads this instead of re-running the backtest every time.
    """
    horizon_dir = _horizon_dir(investment_id, horizon)
    (horizon_dir / f"{model_name}.backtest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def load_backtest_result(investment_id: str, horizon: str, model_name: str) -> dict | None:
    path = _horizon_dir(investment_id, horizon) / f"{model_name}.backtest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
