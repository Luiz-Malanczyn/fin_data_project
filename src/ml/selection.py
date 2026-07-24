"""Portfolio selection layer: which (asset, horizon) combos have shown a
real, backtested edge worth actually acting on -- not just statistical
significance -- and what the trained model says to do about them today.

Motivated by the plateau across v7/v9: adding more data sources kept
raising the count of statistically significant findings (48 -> 56), but
the walk-forward backtest win rate stayed flat at 11/39 both times. Most
of the 39 (asset, horizon) combos have never shown a real edge no matter
how many features get thrown at them. Rather than keep treating all 39
equally, this surfaces only the ones with an actual backtest track
record, so a recommendation means something instead of being one of 39
predictions with unknown reliability.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud import bigquery

from src.config.settings import settings
from src.config.watchlist_loader import load_watchlist
from src.ml.backtest import HORIZONS, choose_model_name
from src.ml.evaluation import evaluate_matured_predictions
from src.ml.predict import predict_ensemble, predict_horizon
from src.ml.storage import load_backtest_result

logger = logging.getLogger(__name__)

RECOMMENDATIONS_TABLE = "daily_recommendations"

_RECOMMENDATIONS_SCHEMA = [
    bigquery.SchemaField("run_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("investment_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("horizon", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("direction", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("last_known_date", "DATE"),
    bigquery.SchemaField("last_known_close", "FLOAT64"),
    bigquery.SchemaField("target_date", "DATE"),
    bigquery.SchemaField("predicted_close", "FLOAT64"),
    bigquery.SchemaField("predicted_change_pct", "FLOAT64"),
    bigquery.SchemaField("backtest_strategy_return", "FLOAT64"),
    bigquery.SchemaField("backtest_buy_hold_return", "FLOAT64"),
    bigquery.SchemaField("backtest_n_trades", "INT64"),
    bigquery.SchemaField("backtest_n_blocks", "INT64"),
]


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project or None)


def _table_ref() -> str:
    return f"{settings.gcp_project}.{settings.bq_dataset}.{RECOMMENDATIONS_TABLE}"


def _ensure_recommendations_table() -> None:
    table = bigquery.Table(_table_ref(), schema=_RECOMMENDATIONS_SCHEMA)
    _bq_client().create_table(table, exists_ok=True)


def save_recommendations(recommendations: list[dict]) -> None:
    """Appends one row per recommendation, tagged with when the run
    happened -- append-only like news_backfill_progress (see news_data.py
    for why: BigQuery refuses UPDATE/DELETE on rows still in the streaming
    buffer, up to ~90 minutes after insert, so there's nothing to gain
    from trying to overwrite "yesterday's" row instead of just adding a
    new one). Readers should filter to MAX(run_at) for the latest state,
    same pattern as news_backfill_progress's checkpoint read.
    """
    if not recommendations:
        return
    run_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "run_at": run_at,
            "investment_id": r["investment_id"],
            "horizon": r["horizon"],
            "model_name": r["model_name"],
            "direction": r["direction"],
            "last_known_date": r["last_known_date"],
            "last_known_close": r["last_known_close"],
            "target_date": r["target_date"],
            "predicted_close": r["predicted_close"],
            "predicted_change_pct": r["predicted_change_pct"],
            "backtest_strategy_return": r["backtest_strategy_return"],
            "backtest_buy_hold_return": r["backtest_buy_hold_return"],
            "backtest_n_trades": r["backtest_n_trades"],
            "backtest_n_blocks": r["backtest_n_blocks"],
        }
        for r in recommendations
    ]
    errors = _bq_client().insert_rows_json(_table_ref(), rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors saving recommendations: {errors}")


def run_daily_recommendations(min_trades: int = 10) -> list[dict]:
    """Entry point for the daily Cloud Run Job: score yesterday's (or
    older) predictions against what actually happened once their
    target_date has arrived, then regenerate today's recommendations from
    the currently-saved models (no retraining -- just a fresh live
    prediction against today's price data) and persist both, so anyone
    can read the latest state from BigQuery instead of needing this
    script run interactively. Evaluation runs first and is best-effort:
    a bug there shouldn't block today's recommendations from being saved.
    """
    try:
        evaluated = evaluate_matured_predictions()
        logger.info("Evaluated %d matured predictions", len(evaluated))
    except Exception:
        logger.exception("Evaluating matured predictions failed; continuing with today's recommendations")

    _ensure_recommendations_table()
    recommendations = get_recommendations(min_trades=min_trades)
    save_recommendations(recommendations)
    return recommendations


def get_qualified_combos(min_trades: int = 10) -> list[dict]:
    """(asset, horizon) combos where the best available backtested
    strategy beat buy-and-hold, with enough trades behind it that it
    isn't a fluke from a handful of lucky calls.

    "Best available" compares the single best statistically significant
    model against the ensemble of all significant models for that combo
    (see backtest.backtest_ensemble) when both have a persisted backtest
    result, and keeps whichever scored the higher walk-forward return --
    an experiment found the ensemble beats the single model in roughly a
    third of combos that have 2+ significant models to combine, and pushes
    a couple of combos over the beats_buy_hold line that the single model
    alone didn't reach. Since both share the same buy-and-hold reference
    (same asset, same horizon, same backtest folds), comparing their raw
    strategy_cumulative_return is enough to know which one also beats it.

    Requires a persisted backtest result (src.ml.backtest must have run
    and saved one via storage.save_backtest_result) -- combos with none
    yet are skipped, not treated as disqualified.
    """
    assets = [a for a in load_watchlist() if a.active]
    qualified = []
    for asset in assets:
        for horizon in HORIZONS:
            model_name = choose_model_name(asset.id, horizon)
            candidates = [
                r
                for r in (
                    load_backtest_result(asset.id, horizon, model_name),
                    load_backtest_result(asset.id, horizon, "ensemble"),
                )
                if r is not None
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: r["strategy_cumulative_return"])
            if best["beats_buy_hold"] and best["n_trades"] >= min_trades:
                qualified.append(best)
    return qualified


def get_recommendations(min_trades: int = 10) -> list[dict]:
    """Today's live prediction for every qualified combo, with its
    backtest track record attached -- the output actually meant to be
    acted on, instead of the full 39-combo grid where most entries have
    no demonstrated edge behind them.
    """
    import gc

    recommendations = []
    for combo in get_qualified_combos(min_trades=min_trades):
        if combo["model_name"] == "ensemble":
            pred = predict_ensemble(combo["investment_id"], combo["horizon"], combo["member_models"])
        else:
            pred = predict_horizon(combo["investment_id"], combo["horizon"], combo["model_name"])
        recommendations.append(
            {
                **pred,
                "direction": "alta" if pred["predicted_change_pct"] > 0 else "baixa",
                "backtest_strategy_return": combo["strategy_cumulative_return"],
                "backtest_buy_hold_return": combo["buy_hold_cumulative_return"],
                "backtest_n_trades": combo["n_trades"],
                "backtest_n_blocks": combo["n_blocks"],
            }
        )
        # predict_horizon() pulls a ticker's *entire* price history plus
        # macro/fundamentals/news joins into pandas DataFrames and
        # deserializes a fresh model (some are multi-estimator tree
        # ensembles) on every call, none of which gets released between
        # combos on its own -- across 11 sequential predictions that was
        # enough to OOM-kill the Cloud Run job even at 2Gi, well past
        # what a single combo needs. Forcing collection here trades a
        # little time for not accumulating 11 combos' worth of live data
        # at once.
        gc.collect()
    recommendations.sort(key=lambda r: -r["backtest_strategy_return"])
    return recommendations


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    for rec in run_daily_recommendations():
        print(
            f"{rec['investment_id']:6s} {rec['horizon']:8s} {rec['direction']:6s} "
            f"previsto={rec['predicted_change_pct']:+.2f}% "
            f"(historico: estrategia={rec['backtest_strategy_return']*100:+.1f}% "
            f"vs buy&hold={rec['backtest_buy_hold_return']*100:+.1f}% "
            f"em {rec['backtest_n_trades']}/{rec['backtest_n_blocks']} operacoes)"
        )
