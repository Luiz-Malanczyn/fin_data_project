"""Closes the loop the backtest can't: did today's live predictions
actually come true? backtest.py validates walk-forward against historical
data, which is the right discipline for choosing a model, but it's not
proof the model still works under live conditions going forward -- feature
pipelines change, market regimes shift, and a combo can pass every
historical check and still decay. This compares daily_recommendations rows
whose target_date has already arrived against the realized close price,
once that price has actually landed in investment_history, and persists
whether the direction call was right.
"""
from __future__ import annotations

from datetime import datetime, timezone

from google.cloud import bigquery

from src.config.settings import settings
from src.ml.data import lookup_investment_type

EVALUATIONS_TABLE = "prediction_evaluations"

_EVALUATIONS_SCHEMA = [
    bigquery.SchemaField("evaluated_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("prediction_run_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("investment_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("horizon", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("target_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("predicted_close", "FLOAT64"),
    bigquery.SchemaField("predicted_change_pct", "FLOAT64"),
    bigquery.SchemaField("actual_close_date", "DATE"),
    bigquery.SchemaField("actual_close", "FLOAT64"),
    bigquery.SchemaField("actual_change_pct", "FLOAT64"),
    bigquery.SchemaField("predicted_direction", "STRING"),
    bigquery.SchemaField("actual_direction", "STRING"),
    bigquery.SchemaField("direction_correct", "BOOL"),
]


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project or None)


def _table_ref(table: str) -> str:
    return f"{settings.gcp_project}.{settings.bq_dataset}.{table}"


def _ensure_evaluations_table() -> None:
    table = bigquery.Table(_table_ref(EVALUATIONS_TABLE), schema=_EVALUATIONS_SCHEMA)
    _bq_client().create_table(table, exists_ok=True)


def find_unevaluated_matured_predictions() -> list[dict]:
    """Predictions whose target_date has already arrived and that don't
    already have a row in prediction_evaluations for the same
    (investment_id, horizon, run_at) -- the LEFT JOIN is what makes this
    safe to run every day without re-evaluating the same prediction twice.
    """
    query = f"""
        SELECT r.investment_id, r.horizon, r.model_name, r.run_at AS prediction_run_at,
               r.target_date, r.last_known_close, r.predicted_close, r.predicted_change_pct
        FROM `{_table_ref('daily_recommendations')}` r
        LEFT JOIN `{_table_ref(EVALUATIONS_TABLE)}` e
          ON r.investment_id = e.investment_id
         AND r.horizon = e.horizon
         AND r.run_at = e.prediction_run_at
        WHERE r.target_date < CURRENT_DATE()
          AND e.prediction_run_at IS NULL
    """
    return [dict(row.items()) for row in _bq_client().query(query).result()]


def _actual_close_on_or_after(investment_id: str, investment_type: str, target_date):
    """The realized close nearest on/after target_date. target_date is a
    business-day estimate (see features.target_date_for_horizon) that can
    land on a B3 holiday with no trading session, so this takes the
    closest available session on or after it rather than requiring an
    exact match. Returns None if the price pipeline hasn't caught up to
    that date yet -- the caller just retries on the next run.
    """
    query = f"""
        SELECT event_date, close
        FROM `{_table_ref(settings.bq_history_table)}`
        WHERE investment_id = @investment_id AND investment_type = @investment_type
          AND event_date >= @target_date
        ORDER BY event_date
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("investment_id", "STRING", investment_id),
            bigquery.ScalarQueryParameter("investment_type", "STRING", investment_type),
            bigquery.ScalarQueryParameter("target_date", "DATE", target_date),
        ]
    )
    rows = list(_bq_client().query(query, job_config=job_config).result())
    if not rows:
        return None
    return rows[0]["event_date"], float(rows[0]["close"])


def evaluate_matured_predictions() -> list[dict]:
    """Entry point for the daily job: score every matured, not-yet-scored
    prediction against what actually happened and persist the verdict.
    Best-effort per row -- a prediction whose target_date hasn't produced
    a trading session yet (pipeline lag, or a run right after a holiday)
    is simply left for the next day's run rather than failing the batch.
    """
    _ensure_evaluations_table()
    pending = find_unevaluated_matured_predictions()
    if not pending:
        return []

    evaluated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for pred in pending:
        investment_type = lookup_investment_type(pred["investment_id"])
        actual = _actual_close_on_or_after(pred["investment_id"], investment_type, pred["target_date"])
        if actual is None:
            continue
        actual_close_date, actual_close = actual
        last_known_close = pred["last_known_close"]
        actual_change_pct = (actual_close - last_known_close) / last_known_close * 100
        predicted_direction = "alta" if pred["predicted_change_pct"] > 0 else "baixa"
        actual_direction = "alta" if actual_change_pct > 0 else "baixa"

        rows.append(
            {
                "evaluated_at": evaluated_at,
                "prediction_run_at": pred["prediction_run_at"].isoformat(),
                "investment_id": pred["investment_id"],
                "horizon": pred["horizon"],
                "model_name": pred["model_name"],
                "target_date": pred["target_date"].isoformat(),
                "predicted_close": pred["predicted_close"],
                "predicted_change_pct": pred["predicted_change_pct"],
                "actual_close_date": actual_close_date.isoformat(),
                "actual_close": actual_close,
                "actual_change_pct": actual_change_pct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "direction_correct": predicted_direction == actual_direction,
            }
        )

    if rows:
        errors = _bq_client().insert_rows_json(_table_ref(EVALUATIONS_TABLE), rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors saving prediction evaluations: {errors}")
    return rows


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    results = evaluate_matured_predictions()
    correct = sum(1 for r in results if r["direction_correct"])
    print(f"Evaluated {len(results)} matured predictions ({correct}/{len(results)} correct direction)")
