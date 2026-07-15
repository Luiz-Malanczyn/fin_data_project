"""Historical news sentiment via GDELT (article discovery) + Gemini free tier
(relevance filtering and financial-impact scoring in one call).

Two problems that sank naive approaches during investigation, both solved
here:
  - GDELT's own entity extraction (V2Organizations) is too weak and/or too
    ambiguous for several B3 tickers (e.g. "Vale" collides with the common
    Portuguese word "worth") to search on directly. Fixed by casting a wide
    net (loose name match, restricted to known Brazilian finance domains)
    and letting the LLM call also answer "is this article actually about
    this company" as part of the same prompt that scores impact -- no
    separate disambiguation layer needed.
  - A generic sentiment classifier (tested: cardiffnlp/twitter-xlm-roberta,
    trained on tweets) reads financial journalism's measured, factual tone
    as "neutral" even when reporting bad news. An LLM asked directly "is
    this good or bad for the stock, and why" performs far better in manual
    testing (confidently correct on all 3 validation articles vs. weak
    'neutral' from the classifier on the same 3).

Scoring is rate-limited (free tier: ~15 requests/minute, and a harder
~500 requests/day cap discovered the hard way after a run spent hours
retrying against a daily quota that wasn't going to clear) and slow
relative to every other source in this project, so -- unlike macro_data.py
/ dividend_data.py's fetch-fresh-every-process pattern -- results are
persisted to a BigQuery table (news_sentiment_cache) keyed by URL, not an
in-memory or local-disk cache: this runs as a daily Cloud Run Job, a fresh
stateless container on every execution, so the cache has to live
somewhere that survives between runs. A second table
(news_backfill_progress) tracks which month to resume from.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests
import trafilatura
from google.cloud import bigquery

from src.config.settings import settings
from src.config.company_dimension import get_company, get_company_dimension


def _news_companies() -> pd.DataFrame:
    """Rows from the company_dimension table that participate in news
    search -- currently every stock (crypto has no news_search_term set;
    see seed_company_dimension.py). Per-ticker search term, URL-slug
    variants, and display name used to be three separate hardcoded dicts
    here (COMPANY_SEARCH_TERMS / COMPANY_URL_TERMS / COMPANY_DISPLAY_NAMES)
    that had to be kept in sync by hand; now they're columns on the same
    dimension row.
    """
    dim = get_company_dimension()
    return dim[dim["news_search_term"].notna()]


# Restricting candidate articles to known Brazilian finance/news domains
# before spending a rate-limited Gemini call on them -- cuts the noise the
# generic-word tickers (Vale, B3, WEG, Equatorial, Localiza) would otherwise
# drown in, without needing GDELT's own (weak) entity extraction to be
# precise.
FINANCE_DOMAINS = [
    "infomoney.com.br", "moneytimes.com.br", "exame.com", "suno.com.br",
    "seudinheiro.com", "valorinveste.globo.com", "einvestidor.estadao.com.br",
    "valor.globo.com", "investing.com", "estadao.com.br", "uol.com.br",
    "g1.globo.com", "cnnbrasil.com.br", "folha.uol.com.br", "oglobo.globo.com",
    "istoedinheiro.com.br", "braziljournal.com", "neofeed.com.br",
]

GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_MIN_INTERVAL_SECONDS = 4.5  # ~13/min, under the ~15/min free-tier cap

PROMPT_TEMPLATE = """Voce e um analista financeiro brasileiro. Leia a noticia abaixo e responda SOMENTE com um JSON valido, sem markdown, no formato exato:
{{"relevante": true ou false, "impacto": "positivo" ou "negativo" ou "neutro", "confianca": numero de 0 a 1, "motivo": "uma frase curta"}}

"relevante" = true somente se a noticia for genuinamente sobre a empresa {company} (ticker {ticker}) como negocio/acao -- responda false se o nome da empresa aparecer so por coincidencia (ex: a palavra usada com outro significado, ou citada de passagem sem ser o assunto).
"impacto" = o efeito esperado da noticia sobre o preco da acao da empresa, do ponto de vista de um investidor.

Noticia:
{article}
"""


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project or None)


def fetch_candidate_urls_all_tickers(
    start_date: date, end_date: date, max_results: int = 20000
) -> list[dict]:
    """Candidate article URLs from GDELT's public BigQuery GKG dataset for
    every stock in company_dimension, in one pass, restricted to known
    Brazilian finance domains. Deliberately loose on the name match (see
    _news_companies()) -- precision comes later, from the LLM relevance
    check.

    Combines every ticker into a single query rather than one query per
    ticker: BigQuery bills by column-bytes-read within the partition range,
    which barely changes whether the WHERE clause checks one name or
    twelve, so querying all tickers together instead of separately is
    roughly a 12x cost reduction for the same coverage.
    """
    domain_filter = " OR ".join(f"SourceCommonName LIKE '%{d}%'" for d in FINANCE_DOMAINS)
    companies = _news_companies()

    def _ticker_condition(row) -> str:
        url_conditions = " OR ".join(
            f"LOWER(DocumentIdentifier) LIKE '%{term}%'" for term in row.news_url_terms
        )
        return f"(LOWER(V2Organizations) LIKE '%{row.news_search_term}%' OR {url_conditions})"

    match_columns = ",\n      ".join(
        f"{_ticker_condition(row)} AS m_{row.investment_id}" for row in companies.itertuples()
    )
    any_match = " OR ".join(f"m_{ticker}" for ticker in companies["investment_id"])
    query = f"""
    SELECT * FROM (
      SELECT
        DATE, DocumentIdentifier, SourceCommonName,
        {match_columns}
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start_date) AND TIMESTAMP(@end_date)
        AND ({domain_filter})
    )
    WHERE {any_match}
    LIMIT {max_results}
    """
    job = _bq_client().query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date.isoformat()),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date.isoformat()),
            ]
        ),
    )
    rows = []
    for row in job.result():
        matched = [t for t in companies["investment_id"] if row[f"m_{t}"]]
        for ticker in matched:
            rows.append(
                {
                    "ticker": ticker,
                    "url": row.DocumentIdentifier,
                    "source": row.SourceCommonName,
                    "gdelt_date": str(row.DATE),
                }
            )
    return rows


def extract_article_text(url: str, timeout: int = 15) -> str | None:
    """Fetches and extracts the main body text of an article, handling the
    encoding-detection issue found in testing (some older Brazilian sites
    mis-declare or omit charset, corrupting accented characters if decoded
    as the wrong encoding)."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return trafilatura.extract(response.text)
    except Exception:
        return None


NEWS_TABLE = "news_sentiment_cache"
PROGRESS_TABLE = "news_backfill_progress"

_NEWS_SCHEMA = [
    bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("gdelt_date", "STRING"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("relevante", "BOOL"),
    bigquery.SchemaField("impacto", "STRING"),
    bigquery.SchemaField("confianca", "FLOAT64"),
    bigquery.SchemaField("motivo", "STRING"),
    bigquery.SchemaField("scored_at", "TIMESTAMP"),
]


def _table_ref(table_name: str) -> str:
    return f"{settings.gcp_project}.{settings.bq_dataset}.{table_name}"


def _ensure_news_table() -> None:
    client = _bq_client()
    table = bigquery.Table(_table_ref(NEWS_TABLE), schema=_NEWS_SCHEMA)
    client.create_table(table, exists_ok=True)
    progress_table = bigquery.Table(
        _table_ref(PROGRESS_TABLE),
        schema=[
            bigquery.SchemaField("year", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("month", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        ],
    )
    client.create_table(progress_table, exists_ok=True)


def _load_cache(ticker: str) -> dict:
    """All cached rows for one ticker, keyed by URL -- same shape as the
    earlier local-JSON cache so the scoring logic didn't need to change,
    just where it's persisted. Backed by BigQuery instead of a local file
    because Cloud Run Jobs are stateless: each execution is a fresh
    container with no disk carried over from the last run, so the cache
    has to live somewhere that survives between daily runs.
    """
    query = f"SELECT * FROM `{_table_ref(NEWS_TABLE)}` WHERE ticker = @ticker"
    job = _bq_client().query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        ),
    )
    cache = {}
    for row in job.result():
        entry = {"status": row.status}
        if row.status == "scored":
            entry.update(
                gdelt_date=row.gdelt_date,
                source=row.source,
                relevante=row.relevante,
                impacto=row.impacto,
                confianca=row.confianca,
                motivo=row.motivo,
                scored_at=row.scored_at.isoformat() if row.scored_at else None,
            )
        cache[row.url] = entry
    return cache


def _insert_new_rows(ticker: str, rows: list[dict]) -> None:
    if not rows:
        return
    payload = [
        {
            "url": url,
            "ticker": ticker,
            "status": entry["status"],
            "gdelt_date": entry.get("gdelt_date"),
            "source": entry.get("source"),
            "relevante": entry.get("relevante"),
            "impacto": entry.get("impacto"),
            "confianca": entry.get("confianca"),
            "motivo": entry.get("motivo"),
            "scored_at": entry.get("scored_at"),
        }
        for url, entry in rows
    ]
    errors = _bq_client().insert_rows_json(_table_ref(NEWS_TABLE), payload)
    if errors:
        raise RuntimeError(f"BigQuery insert errors for {ticker}: {errors}")


class DailyQuotaExhausted(Exception):
    """Raised when Gemini's free-tier *daily* request quota (not the
    per-minute one) is hit -- retrying with backoff cannot help here since
    it won't clear until the quota resets, unlike a per-minute 429 which
    clears in seconds. Found the hard way: an earlier run kept
    retrying-with-backoff on this for hours, burning wall-clock time
    without scoring a single additional article."""


def _score_article(ticker: str, text: str, api_key: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        company=get_company(ticker)["display_name"], ticker=ticker, article=text[:3000]
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(1, 4):
        try:
            response = requests.post(
                f"{GEMINI_URL}?key={api_key}", json=body, timeout=30
            )
            if response.status_code == 429:
                if "PerDay" in response.text:
                    raise DailyQuotaExhausted(response.text[:300])
                time.sleep(20 * attempt)
                continue
            response.raise_for_status()
            raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except DailyQuotaExhausted:
            raise
        except Exception:
            if attempt == 3:
                return None
            time.sleep(5)
    return None


def _score_candidates(ticker: str, candidates: list[dict], api_key: str, cache: dict) -> dict:
    """Extracts and scores each not-yet-cached candidate for one ticker,
    mutating `cache` in place and flushing new rows to BigQuery
    periodically. Resumable: already-cached URLs (success or permanent
    failure) are skipped."""
    to_score = [c for c in candidates if c["url"] not in cache]
    pending: list[tuple[str, dict]] = []

    def flush():
        nonlocal pending
        _insert_new_rows(ticker, pending)
        pending = []

    scored, extract_failed, score_failed = 0, 0, 0
    for i, candidate in enumerate(to_score):
        text = extract_article_text(candidate["url"])
        if not text or len(text) < 200:
            entry = {"status": "extract_failed"}
            cache[candidate["url"]] = entry
            pending.append((candidate["url"], entry))
            extract_failed += 1
            continue

        try:
            result = _score_article(ticker, text, api_key)
        except DailyQuotaExhausted:
            flush()
            raise
        time.sleep(GEMINI_MIN_INTERVAL_SECONDS)
        if result is None:
            entry = {"status": "score_failed"}
            cache[candidate["url"]] = entry
            pending.append((candidate["url"], entry))
            score_failed += 1
            continue

        entry = {
            "status": "scored",
            "gdelt_date": candidate["gdelt_date"],
            "source": candidate["source"],
            "relevante": result.get("relevante"),
            "impacto": result.get("impacto"),
            "confianca": result.get("confianca"),
            "motivo": result.get("motivo"),
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        cache[candidate["url"]] = entry
        pending.append((candidate["url"], entry))
        scored += 1

        if (i + 1) % 20 == 0:
            flush()

    flush()
    return {"newly_scored": scored, "extract_failed": extract_failed, "score_failed": score_failed}


def backfill_ticker(
    ticker: str, start_date: date, end_date: date, api_key: str, max_articles: int | None = None
) -> dict:
    """Single-ticker convenience entry point (used for quick/ad-hoc testing
    on one company). Issues its own BigQuery query -- for backfilling all
    12 tickers, use backfill_month() instead, which shares one query
    across every ticker for the same date range at ~1/12th the cost.
    """
    _ensure_news_table()
    cache = _load_cache(ticker)
    all_candidates = fetch_candidate_urls_all_tickers(start_date, end_date)
    candidates = [c for c in all_candidates if c["ticker"] == ticker]
    if max_articles is not None:
        already_cached = [c for c in candidates if c["url"] in cache]
        not_cached = [c for c in candidates if c["url"] not in cache][:max_articles]
        candidates = already_cached + not_cached

    result = _score_candidates(ticker, candidates, api_key, cache)
    return {"ticker": ticker, "candidates": len(candidates), "cache_size": len(cache), **result}


def backfill_month(
    year: int, month: int, api_key: str, max_per_ticker: int = 3
) -> dict:
    """Backfills one calendar month for all 12 tickers from a single shared
    BigQuery query, capping each ticker to `max_per_ticker` newly-scored
    articles that month -- spreads the rate-limited Gemini budget evenly
    across time instead of exhausting it on whichever week/company GDELT
    happens to cover most heavily (validation found 500+ Petrobras
    candidates in a single 2-week window alone).
    """
    _ensure_news_table()
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    all_candidates = fetch_candidate_urls_all_tickers(start, end)

    by_ticker: dict[str, list[dict]] = {t: [] for t in _news_companies()["investment_id"]}
    for c in all_candidates:
        by_ticker[c["ticker"]].append(c)

    summary = {}
    for ticker, candidates in by_ticker.items():
        cache = _load_cache(ticker)
        already_cached = [c for c in candidates if c["url"] in cache]
        not_cached = [c for c in candidates if c["url"] not in cache][:max_per_ticker]
        result = _score_candidates(ticker, already_cached + not_cached, api_key, cache)
        summary[ticker] = {"candidates": len(candidates), "cache_size": len(cache), **result}
    return summary


def backfill_history_range(
    api_key: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    max_per_ticker_per_month: int = 3,
) -> None:
    """Walks every month in [start_year-start_month, end_year-end_month]
    (inclusive), backfilling all 12 tickers each time. Safe to interrupt
    and rerun: already-scored URLs are skipped (see
    backfill_month/_score_candidates), so a partial run just picks up
    where it left off.

    Bounded by an explicit end month (rather than running to "today" in
    one call) so a multi-year backfill can be driven a few months at a
    time across many short, individually-supervised runs instead of one
    long-lived detached process -- each run comfortably fits inside a
    single tool-call timeout.
    """
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        summary = backfill_month(year, month, api_key, max_per_ticker_per_month)
        total_scored = sum(s["newly_scored"] for s in summary.values())
        total_cached = sum(s["cache_size"] for s in summary.values())
        print(f"{year}-{month:02d}: scored {total_scored} new articles (cache total: {total_cached})")
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year


def _load_progress(default_year: int, default_month: int) -> tuple[int, int]:
    query = f"SELECT year, month FROM `{_table_ref(PROGRESS_TABLE)}` ORDER BY updated_at DESC LIMIT 1"
    rows = list(_bq_client().query(query).result())
    if rows:
        return rows[0].year, rows[0].month
    return default_year, default_month


def _save_progress(year: int, month: int) -> None:
    # Append-only, not delete-then-insert: rows written via the streaming
    # insert API sit in a buffer for up to ~90 minutes during which
    # BigQuery flatly refuses any UPDATE/DELETE against the table ("would
    # affect rows in the streaming buffer") -- a real failure hit in
    # production, not a hypothetical. _load_progress() already reads the
    # single most recent row by updated_at, so the table doesn't need to
    # be kept to one row; it only grows by one tiny row per run (at most
    # daily), which is nowhere near worth periodic cleanup.
    _bq_client().insert_rows_json(
        _table_ref(PROGRESS_TABLE),
        [{"year": year, "month": month, "updated_at": datetime.now(timezone.utc).isoformat()}],
    )


def run_daily_backfill(
    api_key: str, start_year: int = 2015, max_per_ticker_per_month: int = 3
) -> str:
    """Entry point meant to be run once a day (e.g. via a Cloud Run Job on
    a Cloud Scheduler trigger): picks up from a persisted checkpoint (a
    BigQuery table, not local disk -- Cloud Run Jobs are stateless, a
    fresh container every execution) and processes months forward until
    either reaching the current month or hitting Gemini's free-tier
    *daily* request quota (500/day, discovered empirically -- the
    checkpoint means that's a graceful "see you tomorrow" rather than a
    lost run to retry-with-backoff against a quota that won't clear for
    hours).
    """
    _ensure_news_table()
    today = date.today()
    year, month = _load_progress(start_year, 1)

    while (year, month) <= (today.year, today.month):
        try:
            summary = backfill_month(year, month, api_key, max_per_ticker_per_month)
        except DailyQuotaExhausted:
            _save_progress(year, month)
            return f"quota exhausted at {year}-{month:02d}, resuming here tomorrow"

        total_scored = sum(s["newly_scored"] for s in summary.values())
        print(f"{year}-{month:02d}: scored {total_scored} new articles")
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year
        _save_progress(year, month)

    return "full history backfilled through current month"


def get_news_sentiment_history(ticker: str) -> pd.DataFrame:
    """Daily aggregate from the BigQuery cache: event_date,
    news_sentiment_score (mean of signed confidence over relevant articles
    that day; positive impact = +confianca, negative = -confianca, neutro
    = 0), n_articles.
    """
    query = f"""
    SELECT gdelt_date, impacto, confianca
    FROM `{_table_ref(NEWS_TABLE)}`
    WHERE ticker = @ticker AND status = 'scored' AND relevante = TRUE
    """
    job = _bq_client().query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        ),
    )
    rows = []
    for row in job.result():
        sign = {"positivo": 1, "negativo": -1, "neutro": 0}.get(row.impacto, 0)
        confidence = row.confianca or 0
        rows.append(
            {
                "event_date": pd.to_datetime(row.gdelt_date[:8], format="%Y%m%d"),
                "signed_score": sign * confidence,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["event_date", "news_sentiment_score", "n_articles"])

    df = pd.DataFrame(rows)
    daily = df.groupby("event_date").agg(
        news_sentiment_score=("signed_score", "mean"), n_articles=("signed_score", "count")
    ).reset_index()
    return daily.sort_values("event_date").reset_index(drop=True)
