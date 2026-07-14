"""Historical quarterly fundamentals from CVM's open data portal (Comissao
de Valores Mobiliarios -- Brazil's securities regulator). Public companies
are legally required to file standardized quarterly (ITR) and annual (DFP)
financial statements; CVM publishes them as free bulk CSV archives with no
auth, one zip per document type per year, in a standardized chart of
accounts that has been stable since ~2011.

Not part of investment_history: like macro_data.py and dividend_data.py,
this is a reference series fetched and cached at training time rather than
stored in BigQuery.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import requests

from src.config.company_dimension import get_company

CVM_ZIP_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc_type}/DADOS/{doc_type_lower}_cia_aberta_{year}.zip"

# CNPJ (CVM identifies companies by CNPJ, not by B3 ticker -- there is no
# public ticker-to-CNPJ lookup API) and share_class both now live in the
# company_dimension BigQuery table instead of a hardcoded dict here -- see
# src/config/company_dimension.py.

# CVM's standardized XBRL-based chart of accounts (and this bulk-CSV
# distribution) starts in 2010; 2011 is the first year with a clean full
# annual comparison.
FIRST_YEAR = 2011

# Fixed, regulation-mandated account codes -- stable across every company
# regardless of sector (a bank's DRE labels code 3.01 "Receitas de
# Intermediacao Financeira" instead of "Receita de Venda", but the code and
# its position in the statement are the same) -- except net income: banks
# and other financial institutions skip the "cost of goods sold" section
# industrials have, so their DRE is one step shorter and the final
# consolidated-result line lands on 3.09 instead of 3.11. Tried in order;
# whichever code actually has rows for a given company/year wins.
DRE_REVENUE_CODE = "3.01"
DRE_NET_INCOME_CODES = ("3.11", "3.09")
DRE_EPS_ON_CODE = "3.99.01.01"
DRE_EPS_PN_CODE = "3.99.01.02"
BPP_TOTAL_CODE = "2"
BPP_EQUITY_CODE = "2.03"


def _download_zip(doc_type: str, year: int) -> zipfile.ZipFile | None:
    url = CVM_ZIP_URL.format(doc_type=doc_type, doc_type_lower=doc_type.lower(), year=year)
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def _read_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    if name not in zf.namelist():
        return None
    with zf.open(name) as f:
        return pd.read_csv(f, sep=";", encoding="latin1", decimal=".")


def _extract_periods(df: pd.DataFrame | None, cnpj: str, code: str) -> dict[tuple, float]:
    """(period_start, period_end) -> VL_CONTA for one company/account code,
    keeping only the `ULTIMO` column (the filing's own period, not the
    prior-year comparison column also present in every row)."""
    if df is None:
        return {}
    sub = df[
        (df["CNPJ_CIA"] == cnpj) & (df["CD_CONTA"] == code) & (df["ORDEM_EXERC"] == "ÚLTIMO")
    ]
    if "DT_INI_EXERC" not in sub.columns:
        # Balance sheet statements (BPA/BPP) are point-in-time, not a period.
        return {
            pd.to_datetime(row.DT_FIM_EXERC).date(): row.VL_CONTA for row in sub.itertuples()
        }
    return {
        (pd.to_datetime(row.DT_INI_EXERC).date(), pd.to_datetime(row.DT_FIM_EXERC).date()): row.VL_CONTA
        for row in sub.itertuples()
    }


def _isolated_quarters(period_values: dict[tuple, float], year: int) -> dict[int, float]:
    """Derive Q1-Q4 isolated (non-cumulative) values for one calendar year
    from a period->value map that mixes isolated-quarter and
    year-to-date rows (ITR filings report both for the same period end;
    DFP only ever reports the full year).

    Q4 has no isolated row anywhere -- it's backed out as
    full-year (DFP) minus the Jan-Sep year-to-date (ITR Q3's second row).
    """
    q1 = period_values.get((date(year, 1, 1), date(year, 3, 31)))
    q2 = period_values.get((date(year, 4, 1), date(year, 6, 30)))
    q3 = period_values.get((date(year, 7, 1), date(year, 9, 30)))
    full_year = period_values.get((date(year, 1, 1), date(year, 12, 31)))
    ytd_9mo = period_values.get((date(year, 1, 1), date(year, 9, 30)))
    q4 = full_year - ytd_9mo if full_year is not None and ytd_9mo is not None else None
    return {1: q1, 2: q2, 3: q3, 4: q4}


def _receipt_dates(index_df: pd.DataFrame | None, cnpj: str) -> dict[date, date]:
    """DT_REFER (period end) -> DT_RECEB (actual public filing date) for one
    company, from the year's filing index. This is what makes the feature
    leak-free: the period a statement covers and the day it actually became
    public are different (typically 6-8 weeks later for ITR, longer for
    DFP), and only the latter is legitimate to join against price history.
    """
    if index_df is None:
        return {}
    sub = index_df[index_df["CNPJ_CIA"] == cnpj]
    return {
        pd.to_datetime(row.DT_REFER).date(): pd.to_datetime(row.DT_RECEB).date()
        for row in sub.itertuples()
    }


_year_data_cache: dict[tuple[str, int], dict] = {}


def _fetch_year(doc_type: str, year: int) -> dict:
    """Downloads and parses one year's archive once, cached for the life of
    the process -- every ticker's fetch reuses the same parsed frames
    instead of re-downloading the same ~30MB zip per company (all 12
    tracked tickers' filings live in the same yearly archive)."""
    key = (doc_type, year)
    if key in _year_data_cache:
        return _year_data_cache[key]

    zf = _download_zip(doc_type, year)
    if zf is None:
        parsed = {}
    else:
        prefix = f"{doc_type.lower()}_cia_aberta"
        parsed = {
            "index": _read_member(zf, f"{prefix}_{year}.csv"),
            "dre": _read_member(zf, f"{prefix}_DRE_con_{year}.csv"),
            "bpp": _read_member(zf, f"{prefix}_BPP_con_{year}.csv"),
        }
    _year_data_cache[key] = parsed
    return parsed


def fetch_fundamentals_history(ticker: str) -> pd.DataFrame:
    """Quarterly LTM (trailing-twelve-month) fundamentals for one ticker,
    indexed by the date each figure actually became public.

    Columns: event_date, revenue_ltm, net_income_ltm, eps_ltm, equity,
    total_liabilities.
    """
    try:
        company = get_company(ticker)
    except KeyError:
        company = {}
    cnpj = company.get("cnpj")
    if cnpj is None:
        return pd.DataFrame(
            columns=["event_date", "revenue_ltm", "net_income_ltm", "eps_ltm", "equity", "total_liabilities"]
        )

    eps_code = DRE_EPS_PN_CODE if company["share_class"] == "PN" else DRE_EPS_ON_CODE

    quarters: dict[tuple[int, int], dict] = {}
    receipt_dates: dict[date, date] = {}
    current_year = date.today().year

    for year in range(FIRST_YEAR, current_year + 1):
        itr = _fetch_year("ITR", year)
        dfp = _fetch_year("DFP", year)
        if not itr and not dfp:
            continue

        receipt_dates.update(_receipt_dates(itr.get("index"), cnpj))
        receipt_dates.update(_receipt_dates(dfp.get("index"), cnpj))

        for codes, key in (
            ((DRE_REVENUE_CODE,), "revenue"),
            (DRE_NET_INCOME_CODES, "net_income"),
            ((eps_code,), "eps"),
        ):
            periods: dict = {}
            for code in codes:
                periods.update(_extract_periods(itr.get("dre"), cnpj, code))
                periods.update(_extract_periods(dfp.get("dre"), cnpj, code))
                if periods:
                    break
            for q, value in _isolated_quarters(periods, year).items():
                if value is None:
                    continue
                quarters.setdefault((year, q), {})[key] = value

        for code, key in ((BPP_TOTAL_CODE, "total_liabilities_and_equity"), (BPP_EQUITY_CODE, "equity")):
            points = dict(_extract_periods(itr.get("bpp"), cnpj, code))
            points.update(_extract_periods(dfp.get("bpp"), cnpj, code))
            for period_end, value in points.items():
                q = (period_end.month - 1) // 3 + 1
                quarters.setdefault((period_end.year, q), {})[key] = value

    if not quarters:
        return pd.DataFrame(
            columns=["event_date", "revenue_ltm", "net_income_ltm", "eps_ltm", "equity", "total_liabilities"]
        )

    quarter_end_month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    rows = []
    for (year, q), values in sorted(quarters.items()):
        month, day = quarter_end_month_day[q]
        period_end = date(year, month, day)
        published = receipt_dates.get(period_end)
        rows.append(
            {
                "period_end": period_end,
                "published": published,
                "revenue": values.get("revenue"),
                "net_income": values.get("net_income"),
                "eps": values.get("eps"),
                "equity": values.get("equity"),
                "total_liabilities_and_equity": values.get("total_liabilities_and_equity"),
            }
        )
    quarterly = pd.DataFrame(rows).sort_values("period_end").reset_index(drop=True)

    # LTM = rolling sum of the last 4 isolated quarters for flow figures
    # (revenue, net income, EPS); balance-sheet figures (equity, total
    # liabilities) are point-in-time snapshots, used as-is.
    quarterly["revenue_ltm"] = quarterly["revenue"].rolling(4, min_periods=4).sum()
    quarterly["net_income_ltm"] = quarterly["net_income"].rolling(4, min_periods=4).sum()
    quarterly["eps_ltm"] = quarterly["eps"].rolling(4, min_periods=4).sum()
    quarterly["total_liabilities"] = quarterly["total_liabilities_and_equity"] - quarterly["equity"]

    # A statement only becomes usable as a feature on the day it was
    # actually disclosed; rows CVM hasn't logged a receipt date for yet
    # (very recent quarters) are dropped rather than guessed.
    quarterly = quarterly.dropna(subset=["published"])
    quarterly["event_date"] = pd.to_datetime(quarterly["published"])

    return quarterly[
        ["event_date", "revenue_ltm", "net_income_ltm", "eps_ltm", "equity", "total_liabilities"]
    ].dropna(subset=["revenue_ltm", "net_income_ltm"], how="all").reset_index(drop=True)


_cache: dict[str, pd.DataFrame] = {}


def get_fundamentals_history(ticker: str) -> pd.DataFrame:
    """Fetched once per ticker and cached for the life of the process."""
    if ticker not in _cache:
        _cache[ticker] = fetch_fundamentals_history(ticker)
    return _cache[ticker]
