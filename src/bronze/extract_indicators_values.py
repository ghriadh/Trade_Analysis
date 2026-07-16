"""
Bronze extraction — World Bank indicator values.

Extracts all indicator values for all countries and years from the
World Bank API. One indicator at a time, with per-indicator checkpoints
to allow resumable extraction across sessions.

Checkpoint mechanism:
    - Each successfully extracted indicator is saved to a temp CSV
      and marked with a .done file.
    - On re-run, already-done indicators are loaded from checkpoints
      and skipped — no redundant API calls.
    - To force a full re-extraction (e.g. new END_YEAR), delete:
          data/bronze/indicators_values/.checkpoints/

ETL status tracking:
    - After each indicator is extracted, etl_status.json is updated
      with per-indicator progress (done/failed counts + indicator list).
    - At the end of run(), the final consolidated row count is recorded.
"""

import logging
import time
from pathlib import Path

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.db.loader import save_to_db
from src.config.settings import (
    BRONZE_INDICATORS,
    BRONZE_INDICATORS_VALUES,
    BRONZE_INDICATORS_VALUES_PATH,
)

logger = logging.getLogger(__name__)

BASE_URL            = "https://api.worldbank.org/v2/country/all/indicator"
PER_PAGE            = 2000
REQUEST_TIMEOUT     = 60
SLEEP_BETWEEN_CALLS = 1

RETRY_TOTAL     = 3
RETRY_BACKOFF   = 2
RETRY_ON_STATUS = [500, 502, 503, 504]


# ============================================================
# HTTP SESSION WITH RETRY
# ============================================================

def _build_session() -> requests.Session:
    """
    Build a requests Session with automatic retry on transient errors.
    Retries on: connection errors, read timeouts, and 5xx responses.
    Backoff: 2s -> 4s -> 8s between attempts.
    """
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_ON_STATUS,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ============================================================
# CHECKPOINT
# ============================================================

def _checkpoint_dir() -> Path:
    return BRONZE_INDICATORS_VALUES / ".checkpoints"


def _checkpoint_csv(indicator: str) -> Path:
    # Replace dots in indicator code for safe filename (e.g. NY.GDP.MKTP.CD -> NY_GDP_MKTP_CD)
    safe = indicator.replace(".", "_")
    return _checkpoint_dir() / f"{safe}.csv"


def _checkpoint_marker(indicator: str) -> Path:
    safe = indicator.replace(".", "_")
    return _checkpoint_dir() / f"{safe}.done"


def is_done(indicator: str) -> bool:
    return _checkpoint_marker(indicator).exists()


def flatten_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested country and indicator dict columns returned by the WB API.
    Must be called before saving to checkpoint — pd.read_csv() reads dicts
    as strings, making lambda x: x["id"] fail on re-run.
    """
    df = df.copy()
    df["country_id"]     = df["country"].apply(lambda x: x["id"])
    df["country_name"]   = df["country"].apply(lambda x: x["value"])
    df["indicator_code"] = df["indicator"].apply(lambda x: x["id"])
    df["indicator_name"] = df["indicator"].apply(lambda x: x["value"])
    return df.drop(columns=["country", "indicator", "unit", "obs_status", "decimal"],
                   errors="ignore")


def mark_done(indicator: str, df: pd.DataFrame) -> None:
    """Flatten, save indicator data to temp CSV and write .done marker."""
    _checkpoint_dir().mkdir(parents=True, exist_ok=True)
    df.to_csv(_checkpoint_csv(indicator), index=False)
    _checkpoint_marker(indicator).touch()


def load_from_checkpoint(indicator: str) -> pd.DataFrame:
    """Load previously extracted indicator data from temp CSV."""
    return pd.read_csv(_checkpoint_csv(indicator))


# ============================================================
# API CALLS
# ============================================================

def fetch_indicator_page(
    session: requests.Session,
    indicator: str,
    start_year: int,
    end_year: int,
    page: int,
):
    """
    Fetch one page of World Bank indicator data.
    Returns (DataFrame, total_pages). Returns (empty DataFrame, 0) on empty response.
    """
    url = (
        f"{BASE_URL}/{indicator}"
        f"?format=json"
        f"&per_page={PER_PAGE}"
        f"&date={start_year}:{end_year}"
        f"&page={page}"
    )

    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()

    if not data or len(data) < 2 or data[1] is None:
        return pd.DataFrame(), 0

    pages = data[0].get("pages", 0)
    df    = pd.DataFrame(data[1])

    return df, pages


def extract_indicator_values(
    session: requests.Session,
    indicator: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch all pages for a single indicator code."""
    logger.info("Extracting indicator: %s", indicator)

    df_all, pages = fetch_indicator_page(session, indicator, start_year, end_year, page=1)

    for page in range(2, pages + 1):
        time.sleep(SLEEP_BETWEEN_CALLS)
        df_page, _ = fetch_indicator_page(session, indicator, start_year, end_year, page)
        df_all = pd.concat([df_all, df_page], ignore_index=True)

    logger.info("Indicator %s extracted (%d rows)", indicator, len(df_all))
    return df_all


# ============================================================
# SAVE
# ============================================================

def save_indicators_values_to_bronze(df: pd.DataFrame, connector=None) -> None:
    """Save consolidated indicators values to the bronze CSV."""
    BRONZE_INDICATORS_VALUES.mkdir(parents=True, exist_ok=True)
    df.to_csv(BRONZE_INDICATORS_VALUES_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators_values", schema="bronze", connector=connector)
    logger.info("Indicators values saved -> %s", BRONZE_INDICATORS_VALUES_PATH)


# ============================================================
# ETL STATUS HELPERS
# ============================================================

def _record_indicator_progress(
    indicators_done: list[str],
    indicators_failed: list[str],
    indicators_total: int,
) -> None:
    """
    Called after each indicator is processed (success or failure).
    Updates etl_status.json with current per-indicator progress.
    Status stays "partial" until run() completes and marks it "done".
    """
    from src.etl_status_writer import ETLStatusWriter

    writer = ETLStatusWriter()

    n_done   = len(indicators_done)
    n_failed = len(indicators_failed)
    progress = round((n_done + n_failed) / indicators_total * 100) if indicators_total else 0

    writer.record_stage(
        "bronze", "indicators_values",
        rows=0,              # rows unknown until consolidation
        duration_s=0.0,      # cumulative duration recorded at the end
        status="partial",
        checkpoints={
            "total":   indicators_total,
            "done":    n_done,
            "failed":  n_failed,
            "progress_pct": progress,
            "done_list":   indicators_done,
            "failed_list": indicators_failed,
        },
    )


def _record_final_status(
    rows: int,
    elapsed: float,
    indicators_done: list[str],
    indicators_failed: list[str],
    indicators_total: int,
) -> None:
    """
    Called at the end of run() after saving the consolidated CSV.
    Records final row count, duration, and full indicator breakdown.
    """
    from src.etl_status_writer import ETLStatusWriter

    writer = ETLStatusWriter()
    final_status = "done" if not indicators_failed else "done_with_errors"

    writer.record_stage(
        "bronze", "indicators_values",
        rows=rows,
        duration_s=elapsed,
        status=final_status,
        checkpoints={
            "total":       indicators_total,
            "done":        len(indicators_done),
            "failed":      len(indicators_failed),
            "progress_pct": 100,
            "done_list":   indicators_done,
            "failed_list": indicators_failed,
        },
    )
    logger.info(
        "ETL status | indicators_values recorded | rows=%d done=%d failed=%d",
        rows, len(indicators_done), len(indicators_failed),
    )

# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(start_year: int = 2015, end_year: int = 2024, connector=None) -> None:
    """
    Extract all indicator values from the World Bank API.

    Checkpoint mechanism — each indicator is saved individually on success.
    On re-run, already-extracted indicators are loaded from checkpoints
    and skipped — no redundant API calls.

    To force a full re-extraction (e.g. new END_YEAR), delete the
    checkpoints folder:
        data/bronze/indicators_values/.checkpoints/

    Saved to: data/bronze/indicators_values/indicators_values.csv
    """

    _start = time.time()
    logger.info("Starting indicators_values extraction (%d-%d)", start_year, end_year)

    indicators_path = BRONZE_INDICATORS / "indicators.csv"
    if not indicators_path.exists():
        raise FileNotFoundError(f"Bronze indicators file not found: {indicators_path}")

    indicators = pd.read_csv(indicators_path)["indicator_code"].dropna().unique()
    n_total    = len(indicators)
    logger.info("%d indicator codes to process", n_total)

    # Count already done
    already_done = [ind for ind in indicators if is_done(ind)]
    pending      = [ind for ind in indicators if not is_done(ind)]

    if already_done:
        logger.info(
            "Skipping %d already-extracted indicator(s): %s",
            len(already_done), already_done,
        )
    if pending:
        logger.info("%d indicator(s) to extract: %s", len(pending), pending)

    session  = _build_session()
    df_final = pd.DataFrame()
    indicators_done   = list(already_done)
    indicators_failed = [] 

    # Load already-done indicators from checkpoints
    for indicator in already_done:
        df_cached = load_from_checkpoint(indicator)
        df_final  = pd.concat([df_final, df_cached], ignore_index=True)

    # Extract pending indicators
    for indicator in pending:
        try:
            df_indicator = extract_indicator_values(session, indicator, start_year, end_year)
            df_indicator = flatten_indicator(df_indicator)
            mark_done(indicator, df_indicator)
            df_final = pd.concat([df_final, df_indicator], ignore_index=True)
            indicators_done.append(indicator)   
        except Exception:
            logger.exception("Failed to extract indicator %s — skipping", indicator)
            indicators_failed.append(indicator)

        # Update status after each indicator                        
        _record_indicator_progress(   
            indicators_done=indicators_done, 
            indicators_failed=indicators_failed,
            indicators_total=n_total, 
        )            

    if df_final.empty:
        logger.warning("No indicator values extracted — check API connectivity")
        return

    save_indicators_values_to_bronze(df_final, connector=connector)

    # Record final status                                           # + ADDED
    _record_final_status(                                           # + ADDED
        rows=len(df_final),                                         # + ADDED
        elapsed=round(time.time() - _start, 1),                    # + ADDED
        indicators_done=indicators_done,                            # + ADDED
        indicators_failed=indicators_failed,                        # + ADDED
        indicators_total=n_total,                                   # + ADDED
    ) 

    logger.info(
        "Indicators_values extraction complete | %d rows | %d/%d indicators OK",
        len(df_final), len(already_done) + len(pending) - 0, len(indicators),
    )


if __name__ == "__main__":
    run()
