"""
Bronze extraction — Comtrade trade data (CLI script).

This module is NOT part of the standard bronze pipeline (run_bronze.py).
Given the volume of calls (~200 reporters × n flows), extraction must be
executed in batches, one run per (year, product, flows) combination.

Typical usage
-------------
# Standard run — all reporters from bronze/countries.csv
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X

# Reporters with divergent Comtrade codes (not matching ISO country_code in bronze)
# Discovered after a standard run; pass the Comtrade-specific reporterCode values.
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X \\
    --reporters 251 842 757 699 579

# Retry reporters that failed in a previous run (from failed_reporters.json)
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X \\
    --reporters 134 276

# Override concurrency (default 4)
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X \\
    --workers 6

Output layout
-------------
data/bronze/trade_data/
    year=2022/product=TOTAL/flows=M_X/
        {reporter}_{flow}.csv        <- one file per (reporter, flow)
        failed_reporters.json        <- reporters that failed after all retries
        .checkpoints/
            {reporter}_{flow}.done   <- marker written after each successful save

Re-running a partial extraction (e.g. after a crash) skips all reporters
that already have a .done marker — only missing ones are fetched.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import comtradeapicall
import pandas as pd

from src.config.settings import BRONZE_COUNTRIES_PATH, BRONZE_TRADE_DATA

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

TYPE_CODE   = "C"
FREQ_CODE   = "A"
CL_CODE     = "HS"
Custom = "C00"  # None
MAX_RECORDS = 100_000


COLUMNS_TO_KEEP = [
    "typeCode",
    "reporterCode", "reporterISO", "reporterDesc",
    "flowCode",
    "partnerCode",  "partnerISO",  "partnerDesc",
    "cmdCode",
    "cifvalue", "fobvalue", "primaryValue",
]

# 2 retries on 429 with lightweight backoff — keeps the run moving
# without flooding the API after a rate-limit signal.
RETRY_DELAYS = [10, 20]   # seconds


# ============================================================
# REPORTERS
# ============================================================

def load_reporters(custom: list[int] | None) -> list[int]:
    """
    Return the list of reporter codes to extract.

    If `custom` is provided (--reporters CLI arg), use it directly.
    This covers two cases:
      - Countries whose Comtrade reporterCode differs from the ISO country_code
        stored in bronze/countries.csv (discovered post standard-run).
      - Retrying a specific subset from a previous failed_reporters.json.

    Otherwise load all country_code values from bronze/countries.csv.
    """
    if custom:
        logger.info("Using %d custom reporter codes", len(custom))
        return custom

    df = pd.read_csv(BRONZE_COUNTRIES_PATH)
    reporters = df["country_code"].dropna().astype(int).tolist()
    logger.info("Loaded %d reporter codes from bronze countries", len(reporters))
    return reporters


# ============================================================
# CHECKPOINT
# ============================================================

def _checkpoint_path(partition_dir: Path, reporter: int, flow: str) -> Path:
    return partition_dir / ".checkpoints" / f"{reporter}_{flow}.done"


def is_done(partition_dir: Path, reporter: int, flow: str) -> bool:
    return _checkpoint_path(partition_dir, reporter, flow).exists()


def mark_done(partition_dir: Path, reporter: int, flow: str) -> None:
    path = _checkpoint_path(partition_dir, reporter, flow)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# ============================================================
# SAVE
# ============================================================

def save_reporter(
    df: pd.DataFrame,
    partition_dir: Path,
    reporter: int,
    flow: str,
) -> None:
    """Save one (reporter, flow) result and write its checkpoint marker."""
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_file = partition_dir / f"{reporter}_{flow}.csv"
    df.to_csv(output_file, index=False)
    mark_done(partition_dir, reporter, flow)
    logger.info("Saved %d rows -> %s", len(df), output_file)


def save_failed(failed: list[dict], partition_dir: Path) -> None:
    """
    Persist the list of (reporter, flow) pairs that exhausted all retries.
    Re-run with --reporters using the reporter values from this file.
    """
    if not failed:
        return
    out = partition_dir / "failed_reporters.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2)
    logger.warning(
        "%d reporter(s) failed — saved to %s  |  re-run: --reporters %s",
        len(failed),
        out,
        " ".join(str(r["reporter"]) for r in failed),
    )


# ============================================================
# API CALL (sync — executed in thread)
# ============================================================

def fetch_comtrade(
    flow: str,
    year: int,
    reporter: int,
    product: str,
) -> tuple[pd.DataFrame | None, int | None]:
    """
    Single synchronous Comtrade API call.
    Returns (DataFrame, status) where status is 200, 429, or 403.

    No sleep inside this function — sleeping is handled async in run_task
    so the event loop is not blocked and other workers can continue.
    """
    try:
        df = comtradeapicall.previewFinalData (
            typeCode=TYPE_CODE,
            freqCode=FREQ_CODE,
            clCode=CL_CODE,
            period=str(year),
            reporterCode=str(reporter),
            cmdCode=product,
            flowCode=flow,
            partnerCode=None,
            partner2Code=0,
            customsCode=Custom,
            motCode=0,
            maxRecords=MAX_RECORDS,
            format_output="JSON",
            includeDesc=True,
        )
        if not isinstance(df, pd.DataFrame):
            return None, None
        return df, 200

    except Exception as e:
        msg = str(e)
        logger.warning(
            "API error | reporter=%d flow=%s year=%d product=%s | %s",
            reporter, flow, year, product, msg,
        )
        if "429" in msg or "Too Many Requests" in msg:
            return None, 429
        return None, 403


# ============================================================
# ASYNC TASK
# ============================================================

async def run_task(
    semaphore: asyncio.Semaphore,
    flow: str,
    year: int,
    reporter: int,
    product: str,
    partition_dir: Path,
) -> dict | None:
    """
    Fetch one (reporter, flow) pair with checkpoint skip and 429 backoff.

    Returns:
        None
            Skipped — already checkpointed (.done marker exists).
        {"reporter": int, "flow": str, "status": "with_data"}
            API call succeeded and returned data — saved to CSV.
        {"reporter": int, "flow": str, "status": "empty"}
            API call succeeded but returned no data — country has no trade
            data for this (year, product, flow). Marked as done so it is not
            retried on re-run. NOT recorded in failed_reporters.json.
        {"reporter": int, "flow": str, "status": "failed"}
            API call failed after all retries (429 exhausted, 403, exception).
            Recorded in failed_reporters.json for targeted re-run.
    """
    if is_done(partition_dir, reporter, flow):
        logger.debug("Skip (already done) | reporter=%d flow=%s", reporter, flow)
        return None

    async with semaphore:
        df, status = await asyncio.to_thread(fetch_comtrade, flow, year, reporter, product)

        # Async courtesy pause — does not block other workers
        await asyncio.sleep(1)

        # Backoff retries on 429
        for delay in RETRY_DELAYS:
            if status != 429:
                break
            logger.info(
                "429 rate-limit | reporter=%d flow=%s — retry in %ds",
                reporter, flow, delay,
            )
            await asyncio.sleep(delay)
            df, status = await asyncio.to_thread(fetch_comtrade, flow, year, reporter, product)
            await asyncio.sleep(1)

        # True API failure — exhausted all retries
        
     
        if df is None:
            logger.warning(
                "API failure | reporter=%d flow=%s status=%s",
                reporter, flow, status,
            )
            return {"reporter": reporter, "flow": flow, "status": "failed"}
        

        # Valid empty response — country has no trade data for this combination.
        # NOT marked as done — may be re-extracted in future runs if data becomes available.
        if df.empty:
            logger.info(
                "Empty response (no data) | reporter=%d flow=%s — not marking done",
                reporter, flow,
            )
            return {"reporter": reporter, "flow": flow, "status": "empty"}
        
        df = df[COLUMNS_TO_KEEP].copy()
        df["year"] = year

        save_reporter(df, partition_dir, reporter, flow)
        return {"reporter": reporter, "flow": flow, "status": "with_data"}


# ============================================================
# BATCH ORCHESTRATOR
# ============================================================

async def extract_batch(
    year: int,
    product: str,
    flows: list[str],
    reporters: list[int],
    workers: int,
) -> list[dict]:
    """
    Launch all (reporter, flow) tasks concurrently, bounded by `workers`.
    Tasks already checkpointed are silently skipped.
    Returns the list of failed entries for persistence in failed_reporters.json.
    """
    partition_dir = (
        BRONZE_TRADE_DATA
        / f"year={year}"
        / f"product={product}"
        / f"flows={'_'.join(flows)}"
    )

    # Identify pending (reporter, flow) pairs — skip already-done ones
    all_pairs     = [(flow, reporter) for flow in flows for reporter in reporters]
    pending_pairs = [(f, r) for f, r in all_pairs if not is_done(partition_dir, r, f)]
    skipped       = len(all_pairs) - len(pending_pairs)

    if skipped:
        logger.info("Skipping %d already-done (reporter, flow) pairs", skipped)

    if not pending_pairs:
        logger.info("All reporters already done — nothing to extract")
        return []

    semaphore = asyncio.Semaphore(workers)

    tasks = [
        run_task(semaphore, flow, year, reporter, product, partition_dir)
        for flow, reporter in pending_pairs
    ]

    logger.info(
        "Launching %d API calls | year=%d product=%s flows=%s workers=%d",
        len(tasks), year, product, flows, workers,
    )

    results = await asyncio.gather(*tasks)

    with_data_count = sum(1 for r in results if r and r["status"] == "with_data")
    empty_count     = sum(1 for r in results if r and r["status"] == "empty")
    failed_list     = [r for r in results if r and r["status"] == "failed"]

    logger.info(
        "Batch complete | with_data=%d  empty=%d  failed=%d  skipped=%d",
        with_data_count, empty_count, len(failed_list), skipped,
    )

    save_failed(failed_list, partition_dir)
    return results   # return all results for status recording


# ============================================================
# ENTRYPOINT
# ============================================================

def run(
    year: int,
    product: str,
    flows: list[str],
    custom_reporters: list[int] | None,
    workers: int,
) -> None:
    start = time.time()
    logger.info(
        "=== START trade extraction | year=%d  product=%s  flows=%s ===",
        year, product, flows,
    )

    reporters = load_reporters(custom_reporters)
    results   = asyncio.run(extract_batch(year, product, flows, reporters, workers))

    failed = [r for r in results if r and r["status"] == "failed"]
    empty  = [r for r in results if r and r["status"] == "empty"]

    elapsed = round(time.time() - start, 1)
    if failed:
        logger.warning(
            "=== END | duration=%ss | %d reporter(s) failed — see failed_reporters.json ===",
            elapsed, len(failed),
        )
    else:
        logger.info(
            "=== END | duration=%ss | all reporters OK | %d empty ===",
            elapsed, len(empty),
        )

    # Record this batch in etl_status.json
    _record_batch_status(
        year=year,
        product=product,
        flows=flows,
        reporters=reporters,
        results=results,
        elapsed=elapsed,
        custom_reporters=custom_reporters,
    )


# ============================================================
# CONSOLIDATION
# ============================================================

CONSOLIDATE_CHUNKSIZE = 5000   # optimised for ~1M rows with fast_executemany


def consolidate(connector=None) -> None:
    """
    Read all partitioned bronze trade_data CSV files and consolidate into:
        - One CSV file : data/bronze/trade_data/trade_data.csv
        - One DB table : bronze.trade_data  (if connector provided)

    Run this once after all extraction batches are complete.
    Skips .checkpoints directories and failed_reporters.json files.

    Usage:
        python -m src.bronze.extract_trade_data --consolidate
        python -m src.bronze.extract_trade_data --consolidate --db sqlserver
    """
    _start = time.time()                                            

    files = [
        f for f in BRONZE_TRADE_DATA.rglob("*.csv")
        if ".checkpoints" not in f.parts
        and f.name != "failed_reporters.json"
        and f.name != "trade_data.csv"   
    ]

    if not files:
        logger.warning("No partitioned trade_data files found in %s", BRONZE_TRADE_DATA)
        return

    logger.info("Consolidating %d trade_data file(s)", len(files))

    df = pd.concat(
        [pd.read_csv(f, quotechar='"', encoding="cp1252") for f in files],
        ignore_index=True,
    )

    logger.info("Total rows before deduplication: %d", len(df))

    # Remove duplicates that may arise from overlapping extraction runs
    natural_key = ["reporterCode", "partnerCode", "year", "flowCode", "cmdCode"]
    df = df.drop_duplicates(subset=natural_key, keep="first")
    logger.info("Total rows after deduplication: %d", len(df))

    # Save consolidated CSV at the root of BRONZE_TRADE_DATA
    output_path = BRONZE_TRADE_DATA / "trade_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("Consolidated CSV saved -> %s", output_path)

    # Save to DB if connector provided — chunksize optimised for 1M+ rows
    if connector:
        from src.db.loader import save_to_db
        save_to_db(
            df,
            table="trade_data",
            schema="bronze",
            connector=connector,
            chunksize=CONSOLIDATE_CHUNKSIZE,
        )

    logger.info("Consolidation complete | %d rows", len(df))

    # Record final consolidated result in etl_status.json
    _record_consolidation_status(
        rows=len(df),
        elapsed=round(time.time() - _start, 1),
    )


# ============================================================
# ETL STATUS HELPERS
# ============================================================

def _record_batch_status(
    year: int,
    product: str,
    flows: list[str],
    reporters: list[int],
    results: list[dict | None],
    elapsed: float,
    custom_reporters: list[int] | None = None,
) -> None:
    """
    Called at the end of each CLI batch run.

    Each run gets its own timestamped entry in the checkpoints dict —
    so multiple runs on the same partition (standard + corrections) are
    all preserved and visible in Streamlit as separate rows.

    Key format:  "2022/TOTAL/M_X#2026-05-20 08:31 UTC"
    Counts:
        total      — reporters attempted in this run
        with_data  — API returned data (saved to CSV)
        empty      — API returned no data (valid, marked done, not retried)
        failed     — API error after all retries (in failed_reporters.json)
        type       — "standard" or "correction"
    """
    from src.etl_status_writer import ETLStatusWriter
    from datetime import datetime, timezone
 
    # Timestamp-based key — each run gets its own entry
    timestamp     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    partition_key = f"{year}/{product}/{'_'.join(flows)}#{timestamp}"

    n_total     = len(reporters)
    with_data_n = sum(1 for r in results if r and r["status"] == "with_data")
    empty_n     = sum(1 for r in results if r and r["status"] == "empty")
    failed_n    = sum(1 for r in results if r and r["status"] == "failed")
    run_type    = "correction" if custom_reporters else "standard"

    writer = ETLStatusWriter()

    # Read existing checkpoints — preserve all previous runs
    existing_checkpoints = (
        writer._data
        .get("pipelines", {})
        .get("bronze", {})
        .get("stages", {})
        .get("trade_data", {})
        .get("checkpoints", {})
    )

    # Add this run as a new timestamped entry
    existing_checkpoints[partition_key] = {
        "total":     n_total,
        "with_data": with_data_n,
        "empty":     empty_n,
        "failed":    failed_n,
        "type":      run_type,
    }

    batch_status = "partial" if failed_n == 0 else "partial_errors"

    writer.record_stage(
        "bronze", "trade_data",
        rows=0,            # rows unknown until consolidate()
        duration_s=elapsed,
        status=batch_status,
        checkpoints=existing_checkpoints,
    )
    logger.info(
        "ETL status | run=%s type=%s with_data=%d empty=%d failed=%d",
        partition_key, run_type, with_data_n, empty_n, failed_n,
    )


def _record_consolidation_status(rows: int, elapsed: float) -> None:
    """
    Called at the end of consolidate().
    Overwrites the trade_data stage with final row count
    and marks it "done". Preserves existing checkpoints.
    """
    from src.etl_status_writer import ETLStatusWriter

    writer = ETLStatusWriter()

    # Preserve checkpoints accumulated by _record_batch_status
    existing_checkpoints = (
        writer._data
        .get("pipelines", {})
        .get("bronze", {})
        .get("stages", {})
        .get("trade_data", {})
        .get("checkpoints", {})
    )

    writer.record_stage(
        "bronze", "trade_data",
        rows=rows,
        duration_s=elapsed,
        status="done",
        checkpoints=existing_checkpoints,
    )
    logger.info(
        "ETL status | trade_data consolidation recorded | %d rows", rows
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Bronze trade data extractor — Comtrade API.\n"
            "One batch at a time: one year, one product, one or more flows.\n\n"
            "Examples:\n"
            "  Extraction batch:\n"
            "    python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X\n\n"
            "  Reporters with divergent Comtrade codes:\n"
            "    python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X --reporters 251 842\n\n"
            "  Consolidate all batches into one CSV + DB (run after all batches are done):\n"
            "    python -m src.bronze.extract_trade_data --consolidate\n"
            "    python -m src.bronze.extract_trade_data --consolidate --db sqlserver"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--consolidate", action="store_true", default=False,
        help=(
            "Consolidate all partitioned extraction files into one CSV "
            "and optionally save to DB (requires --db). "
            "Run this once after all extraction batches are complete."
        ),
    )
    parser.add_argument(
        "--db",
        type=str, default=None, choices=["duckdb", "sqlserver", "databricks"],
        help="Optional. Save consolidated DataFrame to DB. Only used with --consolidate.",
    )
    parser.add_argument(
        "--year",      type=int,       default=None,
        help="Reporting year (e.g. 2022). Required for extraction, ignored for --consolidate.",
    )
    parser.add_argument(
        "--product",   type=str,       default=None,
        help="HS product code or TOTAL. Required for extraction, ignored for --consolidate.",
    )
    parser.add_argument(
        "--flows",     type=str,       nargs="+", default=None,
        help="Flow codes: M (import), X (export), RM, RX. Required for extraction.",
    )
    parser.add_argument(
        "--reporters", type=int,       nargs="+", default=None,
        help=(
            "Optional. Comtrade reporter codes to extract. "
            "Use for countries with divergent codes or to retry from failed_reporters.json."
        ),
    )
    parser.add_argument(
        "--workers",   type=int,       default=4,
        help="Max concurrent API calls (default: 4).",
    )

    args = parser.parse_args()

    if args.consolidate:
        from src.db.connector import get_connector
        connector = get_connector(backend=args.db) if args.db else None
        consolidate(connector=connector)
    else:
        if not all([args.year, args.product, args.flows]):
            parser.error("--year, --product and --flows are required for extraction mode.")
        run(
            year=args.year,
            product=args.product,
            flows=args.flows,
            custom_reporters=args.reporters,
            workers=args.workers,
        )
