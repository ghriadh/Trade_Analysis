"""
Silver pipeline entrypoint.

Transforms and saves the six silver datasets from their bronze sources:
    1. Countries
    2. Trade flows
    3. Products
    4. Indicators
    5. Indicator values
    6. Trade data

Existing silver files are deleted before each run to guarantee a clean slate.
The pipeline logs whether it is a first run or a re-run based on file presence.

Usage
-----
# Full run (first time or complete re-run)
python -m src.pipelines.run_silver

# Targeted re-run after a bronze trade correction extraction
python -m src.pipelines.run_silver --module trade_data
"""

import argparse
import logging
import time

from src.config.logging_config import setup_logger
from src.config.settings import (
    ensure_directories,
    SILVER_COUNTRIES_PATH,
    SILVER_TRADE_FLOW_PATH,
    SILVER_PRODUCTS_PATH,
    SILVER_INDICATORS_PATH,
    SILVER_INDICATORS_VALUES_PATH,
    SILVER_TRADE_DATA_PATH,
)
from src.db.connector import get_connector

from src.silver.transform_countries          import run as run_countries
from src.silver.transform_trade_flow         import run as run_trade_flow
from src.silver.transform_products           import run as run_products
from src.silver.transform_indicators         import run as run_indicators
from src.silver.transform_indicators_values  import run as run_indicators_values
from src.silver.transform_trade_data         import run as run_trade_data
from src.etl_status_writer import ETLStatusWriter


# ============================================================
# MODULE REGISTRY
# Ordered: no inter-module dependencies in silver, but kept
# consistent with the bronze pipeline sequence for readability.
# ============================================================

MODULES = {
    "countries":         (run_countries,         SILVER_COUNTRIES_PATH),
    "trade_flow":        (run_trade_flow,         SILVER_TRADE_FLOW_PATH),
    "products":          (run_products,           SILVER_PRODUCTS_PATH),
    "indicators":        (run_indicators,         SILVER_INDICATORS_PATH),
    "indicators_values": (run_indicators_values,  SILVER_INDICATORS_VALUES_PATH),
    "trade_data":        (run_trade_data,         SILVER_TRADE_DATA_PATH),
}

# ============================================================
# HELPERS
# ============================================================

def detect_existing_files(modules: dict) -> list[str]:
    """Return names of modules whose silver output file already exists."""
    return [name for name, (_, path) in modules.items() if path.exists()]


def delete_silver_files(modules: dict, logger: logging.Logger) -> None:
    """Delete silver output files for the given modules."""
    for name, (_, path) in modules.items():
        if path.exists():
            path.unlink()
            logger.info("Deleted existing silver file | %s", path.name)



def _count_csv_rows(path) -> int:  
    """Count rows in a CSV without loading into memory.""" 
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f: 
        return sum(1 for _ in f) - 1  # subtract header


# ============================================================
# PIPELINE
# ============================================================

def run_silver_pipeline(module: str | None, db: str | None = None) -> None:
    logger = setup_logger("src")
    ensure_directories()
    status = ETLStatusWriter()
    status.start_pipeline("silver")

    # Initialise DB connector if requested -- test connection before running transforms
    connector = None
    if db:
        connector = get_connector(backend=db)
        result    = connector.test_connection()
        if result["status"] == "error":
            status.end_pipeline("silver", status="failed") 
            raise RuntimeError(
                f"DB connection failed | backend={result['backend']} | {result['detail']}\n"
                f"Check your .env configuration and database availability."
            )
        logger.info("DB connection OK | backend=%s", connector.backend)

    # Resolve target modules
    if module:
        if module not in MODULES:
            raise ValueError(
                f"Unknown module '{module}'. "
                f"Available: {', '.join(MODULES.keys())}"
            )
        target = {module: MODULES[module]}
    else:
        target = MODULES

    # Detect first run vs re-run
    existing = detect_existing_files(target)
    if existing:
        logger.info(
            "Re-run detected -- existing silver file(s) will be replaced: %s",
            ", ".join(existing),
        )
    else:
        logger.info("First run -- no existing silver files found")


    # Clean slate
    delete_silver_files(target, logger)

    # Execute transforms
    logger.info("=== START SILVER PIPELINE | modules: %s ===", ", ".join(target.keys()))

    for name, (run_fn, output_path) in target.items():
        logger.info("--- %s ---", name)
        t = time.time()
        try:  
            run_fn(connector=connector)
            status.record_stage( 
                "silver", name, 
                rows=_count_csv_rows(output_path), 
                duration_s=time.time() - t, 
            )
        except Exception as e: 
            status.record_stage(
                "silver", name,
                rows=0, 
                duration_s=time.time() - t,
                status="failed",
                error=str(e),
            )
            status.end_pipeline("silver", status="failed")
            raise
 
    status.end_pipeline("silver")

    logger.info("=== END SILVER PIPELINE ===")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Silver pipeline -- transform bronze data to silver layer.\n\n"
            "Examples:\n"
            "  Full run:\n"
            "    python -m src.pipelines.run_silver\n\n"
            "  Targeted re-run after bronze trade correction:\n"
            "    python -m src.pipelines.run_silver --module trade_data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        choices=list(MODULES.keys()),
        help=(
            "Optional. Run a single silver transform. "
            "Use 'trade_data' after a bronze correction extraction. "
            "Defaults to running all modules."
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        choices=["duckdb", "sqlserver", "databricks"],
        help=(
            "Optional. Save each DataFrame to the database in addition to CSV. "
            "Backend credentials must be set in .env. "
            "Defaults to CSV only."
        ),
    )

    args = parser.parse_args()
    run_silver_pipeline(module=args.module, db=args.db)
