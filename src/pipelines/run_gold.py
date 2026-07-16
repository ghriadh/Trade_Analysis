"""
Gold pipeline entrypoint.

Builds the six gold tables from silver inputs and gold dimension tables:

    Dimensions (no dependencies between them):
        1. countries      -> entities      (built from silver: countries, indicators_values, trade_data)
        2. trade_flow     -> trade_flow    (built from silver: trade_flow, trade_data)
        3. products       -> products      (built from silver: products, trade_data)
        4. indicators     -> indicators    (built from silver: indicators, indicators_values)

    Fact tables (depend on dimension tables):
        5. indicators_values -> fact table (depends on: entities, indicators)
        6. trade_data        -> fact table (depends on: entities, products, trade_flow)

Existing gold files are deleted before each run to guarantee a clean slate.

Correction flow (after detect_code_divergence in transform_countries):
---------------------------------------------------------------------
1. Re-extract missing bronze trade data:
       python -m src.bronze.extract_trade_data --year <y> --product <p> --flows <f> --reporters <codes>
2. Re-run silver trade data:
       python -m src.pipelines.run_silver --module trade_data
3. Re-run gold (full) -- entities are rebuilt with updated trade data,
   all surrogate keys are regenerated, fact tables are consistent:
       python -m src.pipelines.run_gold

   If you are certain no new partner entities were introduced by the
   correction (i.e. entities table is unchanged), you may run only:
       python -m src.pipelines.run_gold --module trade_data
   This requires gold entities, products and trade_flow to already exist.

Usage
-----
# Full run (first time or complete re-run)
python -m src.pipelines.run_gold

# Targeted re-run of a single module
python -m src.pipelines.run_gold --module trade_data
python -m src.pipelines.run_gold --module indicators_values
"""
import pandas as pd
import argparse
import logging
import time

from src.config.logging_config import setup_logger
from src.config.settings import (
    ensure_directories,
    GOLD_ENTITIES_PATH,
    GOLD_TRADE_FLOW_PATH,
    GOLD_PRODUCTS_PATH,
    GOLD_INDICATORS_PATH,
    GOLD_INDICATORS_VALUES_PATH,
    GOLD_TRADE_DATA_PATH,
)

from src.db.connector import get_connector
from src.gold.transform_countries         import run as run_countries
from src.gold.transform_trade_flow        import run as run_trade_flow
from src.gold.transform_products          import run as run_products
from src.gold.transform_indicators        import run as run_indicators
from src.gold.transform_indicators_values import run as run_indicators_values
from src.gold.transform_trade_data        import run as run_trade_data
from src.etl_status_writer import ETLStatusWriter

# ============================================================
# TERMINAL DISPLAY
# ============================================================

def _print_divergence_summary(divergent: "pd.DataFrame | None") -> None:
    """
    Print a visually distinct warning at the end of the gold pipeline
    when code divergence is detected between WB and Comtrade.
    Uses ANSI colors (supported on Windows 10+ terminals).
    """

    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    if divergent is None or (isinstance(divergent, pd.DataFrame) and divergent.empty):
        print(f"\n{BOLD}[GOLD] Code divergence check : OK -- no divergence detected{RESET}\n")
        return

    comtrade_codes = " ".join(
        divergent["country_code"].dropna().astype(int).astype(str).tolist()
    )

    sep = "=" * 64
    print(f"\n{BOLD}{YELLOW}{sep}{RESET}")
    print(f"{BOLD}{RED}  WARNING -- CODE DIVERGENCE DETECTED{RESET}")
    print(f"{BOLD}{YELLOW}{sep}{RESET}")
    print(f"{YELLOW}  {len(divergent)} country/ies have different WB and Comtrade codes:{RESET}\n")
    print(divergent.to_string(index=False))
    print(f"\n{BOLD}  Re-extraction required:{RESET}")
    print(f"  python -m src.bronze.extract_trade_data \\")
    print(f"      --year <year> --product <product> --flows <flows> \\")
    print(f"      --reporters {comtrade_codes}")
    print(f"\n{BOLD}  Then re-run:{RESET}")
    print(f"  python -m src.pipelines.run_silver --module trade_data")
    print(f"  python -m src.pipelines.run_gold")
    print(f"{BOLD}{YELLOW}{sep}{RESET}\n")


# ============================================================
# MODULE REGISTRY
# Order respects dependencies:
#   dimensions first, fact tables last.
#   indicators_values depends on entities + indicators.
#   trade_data depends on entities + products + trade_flow.
# ============================================================

MODULES = {
    "countries":         (run_countries,         GOLD_ENTITIES_PATH),
    "trade_flow":        (run_trade_flow,         GOLD_TRADE_FLOW_PATH),
    "products":          (run_products,           GOLD_PRODUCTS_PATH),
    "indicators":        (run_indicators,         GOLD_INDICATORS_PATH),
    "indicators_values": (run_indicators_values,  GOLD_INDICATORS_VALUES_PATH),
    "trade_data":        (run_trade_data,         GOLD_TRADE_DATA_PATH),
}

# Dependencies required by each fact table module for a targeted run.
# If any dependency file is missing, the targeted run is blocked.

MODULE_DEPENDENCIES: dict[str, list[tuple[str, object]]] = {
    "indicators_values": [
        ("countries",   GOLD_ENTITIES_PATH),
        ("indicators",  GOLD_INDICATORS_PATH),
    ],
    "trade_data": [
        ("countries",   GOLD_ENTITIES_PATH),
        ("products",    GOLD_PRODUCTS_PATH),
        ("trade_flow",  GOLD_TRADE_FLOW_PATH),
    ],
}

# ============================================================
# HELPERS
# ============================================================

def detect_existing_files(modules: dict) -> list[str]:
    """Return names of modules whose gold output file already exists."""
    return [name for name, (_, path) in modules.items() if path.exists()]


def delete_gold_files(modules: dict, logger: logging.Logger) -> None:
    """Delete gold output files for the given modules."""
    for name, (_, path) in modules.items():
        if path.exists():
            path.unlink()
            logger.info("Deleted existing gold file | %s", path.name)


def check_dependencies(module: str, logger: logging.Logger) -> None:
    """
    For a targeted module run, verify that all required dimension tables
    already exist. Raises RuntimeError if any dependency is missing.

    This guard ensures that running --module trade_data after a bronze
    correction does not produce a fact table with stale or missing keys.
    """
    deps = MODULE_DEPENDENCIES.get(module, [])
    missing = [(name, path) for name, path in deps if not path.exists()]
    if missing:
        missing_names = ", ".join(name for name, _ in missing)
        raise RuntimeError(
            f"Cannot run --module {module}: "
            f"missing gold dependency file(s): {missing_names}.\n"
            f"Run the full pipeline first: python -m src.pipelines.run_gold"
        )
    if deps:
        logger.info(
            "Dependency check passed for module '%s' | requires: %s",
            module,
            ", ".join(name for name, _ in deps),
        )


def _count_csv_rows(path) -> int:              
    """Count rows in a CSV without loading into memory."""  
    if not path.exists():
        return 0   
    with open(path, encoding="utf-8") as f: 
        return sum(1 for _ in f) - 1  # subtract header 


# ============================================================
# PIPELINE
# ============================================================

def run_gold_pipeline(module: str | None, db: str | None = None) -> None:
    logger = setup_logger("src")
    ensure_directories()
    status = ETLStatusWriter()
    status.start_pipeline("gold")
    # Initialise DB connector if requested — test connection before running transforms
    connector = None
    if db:
        connector = get_connector(backend=db)
        result    = connector.test_connection()
        if result["status"] == "error":
            status.end_pipeline("gold", status="failed")
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
        check_dependencies(module, logger)
    else:
        target = MODULES

    # Detect first run vs re-run
    existing = detect_existing_files(target)
    if existing:
        logger.info(
            "Re-run detected -- existing gold file(s) will be replaced: %s",
            ", ".join(existing),
        )
    else:
        logger.info("First run -- no existing gold files found")

    # Clean slate
    delete_gold_files(target, logger)

    # Execute transforms
    logger.info("=== START GOLD PIPELINE | modules: %s ===", ", ".join(target.keys()))

    divergent = None

    for name, (run_fn, output_path) in target.items():
        logger.info("--- %s ---", name)
        t = time.time() 
        
        try:       
            result = run_fn(connector=connector)
            if name == "countries":
                divergent = result
            status.record_stage( 
                "gold", name,
                rows=_count_csv_rows(output_path),
                duration_s=time.time() - t,
            )
        except Exception as e: 
            status.record_stage( 
                "gold", name, 
                rows=0, 
                duration_s=time.time() - t,
                status="failed",
                error=str(e),
            ) 
            status.end_pipeline("gold", status="failed") 
            raise 
    
    if name == "countries":
            divergent = result   # capture divergence report from transform_countries

    status.end_pipeline("gold")            
    logger.info("=== END GOLD PIPELINE ===")

    # Display divergence summary at the end — visible and distinct
    _print_divergence_summary(divergent)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Gold pipeline -- build gold dimension and fact tables.\n\n"
            "Examples:\n"
            "  Full run:\n"
            "    python -m src.pipelines.run_gold\n\n"
            "  Targeted re-run after bronze trade correction:\n"
            "    python -m src.pipelines.run_gold --module trade_data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        choices=list(MODULES.keys()),
        help=(
            "Optional. Run a single gold module. "
            "Fact table modules (trade_data, indicators_values) require "
            "their dimension dependencies to already exist. "
            "Defaults to running all modules in dependency order."
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
    run_gold_pipeline(module=args.module, db=args.db)
