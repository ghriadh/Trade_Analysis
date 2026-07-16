"""
Bronze pipeline entrypoint.

Extracts and saves the five static reference datasets to the bronze layer:
    1. Countries      — from mledoze/countries GitHub (REST)
    2. Trade flows    — hardcoded Comtrade flow codes (no API)
    3. Products       — from data/resources/products.csv
    4. Indicators     — from data/resources/indicators.csv
    5. Indicator values — from World Bank API (all indicators × all years)

Trade data (Comtrade) is intentionally excluded from this pipeline.
Given the volume of calls (~200 reporters × 10 years × 2 flows), it must
be executed in batches via the CLI scripts:
    python -m src.bronze.extract_trade_data          --year ... --product ... --flows ...
"""

import time
from src.config.logging_config import setup_logger
from src.db.connector import get_connector
from src.config.settings import ensure_directories, START_YEAR, END_YEAR

from src.bronze.extract_countries         import extract_countries, save_countries_to_bronze
from src.bronze.extract_trade_flow        import extract_trade_flow, save_flows_to_bronze
from src.bronze.extract_products          import extract_products, save_products_to_bronze
from src.bronze.extract_indicators        import extract_indicators, save_indicators_to_bronze
from src.bronze.extract_indicators_values import run as extract_indicators_values
from src.etl_status_writer import ETLStatusWriter 


def run_bronze_pipeline(db: str | None = None) -> None:
    """
    Run the full bronze pipeline for static reference data.
    Steps are intentionally sequential: indicators_values depends on
    the indicators CSV being present in bronze before it starts.
    """
    logger = setup_logger("src")
    ensure_directories()
    status = ETLStatusWriter()                                      
    status.start_pipeline("bronze") 

    # Initialise DB connector if requested — test connection before running transforms
    connector = None
    if db:
        connector = get_connector(backend=db)
        result    = connector.test_connection()
        if result["status"] == "error":
            status.end_pipeline("bronze", status="failed") 
            raise RuntimeError(
                f"DB connection failed | backend={result['backend']} | {result['detail']}\n"
                f"Check your .env configuration and database availability."
            )
        logger.info("DB connection OK | backend=%s", connector.backend)

    logger.info("=== START BRONZE PIPELINE ===")


    # --- Countries
    t = time.time()  
    df_countries = extract_countries()
    save_countries_to_bronze(df_countries, connector=connector)
    status.record_stage("bronze", "countries",                     
        rows=len(df_countries), duration_s=time.time() - t)

    # --- Trade flows
    t = time.time()  
    df_flows = extract_trade_flow()
    save_flows_to_bronze(df_flows, connector=connector)
    status.record_stage("bronze", "trade_flow",                     
        rows=len(df_flows), duration_s=time.time() - t)
    
    # --- Products
    t = time.time()  
    df_products = extract_products()
    save_products_to_bronze(df_products, connector=connector)
    status.record_stage("bronze", "products",                       
        rows=len(df_products), duration_s=time.time() - t)
    
    # --- Indicators reference
    t = time.time()
    df_indicators = extract_indicators()
    save_indicators_to_bronze(df_indicators, connector=connector)
    status.record_stage("bronze", "indicators",                       
        rows=len(df_indicators), duration_s=time.time() - t)

    # --- Indicator values (World Bank API — depends on indicators CSV above)
    t = time.time()
    extract_indicators_values(start_year=START_YEAR, end_year=END_YEAR, connector=connector)
    status.record_stage("bronze", "indicator_values",                       
        rows=0, duration_s=time.time() - t)

    status.end_pipeline("bronze")
    logger.info("=== END BRONZE PIPELINE ===")

# helper to read row count from CSV without loading full DF
def _count_rows_csv(layer: str, table: str) -> int:                 
    """Count rows in a layer CSV without loading into memory."""    
    from src.config.settings import DATA_DIR                        
    path = DATA_DIR / layer / table / f"{table}.csv"                
    if not path.exists():                                           
        return 0                                                    
    with open(path, encoding="utf-8") as f:                       
        return sum(1 for _ in f) - 1  # subtract header row         
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bronze pipeline")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        choices=["duckdb", "sqlserver", "databricks"],
        help=(
            "Optional. Save each DataFrame to the database in addition to CSV. "
            "Defaults to CSV only."
        ),
    )
    args = parser.parse_args()
    run_bronze_pipeline(db=args.db)