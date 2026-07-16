import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_TRADE_FLOW_PATH, SILVER_TRADE_FLOW_PATH

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_trade_flow() -> pd.DataFrame:
    if not BRONZE_TRADE_FLOW_PATH.exists():
        raise FileNotFoundError(f"Bronze trade flow file not found: {BRONZE_TRADE_FLOW_PATH}")

    logger.info("Loading bronze trade flow")
    return pd.read_csv(BRONZE_TRADE_FLOW_PATH)


# ============================================================
# DATA QUALITY
# ============================================================

def log_data_quality(df: pd.DataFrame) -> None:
    logger.info("Rows: %d", len(df))

    # Null values
    nulls = df.isna().sum()
    for col, count in nulls.items():
        if count > 0:
            logger.warning("Null values | %s: %d", col, count)

    # Duplicate flow codes
    duplicates = df[df["flow_code"].duplicated(keep=False)]
    if not duplicates.empty:
        logger.warning("Duplicated flow_code: %d rows", len(duplicates))

    # Unexpected leading/trailing spaces
    unexpected_space = df[
        df["flow_description"] != df["flow_description"].astype(str).str.strip()
    ]
    if not unexpected_space.empty:
        logger.warning("Unexpected spaces in flow_description: %d rows", len(unexpected_space))


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_TRADE_FLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_TRADE_FLOW_PATH, index=False)
    if connector:
        save_to_db(df, table="trade_flow", schema="silver", connector=connector)
    logger.info("Silver trade flow saved -> %s", SILVER_TRADE_FLOW_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df = load_bronze_trade_flow()
    log_data_quality(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
