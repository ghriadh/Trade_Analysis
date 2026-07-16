import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_INDICATORS_PATH, SILVER_INDICATORS_PATH

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_indicators() -> pd.DataFrame:
    if not BRONZE_INDICATORS_PATH.exists():
        raise FileNotFoundError(f"Bronze indicators file not found: {BRONZE_INDICATORS_PATH}")

    logger.info("Loading bronze indicators")
    return pd.read_csv(BRONZE_INDICATORS_PATH)


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

    # Duplicate indicator codes
    duplicates = df[df["indicator_code"].duplicated(keep=False)]
    if not duplicates.empty:
        logger.warning("Duplicated indicator_code: %d rows", len(duplicates))

    # Unexpected leading/trailing spaces
    for col in ["indicator_code", "indicator_name", "indicator_description", "indicator_unit"]:
        unexpected_space = df[df[col] != df[col].astype(str).str.strip()]
        if not unexpected_space.empty:
            logger.warning("Unexpected spaces in %s: %d rows", col, len(unexpected_space))


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_INDICATORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_INDICATORS_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators", schema="silver", connector=connector)
    logger.info("Silver indicators saved -> %s", SILVER_INDICATORS_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df = load_bronze_indicators()
    log_data_quality(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
