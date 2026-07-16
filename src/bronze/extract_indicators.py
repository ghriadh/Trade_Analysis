import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_INDICATORS_PATH, DATA_SOURCE

logger = logging.getLogger(__name__)


def extract_indicators() -> pd.DataFrame:
    """
    Load indicators reference data from the static source file.
    Bronze layer: direct copy from DATA_SOURCE with no transformation.
    """
    source_path = DATA_SOURCE / "indicators.csv"

    logger.info("Starting indicators extraction from %s", source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    logger.info("%d indicators extracted", len(df))

    return df


def save_indicators_to_bronze(df: pd.DataFrame, connector=None) -> None:
    """Save indicators DataFrame to the bronze layer."""
    df.to_csv(BRONZE_INDICATORS_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators", schema="bronze", connector=connector)
    logger.info("Indicators saved → %s", BRONZE_INDICATORS_PATH)


def run(connector=None) -> None:
    """Orchestrate indicators extraction (extract + save)."""
    df = extract_indicators()
    save_indicators_to_bronze(df, connector=connector)


if __name__ == "__main__":
    run()
