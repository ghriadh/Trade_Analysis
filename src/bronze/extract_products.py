import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_PRODUCTS_PATH, DATA_SOURCE

logger = logging.getLogger(__name__)


def extract_products() -> pd.DataFrame:
    """
    Load HS product reference data from the static source file.
    Bronze layer: direct copy from DATA_SOURCE with no transformation.
    """
    source_path = DATA_SOURCE / "products.csv"

    logger.info("Starting products extraction from %s", source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    logger.info("%d products extracted", len(df))

    return df


def save_products_to_bronze(df: pd.DataFrame, connector=None) -> None:
    """Save products DataFrame to the bronze layer."""
    df.to_csv(BRONZE_PRODUCTS_PATH, index=False)
    if connector:
        save_to_db(df, table="products", schema="bronze", connector=connector)
    logger.info("Products saved → %s", BRONZE_PRODUCTS_PATH)


def run(connector=None) -> None:
    """Orchestrate products extraction (extract + save)."""
    df = extract_products()
    save_products_to_bronze(df, connector=connector)


if __name__ == "__main__":
    run()
