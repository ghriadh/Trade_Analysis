import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_PRODUCTS_PATH, SILVER_PRODUCTS_PATH

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_products() -> pd.DataFrame:
    if not BRONZE_PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Bronze products file not found: {BRONZE_PRODUCTS_PATH}")

    logger.info("Loading bronze products")
    return pd.read_csv(BRONZE_PRODUCTS_PATH, dtype={"chapter": str})


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

    # Duplicate chapters
    duplicates = df[df["chapter"].duplicated(keep=False)]
    if not duplicates.empty:
        logger.warning("Duplicated chapter: %d rows", len(duplicates))

    # Invalid chapter range (HS chapters: 01–99)
    invalid_chapters = df[
        df["chapter"].str.isnumeric()
        & (~df["chapter"].str.lstrip("0").str.isnumeric().fillna(False)
           | (df["chapter"].astype(float) <= 0)
           | (df["chapter"].astype(float) >= 100))
    ]
    if not invalid_chapters.empty:
        logger.warning("Invalid chapter values: %d rows", len(invalid_chapters))

    # Unexpected leading/trailing spaces
    for col in ["description", "sector"]:
        unexpected_space = df[df[col] != df[col].astype(str).str.strip()]
        if not unexpected_space.empty:
            logger.warning("Unexpected spaces in %s: %d rows", col, len(unexpected_space))


# ============================================================
# TRANSFORM
# ============================================================

def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying silver transformations")

    # HS chapter codes are universal 3-character strings (e.g. "084", "027").
    # When read as integers they lose their leading zero — zero-pad to 3 chars.
    df["chapter"] = df["chapter"].astype(str).str.zfill(3)

    logger.info("Transform complete | %d products", len(df))
    return df


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_PRODUCTS_PATH, index=False)
    if connector:
        save_to_db(df, table="products", schema="silver", connector=connector)
    logger.info("Silver products saved -> %s", SILVER_PRODUCTS_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df = load_bronze_products()
    log_data_quality(df)
    df = transform_products(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
