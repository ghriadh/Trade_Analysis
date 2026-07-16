import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    SILVER_PRODUCTS_PATH,
    SILVER_TRADE_DATA_PATH,
    GOLD_PRODUCTS_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver products reference and silver trade data."""
    logger.info("Loading silver datasets")
    df_products = pd.read_csv(SILVER_PRODUCTS_PATH, dtype={"chapter": str})
    df_trade    = pd.read_csv(SILVER_TRADE_DATA_PATH)
    return df_products, df_trade


# ============================================================
# VALIDATE & ENRICH
# ============================================================

def validate_and_enrich_products(
    df_products: pd.DataFrame,
    df_trade: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all cmdCode values in trade data exist in the products reference.
    TOTAL is a valid aggregation code -- it is added to the reference if absent.
    Any other missing code raises a ValueError.
    """
    product_codes = set(df_products["chapter"].dropna().unique())
    trade_codes   = set(df_trade["cmdCode"].dropna().unique())
    missing_codes = trade_codes - product_codes

    if missing_codes == {"TOTAL"}:
        if "TOTAL" not in product_codes:
            logger.info("TOTAL not in products reference -- adding aggregation row")
            df_products = pd.concat(
                [
                    df_products,
                    pd.DataFrame([{
                        "chapter":             "TOTAL",
                        "product_description": "AGGREGATION ALL PRODUCTS",
                        "sector":              "AGGREGATION ALL SECTORS",
                    }]),
                ],
                ignore_index=True,
            )
    elif missing_codes:
        raise ValueError(f"Unexpected missing product codes in trade data: {missing_codes}")

    return df_products


# ============================================================
# BUILD GOLD DIMENSION
# ============================================================

def build_gold_products(df: pd.DataFrame) -> pd.DataFrame:
    """Build the gold products dimension with a surrogate key."""
    df = (
        df.rename(columns={"chapter": "cmdCode"})
        .sort_values("cmdCode")
        .reset_index(drop=True)
        .copy()
    )
    df["product_key"] = df.index + 1

    return df[["product_key", "cmdCode", "product_description", "sector"]]


# ============================================================
# SAVE
# ============================================================

def save_products(df: pd.DataFrame, connector=None) -> None:
    GOLD_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_PRODUCTS_PATH, index=False)
    if connector:
        save_to_db(df, table="products", schema="gold", connector=connector)
    logger.info("Gold products saved -> %s", GOLD_PRODUCTS_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df_products, df_trade = load_tables()
    df_products = validate_and_enrich_products(df_products, df_trade)
    df_gold     = build_gold_products(df_products)
    save_products(df_gold, connector=connector)


if __name__ == "__main__":
    run()
