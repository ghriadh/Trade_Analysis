import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    SILVER_INDICATORS_VALUES_PATH,
    GOLD_ENTITIES_PATH,
    GOLD_INDICATORS_PATH,
    GOLD_INDICATORS_VALUES_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gold reference tables and silver indicator values."""
    logger.info("Loading gold reference tables and silver indicators values")
    df_entities   = pd.read_csv(GOLD_ENTITIES_PATH)
    df_indicators = pd.read_csv(GOLD_INDICATORS_PATH)
    df_values     = pd.read_csv(SILVER_INDICATORS_VALUES_PATH)
    logger.info(
        "Loaded | entities=%d  indicators=%d  values=%d",
        len(df_entities), len(df_indicators), len(df_values),
    )
    return df_entities, df_indicators, df_values


# ============================================================
# VALIDATE REFERENTIAL INTEGRITY
# ============================================================

def validate_references(
    df_values: pd.DataFrame,
    df_entities: pd.DataFrame,
    df_indicators: pd.DataFrame,
) -> None:
    """
    Ensure all indicator codes and country ISO3 codes in the silver values
    table exist in their respective gold dimension tables.
    Raises ValueError on any mismatch.
    """
    logger.info("Validating referential integrity")

    missing_indicators = (
        set(df_values["indicator_code"].dropna().unique())
        - set(df_indicators["indicator_code"].unique())
    )
    if missing_indicators:
        raise ValueError(f"Missing indicator_code(s) in gold indicators: {missing_indicators}")

    missing_entities = (
        set(df_values["countryiso3code"].dropna().unique())
        - set(df_entities["wb_country_iso3"].unique())
    )
    if missing_entities:
        raise ValueError(f"Missing countryiso3code(s) in gold entities: {missing_entities}")

    logger.info("Referential integrity checks passed")


# ============================================================
# BUILD GOLD FACT TABLE
# ============================================================

def build_gold_indicators_values(
    df_values: pd.DataFrame,
    df_entities: pd.DataFrame,
    df_indicators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the gold indicators_values fact table by joining silver values
    with gold entity and indicator surrogate keys.
    """
    logger.info("Building gold indicators_values fact table")

    df_values = df_values.rename(columns={"date": "year"})

    df = (
        df_values
        .merge(
            df_entities[["wb_country_iso3", "entity_key"]],
            left_on="countryiso3code",
            right_on="wb_country_iso3",
            how="left",
        )
        .merge(
            df_indicators[["indicator_code", "indicator_key"]],
            on="indicator_code",
            how="left",
        )
    )

    # Post-join safety check — null surrogate keys indicate unresolved references
    if df[["entity_key", "indicator_key"]].isna().any().any():
        raise ValueError("Null surrogate keys detected after joins — check referential integrity")

    df = (
        df.sort_values(["indicator_key", "entity_key", "year"])
        .reset_index(drop=True)
        .copy()
    )
    df["indicator_value_key"] = df.index + 1

    df_gold = df[["indicator_value_key", "indicator_key", "entity_key", "year", "value"]]

    logger.info("Gold indicators_values built | %d rows", len(df_gold))
    return df_gold


# ============================================================
# SAVE
# ============================================================

def save_indicators_values(df: pd.DataFrame, connector=None) -> None:
    GOLD_INDICATORS_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_INDICATORS_VALUES_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators_values", schema="gold", connector=connector)
    logger.info("Gold indicators_values saved -> %s", GOLD_INDICATORS_VALUES_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df_entities, df_indicators, df_values = load_tables()
    validate_references(df_values, df_entities, df_indicators)
    df_gold = build_gold_indicators_values(df_values, df_entities, df_indicators)
    save_indicators_values(df_gold, connector=connector)


if __name__ == "__main__":
    run()
