import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    SILVER_INDICATORS_PATH,
    SILVER_INDICATORS_VALUES_PATH,
    GOLD_INDICATORS_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver indicators reference and silver indicator values."""
    logger.info("Loading silver datasets")
    df_indicators = pd.read_csv(SILVER_INDICATORS_PATH)
    df_values     = pd.read_csv(SILVER_INDICATORS_VALUES_PATH)
    return df_indicators, df_values


# ============================================================
# VALIDATE
# ============================================================

def validate_indicator_consistency(
    df_indicators: pd.DataFrame,
    df_values: pd.DataFrame,
) -> None:
    """
    Ensure all indicator codes and names present in values
    exist in the indicators reference table.
    Raises ValueError on any mismatch.
    """
    missing_codes = (
        set(df_values["indicator_code"].dropna().unique())
        - set(df_indicators["indicator_code"].dropna().unique())
    )
    missing_names = (
        set(df_values["indicator_name"].dropna().unique())
        - set(df_indicators["indicator_name"].dropna().unique())
    )

    if missing_codes:
        raise ValueError(f"Indicator codes in values not found in reference: {missing_codes}")
    if missing_names:
        raise ValueError(f"Indicator names in values not found in reference: {missing_names}")

    logger.info("Indicator consistency check passed")


# ============================================================
# BUILD GOLD DIMENSION
# ============================================================

def build_gold_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the gold indicators dimension with a deterministic surrogate key.
    Sorted by indicator_code before key assignment for within-run consistency.
    """
    if df["indicator_code"].duplicated().any():
        raise ValueError("Duplicate indicator_code found in indicators reference")

    df = df.sort_values("indicator_code").reset_index(drop=True).copy()
    df["indicator_key"] = df.index + 1

    return df[[
        "indicator_key",
        "indicator_code",
        "indicator_name",
        "indicator_description",
        "indicator_unit",
    ]]


# ============================================================
# SAVE
# ============================================================

def save_indicators(df: pd.DataFrame, connector=None) -> None:
    GOLD_INDICATORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_INDICATORS_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators", schema="gold", connector=connector)
    logger.info("Gold indicators saved -> %s", GOLD_INDICATORS_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df_indicators, df_values = load_tables()
    validate_indicator_consistency(df_indicators, df_values)
    df_gold = build_gold_indicators(df_indicators)
    save_indicators(df_gold, connector=connector)


if __name__ == "__main__":
    run()
