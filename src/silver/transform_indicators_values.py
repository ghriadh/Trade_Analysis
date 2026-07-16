import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    BRONZE_INDICATORS_VALUES_PATH,
    SILVER_INDICATORS_VALUES,
    SILVER_INDICATORS_VALUES_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_indicators_values() -> pd.DataFrame:
    if not BRONZE_INDICATORS_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"Bronze indicators_values file not found: {BRONZE_INDICATORS_VALUES_PATH}"
        )

    logger.info("Loading bronze indicators_values")
    df = pd.read_csv(BRONZE_INDICATORS_VALUES_PATH)
    logger.info("Rows loaded: %d", len(df))
    return df


# ============================================================
# DATA QUALITY
# ============================================================

def log_data_quality(df: pd.DataFrame) -> None:
    logger.info("Rows: %d", len(df))

    # Null summary
    nulls = df.isna().sum()
    for col, count in nulls.items():
        if count > 0:
            logger.warning("Null values | %s: %d", col, count)

    # Year range validity
    invalid_years = df[(df["date"] < 1950) | (df["date"] > 2025)]
    if not invalid_years.empty:
        logger.warning("Invalid years detected: %d rows", len(invalid_years))

    # Zero indicator values (informational — not necessarily an error)
    zero_values = (df["value"] == 0).sum()
    if zero_values > 0:
        logger.info("Zero values in indicator values: %d rows", zero_values)

    # Unexpected spaces in string columns
    for col in ["countryiso3code", "country_id", "country_name", "indicator_code", "indicator_name"]:
        unexpected_space = df[df[col].notna() & (df[col] != df[col].astype(str).str.strip())]
        if not unexpected_space.empty:
            logger.warning("Unexpected spaces in %s: %d rows", col, len(unexpected_space))

    # Null country_id and countryiso3code — log affected countries for traceability
    null_country_id = df[df["country_id"].isna()]["country_name"].unique().tolist()
    null_iso3       = df[df["countryiso3code"].isna()]["country_name"].unique().tolist()

    if null_country_id:
        logger.info("Null country_id for: %s", null_country_id)
    if null_iso3:
        logger.info("Null countryiso3code for: %s", null_iso3)


# ============================================================
# TRANSFORM
# ============================================================

def transform_indicators_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying silver transformations")

    # Namibia — country_id is null in the World Bank API response because
    # "NA" is interpreted as NaN by pandas on read. Restore the correct id.
    df["country_id"] = df["country_id"].fillna("NA")

    # Non-country aggregates (regions, income groups, etc.) have no ISO3 code
    # in the World Bank API. Assign a synthetic code: "X" + country_id,
    # consistent with the pattern observed across other such entities.
    mask_iso_null = df["countryiso3code"].isna()
    if mask_iso_null.any():
        df.loc[mask_iso_null, "countryiso3code"] = (
            "X" + df.loc[mask_iso_null, "country_id"].astype(str)
        )

    # Type casting
    df["date"]  = df["date"].astype(int)
    df["value"] = df["value"].astype(float)

    logger.info("Transform complete | %d rows", len(df))
    return df


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_INDICATORS_VALUES.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_INDICATORS_VALUES_PATH, index=False)
    if connector:
        save_to_db(df, table="indicators_values", schema="silver", connector=connector)
    logger.info("Silver indicators_values saved -> %s", SILVER_INDICATORS_VALUES_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df = load_bronze_indicators_values()
    log_data_quality(df)
    df = transform_indicators_values(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
