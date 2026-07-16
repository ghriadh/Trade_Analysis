import logging

import pandas as pd

from src.config.settings import BRONZE_COUNTRIES_PATH, SILVER_COUNTRIES_PATH
from src.db.loader import save_to_db

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_countries() -> pd.DataFrame:
    if not BRONZE_COUNTRIES_PATH.exists():
        raise FileNotFoundError(f"Bronze countries file not found: {BRONZE_COUNTRIES_PATH}")

    logger.info("Loading bronze countries")
    return pd.read_csv(BRONZE_COUNTRIES_PATH)


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

    # Duplicate country codes
    duplicates = df[df["country_code"].duplicated(keep=False)]
    if not duplicates.empty:
        logger.warning("Duplicated country_code: %d rows", len(duplicates))

    # Invalid country codes (expected range 1–999)
    invalid_codes = df[
        df["country_code"].notna()
        & ((df["country_code"] <= 0) | (df["country_code"] >= 1000))
    ]
    if not invalid_codes.empty:
        logger.warning("Invalid country_code values: %d rows", len(invalid_codes))

    # Unexpected leading/trailing spaces in string identifiers
    for col in ["countryiso3code", "country_name"]:
        unexpected_space = df[df[col] != df[col].astype(str).str.strip()]
        if not unexpected_space.empty:
            logger.warning("Unexpected spaces in %s: %d rows", col, len(unexpected_space))


# ============================================================
# TRANSFORM
# ============================================================

def transform_countries(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying silver transformations")

    # Kosovo (XKX) has no numeric country_code in the source — assign 990.
    # Verified: Kosovo is the only country with a null country_code.
    df["country_code"] = df["country_code"].fillna(990).astype(int)
    df.loc[df["country_code"] == 990, "countryiso3code"] = "XKX"

    # Population is dropped: all values in the source are 0 and the column
    # is not used in any downstream analysis (covered by indicators_values).
    if "population" in df.columns:
        df = df.drop(columns=["population"])

    # Standardize country names with special characters
    NAME_CORRECTIONS = {
    "Côte d'Ivoire":  "Cote d'Ivoire",
    "CÃƒÂ´te d'Ivoire": "Cote d'Ivoire",
    "Curaçao":        "Curacao",
    "CuraÃƒÂ§ao":      "Curacao",
    "Türkiye":        "Turkiye",
    "TÃ¼rkiye":       "Turkiye",  
    "TÃƒÂ¼rkiye" :    "Turkiye",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    }
    df["country_name"] = df["country_name"].replace(NAME_CORRECTIONS)

    logger.info("Transform complete | %d countries", len(df))

    return df


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_COUNTRIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_COUNTRIES_PATH, index=False)
    logger.info("Silver countries saved -> %s", SILVER_COUNTRIES_PATH)
    if connector:
        save_to_db(df, table="countries", schema="silver", connector=connector)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df = load_bronze_countries()
    log_data_quality(df)
    df = transform_countries(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
