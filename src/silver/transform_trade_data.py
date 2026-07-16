import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_TRADE_DATA, SILVER_TRADE_DATA_PATH

logger = logging.getLogger(__name__)

NATURAL_KEY = ["reporterCode", "partnerCode", "year", "flowCode", "cmdCode"]

YEAR_MIN = 1950
YEAR_MAX = 2026


# ============================================================
# LOAD BRONZE
# ============================================================

def load_bronze_trade_data() -> pd.DataFrame:
    """
    Load bronze trade data into a single DataFrame.

    Priority:
      1. Consolidated file (BRONZE_TRADE_DATA/trade_data.csv) — created by
         extract_trade_data.py --consolidate. Preferred: faster, no duplicates.
      2. Partitioned files via rglob — fallback when consolidation has not
         been run yet (extraction still in progress).

    Files are encoded in cp1252 (Comtrade API output on Windows).
    """
    consolidated = BRONZE_TRADE_DATA / "trade_data.csv"

    if consolidated.exists():
        logger.info("Loading from consolidated bronze file -> %s", consolidated)
        df = pd.read_csv(consolidated, quotechar='"', encoding="cp1252")
        logger.info("Rows loaded: %d", len(df))
        return df

    # Fallback: read all partitioned reporter files
    files = [
        f for f in BRONZE_TRADE_DATA.rglob("*.csv")
        if ".checkpoints" not in f.parts
        and f.name != "failed_reporters.json"
    ]

    if not files:
        raise FileNotFoundError(
            f"No bronze trade_data files found in {BRONZE_TRADE_DATA}. "
            f"Run extraction batches first, then consolidate with: "
            f"python -m src.bronze.extract_trade_data --consolidate"
        )

    logger.info(
        "Loading from %d partitioned file(s) — consider running --consolidate first",
        len(files),
    )
    df = pd.concat(
        [pd.read_csv(f, quotechar='"', encoding="cp1252") for f in files],
        ignore_index=True,
    )
    logger.info("Rows loaded: %d", len(df))
    return df


# ============================================================
# DATA QUALITY
# ============================================================

def log_data_quality(df: pd.DataFrame) -> None:
    logger.info("Rows: %d", len(df))

    # --- Nulls on key columns
    for col in ["reporterCode", "partnerCode", "year", "flowCode", "cmdCode", "primaryValue"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning("Null values | %s: %d", col, null_count)

    # --- primaryValue validity
    negative_values = (df["primaryValue"] < 0).sum()
    zero_values     = (df["primaryValue"] == 0).sum()
    if negative_values > 0:
        logger.warning("Negative primaryValue: %d rows", negative_values)
    if zero_values > 0:
        logger.info("Zero primaryValue: %d rows", zero_values)

    # --- Year range
    invalid_years = df[(df["year"] < YEAR_MIN) | (df["year"] > YEAR_MAX)]
    if not invalid_years.empty:
        logger.warning(
            "Year out of range [%d–%d]: %d rows", YEAR_MIN, YEAR_MAX, len(invalid_years)
        )

    # --- Unexpected spaces in string columns
    for col in ["cmdCode", "flowCode", "reporterISO", "reporterDesc", "partnerDesc"]:
        unexpected_space = df[df[col].notna() & (df[col] != df[col].astype(str).str.strip())]
        if not unexpected_space.empty:
            logger.warning("Unexpected spaces in %s: %d rows", col, len(unexpected_space))

    # partnerISO is intentionally NOT checked for spaces —
    # Comtrade uses padded codes (_X, XX, etc.) for aggregated partner groups.
    logger.debug("partnerISO space check skipped — padded codes are expected")

    # --- Duplicates on natural key
    duplicates = df[df.duplicated(subset=NATURAL_KEY, keep=False)]
    if not duplicates.empty:
        logger.warning(
            "Duplicate rows on natural key %s: %d rows — "
            "may result from overlapping extraction runs",
            NATURAL_KEY, len(duplicates),
        )


# ============================================================
# TRANSFORM
# ============================================================

def transform_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying silver transformations")

    # Remove duplicates on natural key — keeps first occurrence.
    # Duplicates can arise when a reporter appears in both a standard
    # extraction run and a subsequent --reporters correction run.
    before = len(df)
    df = df.drop_duplicates(subset=NATURAL_KEY, keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d duplicate rows", dropped)

    # Type casting
    df["year"]         = df["year"].astype(int)
    df["primaryValue"] = df["primaryValue"].astype(float)

    # Standardize country names with special characters
    NAME_CORRECTIONS = {
    "Côte d'Ivoire":  "Cote d'Ivoire",
    "CÃƒÂ´te d'Ivoire": "Cote d'Ivoire",
    "Curaçao":        "Curacao",
    "CuraÃƒÂ§ao":     "Curacao",
    "Türkiye":        "Turkiye",
    "TÃ¼rkiye":       "Turkiye",  
    "TÃƒÂ¼rkiye" :    "Turkiye",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    }
    df["reporterDesc"] = df["reporterDesc"].replace(NAME_CORRECTIONS)
    df["partnerDesc"] = df["partnerDesc"].replace(NAME_CORRECTIONS)

    logger.info("Transform complete | %d rows", len(df))
    return df


# ============================================================
# SAVE SILVER
# ============================================================

def save_to_silver(df: pd.DataFrame, connector=None) -> None:
    SILVER_TRADE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_TRADE_DATA_PATH, index=False)
    if connector:
        save_to_db(df, table="trade_data", schema="silver", connector=connector)
    logger.info("Silver trade data saved -> %s", SILVER_TRADE_DATA_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    """
    Load, validate and save silver trade data.
    Re-runnable after any bronze correction extraction —
    rglob picks up new reporter files automatically.
    """
    df = load_bronze_trade_data()
    log_data_quality(df)
    df = transform_trade_data(df)
    save_to_silver(df, connector=connector)


if __name__ == "__main__":
    run()
