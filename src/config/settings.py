from pathlib import Path

# =========================
# Project base directory
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

# =========================
# Data layers
# =========================
DATA_DIR    = BASE_DIR / "data"

DATA_BRONZE = DATA_DIR / "bronze"
DATA_SILVER = DATA_DIR / "silver"
DATA_GOLD   = DATA_DIR / "gold"
DATA_SOURCE = DATA_DIR / "resources"   # static source files (indicators.csv, products.csv)

# =========================
# Bronze subfolders
# =========================
BRONZE_COUNTRIES         = DATA_BRONZE / "countries"
BRONZE_PRODUCTS          = DATA_BRONZE / "products"
BRONZE_TRADE_DATA        = DATA_BRONZE / "trade_data"       # partitioned extraction output
BRONZE_TRADE_FLOW        = DATA_BRONZE / "trade_flow"
BRONZE_INDICATORS        = DATA_BRONZE / "indicators"
BRONZE_INDICATORS_VALUES = DATA_BRONZE / "indicators_values"

BRONZE_COUNTRIES_PATH         = BRONZE_COUNTRIES         / "countries.csv"
BRONZE_PRODUCTS_PATH          = BRONZE_PRODUCTS          / "products.csv"
BRONZE_TRADE_FLOW_PATH        = BRONZE_TRADE_FLOW        / "trade_flow.csv"
BRONZE_INDICATORS_PATH        = BRONZE_INDICATORS        / "indicators.csv"
BRONZE_INDICATORS_VALUES_PATH = BRONZE_INDICATORS_VALUES / "indicators_values.csv"

# =========================
# Silver subfolders
# =========================
SILVER_COUNTRIES         = DATA_SILVER / "countries"
SILVER_PRODUCTS          = DATA_SILVER / "products"
SILVER_TRADE_DATA        = DATA_SILVER / "trade_data"
SILVER_TRADE_FLOW        = DATA_SILVER / "trade_flow"
SILVER_INDICATORS        = DATA_SILVER / "indicators"
SILVER_INDICATORS_VALUES = DATA_SILVER / "indicators_values"

SILVER_COUNTRIES_PATH         = SILVER_COUNTRIES         / "countries.csv"
SILVER_PRODUCTS_PATH          = SILVER_PRODUCTS          / "products.csv"
SILVER_TRADE_FLOW_PATH        = SILVER_TRADE_FLOW        / "trade_flow.csv"
SILVER_INDICATORS_PATH        = SILVER_INDICATORS        / "indicators.csv"
SILVER_INDICATORS_VALUES_PATH = SILVER_INDICATORS_VALUES / "indicators_values.csv"
SILVER_TRADE_DATA_PATH        = SILVER_TRADE_DATA        / "trade_data.csv"

# =========================
# Gold subfolders
# =========================
GOLD_ENTITIES         = DATA_GOLD / "entities"
GOLD_PRODUCTS         = DATA_GOLD / "products"
GOLD_TRADE_DATA       = DATA_GOLD / "trade_data"
GOLD_TRADE_FLOW       = DATA_GOLD / "trade_flow"
GOLD_INDICATORS       = DATA_GOLD / "indicators"
GOLD_INDICATORS_VALUES = DATA_GOLD / "indicators_values"

GOLD_ENTITIES_PATH         = GOLD_ENTITIES         / "entities.csv"
GOLD_PRODUCTS_PATH         = GOLD_PRODUCTS         / "products.csv"
GOLD_TRADE_FLOW_PATH       = GOLD_TRADE_FLOW       / "trade_flow.csv"
GOLD_INDICATORS_PATH       = GOLD_INDICATORS       / "indicators.csv"
GOLD_INDICATORS_VALUES_PATH = GOLD_INDICATORS_VALUES / "indicators_values.csv"
GOLD_TRADE_DATA_PATH       = GOLD_TRADE_DATA       / "trade_data.csv"

# =========================
# Time configuration
# =========================
START_YEAR = 2015
END_YEAR   = 2024
YEARS      = list(range(START_YEAR, END_YEAR + 1))


# =========================
# Directory initialisation
# =========================
def ensure_directories() -> None:
    """
    Create all data layer directories if they do not exist.
    Call this once at the start of each pipeline entrypoint.
    Not called at import time so that notebooks and tests do not
    trigger unexpected filesystem writes.

    Note: DATA_SOURCE is intentionally excluded — it must be
    pre-populated by the user with source files (indicators.csv,
    products.csv) before running any pipeline.
    """
    dirs = [
        DATA_BRONZE,
        DATA_SILVER,
        DATA_GOLD,
        # Bronze
        BRONZE_COUNTRIES,
        BRONZE_PRODUCTS,
        BRONZE_TRADE_DATA,
        BRONZE_TRADE_FLOW,
        BRONZE_INDICATORS,
        BRONZE_INDICATORS_VALUES,
        # Silver
        SILVER_COUNTRIES,
        SILVER_PRODUCTS,
        SILVER_TRADE_DATA,
        SILVER_TRADE_FLOW,
        SILVER_INDICATORS,
        SILVER_INDICATORS_VALUES,
        # Gold
        GOLD_ENTITIES,
        GOLD_PRODUCTS,
        GOLD_TRADE_DATA,
        GOLD_TRADE_FLOW,
        GOLD_INDICATORS,
        GOLD_INDICATORS_VALUES,
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
