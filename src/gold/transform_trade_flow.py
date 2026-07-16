import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    SILVER_TRADE_FLOW_PATH,
    SILVER_TRADE_DATA_PATH,
    GOLD_TRADE_FLOW_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver trade flow reference and silver trade data."""
    logger.info("Loading silver datasets")
    df_flows = pd.read_csv(SILVER_TRADE_FLOW_PATH)
    df_trade = pd.read_csv(SILVER_TRADE_DATA_PATH)
    return df_flows, df_trade


# ============================================================
# VALIDATE
# ============================================================

def validate_flows(df_flows: pd.DataFrame, df_trade: pd.DataFrame) -> pd.DataFrame:
    """Ensure all flow codes present in trade data exist in the trade flow reference."""
    flow_codes  = set(df_flows["flow_code"].dropna().unique())
    trade_codes = set(df_trade["flowCode"].dropna().unique())

    missing_codes = trade_codes - flow_codes
    if missing_codes:
        raise ValueError(f"Flow codes in trade data not found in reference: {missing_codes}")

    return df_flows


# ============================================================
# BUILD GOLD DIMENSION
# ============================================================

def build_gold_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Build the gold trade_flow dimension with a surrogate key."""
    df = df.reset_index(drop=True).copy()
    df["flow_key"] = df.index + 1

    return df[["flow_key", "flow_code", "flow_description"]]


# ============================================================
# SAVE
# ============================================================

def save_flows(df: pd.DataFrame, connector=None) -> None:
    GOLD_TRADE_FLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_TRADE_FLOW_PATH, index=False)
    if connector:
        save_to_db(df, table="trade_flow", schema="gold", connector=connector)
    logger.info("Gold trade flow saved -> %s", GOLD_TRADE_FLOW_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df_flows, df_trade = load_tables()
    df_flows = validate_flows(df_flows, df_trade)
    df_gold  = build_gold_flows(df_flows)
    save_flows(df_gold, connector=connector)


if __name__ == "__main__":
    run()
