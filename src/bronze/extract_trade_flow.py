import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_TRADE_FLOW_PATH

logger = logging.getLogger(__name__)


def extract_trade_flow() -> pd.DataFrame:
    """
    Build the trade flow reference table (static mapping).
    Bronze layer: hardcoded Comtrade flow codes — no API call required.

    Comtrade flow codes:
        M  — Import
        X  — Export
        RM — Re-Import
        RX — Re-Export
    """
    logger.info("Building trade flow reference data")

    data = {
        "flow_code":        ["M",      "X",      "RM",        "RX"],
        "flow_description": ["IMPORT", "EXPORT", "RE_IMPORT", "RE_EXPORT"],
    }

    df = pd.DataFrame(data)
    logger.info("%d trade flow codes defined", len(df))

    return df


def save_flows_to_bronze(df: pd.DataFrame, connector=None) -> None:
    """Save trade flow DataFrame to the bronze layer."""
    df.to_csv(BRONZE_TRADE_FLOW_PATH, index=False)
    if connector:
        save_to_db(df, table="trade_flow", schema="bronze", connector=connector)
    logger.info("Trade flow saved → %s", BRONZE_TRADE_FLOW_PATH)


def run(connector=None) -> None:
    """Orchestrate trade flow extraction (build + save)."""
    df = extract_trade_flow()
    save_flows_to_bronze(df, connector=connector)


if __name__ == "__main__":
    run()
