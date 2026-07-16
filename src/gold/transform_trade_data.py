import logging

import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import (
    SILVER_TRADE_DATA_PATH,
    GOLD_ENTITIES_PATH,
    GOLD_PRODUCTS_PATH,
    GOLD_TRADE_FLOW_PATH,
    GOLD_TRADE_DATA_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load inputs for the gold trade_data fact table.

    Trade data is read from silver (raw fact data to be joined).
    Dimension tables (products, trade_flow, entities) are read from gold —
    they carry the surrogate keys needed for joins, and gold products
    includes the TOTAL aggregation row absent from silver.
    """
    logger.info("Loading silver trade data and gold dimension tables")
    df_trade    = pd.read_csv(SILVER_TRADE_DATA_PATH)
    df_products = pd.read_csv(GOLD_PRODUCTS_PATH)
    df_flows    = pd.read_csv(GOLD_TRADE_FLOW_PATH)
    df_entities = pd.read_csv(GOLD_ENTITIES_PATH)
    logger.info(
        "Loaded | trade=%d  products=%d  flows=%d  entities=%d",
        len(df_trade), len(df_products), len(df_flows), len(df_entities),
    )
    return df_trade, df_products, df_flows, df_entities


# ============================================================
# VALIDATE REFERENTIAL INTEGRITY
# ============================================================

def validate_references(
    df_trade: pd.DataFrame,
    df_products: pd.DataFrame,
    df_flows: pd.DataFrame,
    df_entities: pd.DataFrame,
) -> None:
    """
    Ensure all foreign keys in the trade data exist in their respective
    dimension tables. Raises ValueError on any mismatch.

    This validation runs after all bronze corrections have been applied —
    failures here indicate a missing correction step, not a data gap to log.
    """
    logger.info("Validating referential integrity")

    # Products
    missing_products = (
        set(df_trade["cmdCode"].dropna().unique())
        - set(df_products["cmdCode"].dropna().unique())
    )
    if missing_products:
        raise ValueError(f"cmdCode values not in gold products: {missing_products}")

    # Trade flows
    missing_flows = (
        set(df_trade["flowCode"].dropna().unique())
        - set(df_flows["flow_code"].dropna().unique())
    )
    if missing_flows:
        raise ValueError(f"flowCode values not in gold trade_flow: {missing_flows}")

    # Reporter entities — join on country_code (Comtrade code)
    entity_codes = set(df_entities["country_code"].dropna().astype(int).unique())
    missing_reporters = (
        set(df_trade["reporterCode"].dropna().astype(int).unique())
        - entity_codes
    )
    if missing_reporters:
        raise ValueError(
            f"reporterCode values not in gold entities: {missing_reporters}\n"
            f"Run bronze correction: "
            f"python -m src.bronze.extract_trade_data --reporters {' '.join(str(c) for c in sorted(missing_reporters))}"
        )

    # Partner entities — join on country_code (Comtrade code)
    missing_partners = (
        set(df_trade["partnerCode"].dropna().astype(int).unique())
        - entity_codes
    )
    if missing_partners:
        raise ValueError(
            f"partnerCode values not in gold entities: {missing_partners}"
        )

    logger.info("Referential integrity checks passed")


# ============================================================
# BUILD GOLD FACT TABLE
# ============================================================

def build_gold_trade(
    df_trade: pd.DataFrame,
    df_products: pd.DataFrame,
    df_flows: pd.DataFrame,
    df_entities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the gold trade_data fact table by joining silver trade data
    with gold dimension surrogate keys.

    Each row has two entity relationships:
      - reporter_key  -> the country reporting the trade flow
      - partner_key   -> the trading partner country
    Both are resolved via separate joins on the entities table using
    country_code (Comtrade code).
    """
    logger.info("Building gold trade_data fact table")

    # Prepare dimension lookup tables
    products_lookup  = df_products[["cmdCode", "product_key"]]
    flows_lookup     = df_flows[["flow_code", "flow_key"]].rename(columns={"flow_code": "flowCode"})
    # Drop NaN country_codes — WB-only entities have no Comtrade code
    # and cannot match any reporter/partner in trade data
    entities_lookup = (
        df_entities[["country_code", "entity_key"]]
        .dropna(subset=["country_code"])
        .copy()
    )
    entities_lookup["country_code"] = entities_lookup["country_code"].astype(int)

    df = (
        df_trade
        # Join product key
        .merge(products_lookup, on="cmdCode", how="left")
        # Join flow key
        .merge(flows_lookup, on="flowCode", how="left")
        # Join reporter entity key
        .merge(
            entities_lookup.rename(columns={"country_code": "reporterCode", "entity_key": "reporter_key"}),
            on="reporterCode",
            how="left",
        )
        # Join partner entity key
        .merge(
            entities_lookup.rename(columns={"country_code": "partnerCode", "entity_key": "partner_key"}),
            on="partnerCode",
            how="left",
        )
    )

    # Post-join NaN check — null surrogate keys indicate countries whose
    # Comtrade code differs from their WB code (code divergence).
    # These reporters/partners were not matched in the entities table.
    # Re-extract their trade data using the Comtrade codes via --reporters.
    key_cols = ["product_key", "flow_key", "reporter_key", "partner_key"]
    null_keys = df[key_cols].isna().sum()

    if null_keys.any():
        # Identify missing reporter codes
        missing_reporters = (
            df[df["reporter_key"].isna()]["reporterCode"]
            .dropna().astype(int).unique().tolist()
        )
        # Identify missing partner codes
        missing_partners = (
            df[df["partner_key"].isna()]["partnerCode"]
            .dropna().astype(int).unique().tolist()
        )

        all_missing = sorted(set(missing_reporters + missing_partners))

        logger.warning(
            "Null surrogate keys detected — likely due to WB/Comtrade code divergence.\n"
            "Missing reporter codes : %s\n"
            "Missing partner  codes : %s\n"
            "Re-extraction required :\n"
            "  python -m src.bronze.extract_trade_data "
            "--year <year> --product <product> --flows <flows> --reporters %s\n"
            "Then re-run: run_silver --module trade_data && run_gold",
            missing_reporters,
            missing_partners,
            " ".join(str(c) for c in all_missing),
        )

        raise ValueError(
            f"Null surrogate keys on: {null_keys[null_keys > 0].to_dict()}\n"
            f"Missing Comtrade codes: {all_missing}\n"
            f"See log above for the correction command."
        )

    # Surrogate key
    df = df.reset_index(drop=True)
    df["trade_key"] = df.index + 1

    # Final schema
    df_gold = df[[
        "trade_key", "typeCode",
        "reporter_key", "partner_key",
        "product_key",  "flow_key",
        "year",
        "primaryValue",
    ]]

    logger.info("Gold trade_data built | %d rows", len(df_gold))
    return df_gold


# ============================================================
# SAVE
# ============================================================

def save_trade_data(df: pd.DataFrame, connector=None) -> None:
    GOLD_TRADE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_TRADE_DATA_PATH, index=False)
    if connector:
        save_to_db(df, table="trade_data", schema="gold", connector=connector)
    logger.info("Gold trade_data saved -> %s", GOLD_TRADE_DATA_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    df_trade, df_products, df_flows, df_entities = load_tables()
    validate_references(df_trade, df_products, df_flows, df_entities)
    df_gold = build_gold_trade(df_trade, df_products, df_flows, df_entities)
    save_trade_data(df_gold, connector=connector)


if __name__ == "__main__":
    run()
