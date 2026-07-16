import json
import logging
from pathlib import Path

import pandas as pd

from src.db.loader import save_to_db
# Path to the known divergences config file.
# Add ISO3 codes here after a successful correction extraction
# to suppress the warning on subsequent gold runs.
KNOWN_DIVERGENCES_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "known_divergences.json"

from src.config.settings import (
    SILVER_COUNTRIES_PATH,
    SILVER_INDICATORS_VALUES_PATH,
    SILVER_TRADE_DATA_PATH,
    GOLD_ENTITIES_PATH,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three silver datasets needed to build the entities table."""
    logger.info("Loading silver datasets")
    countries  = pd.read_csv(SILVER_COUNTRIES_PATH)
    indicators = pd.read_csv(SILVER_INDICATORS_VALUES_PATH)
    trade      = pd.read_csv(SILVER_TRADE_DATA_PATH)
    return countries, indicators, trade


# ============================================================
# BUILD ENTITIES
# ============================================================

def build_entities(
    countries: pd.DataFrame,
    indicators: pd.DataFrame,
    trade: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical entities table by merging country references
    from three sources: WB countries, WB indicator values, and Comtrade
    trade data (partners).

    Schema outcome:
        country_code / country_iso3 / country_name   -> Comtrade identifiers
        wb_country_code / wb_country_iso3 / wb_country_name -> WB identifiers
        country_id  -> WB 2-letter internal identifier (from indicators_values)

    Quality warnings are logged for unmatched codes — these flag candidates
    for a bronze correction extraction (--reporters).
    No exception is raised: the pipeline produces a complete entities table
    even when gaps exist.
    """
    logger.info("Building canonical entities")

    # ----------------------------------------------------------
    # 1. Enrich WB countries with country_id from indicators_values.
    #    country_id is the WB internal 2-letter code (e.g. "FR", "US").
    # ----------------------------------------------------------
    df_country_id = (
        indicators[["countryiso3code", "country_id"]]
        .drop_duplicates(subset=["countryiso3code"])
    )
    countries = pd.merge(countries, df_country_id, on="countryiso3code", how="left")

    # ----------------------------------------------------------
    # 2. WB entities dimension
    # ----------------------------------------------------------
    entities_wb = (
        countries[["country_code", "countryiso3code", "country_name", "country_id"]]
        .rename(columns={
            "countryiso3code": "wb_country_iso3",
            "country_code":    "wb_country_code",
            "country_name":    "wb_country_name",
        })
    )

    # ----------------------------------------------------------
    # 3. Entities from indicators_values not in WB countries.
    #    These are non-country aggregates (regions, income groups, etc.)
    #    They have no numeric WB code — assigned 999.
    # ----------------------------------------------------------
    entities_ind = (
        indicators[["countryiso3code", "country_name", "country_id"]]
        .drop_duplicates(subset=["countryiso3code"])
        .rename(columns={
            "countryiso3code": "wb_country_iso3",
            "country_name":    "wb_country_name",
        })
    )

    missing_ind = (
        set(entities_ind["wb_country_iso3"].dropna())
        - set(entities_wb["wb_country_iso3"].dropna())
    )
    if missing_ind:
        logger.warning(
            "Indicator entities not in WB countries (%d) — assigned wb_country_code=999: %s",
            len(missing_ind), sorted(missing_ind),
        )

    df_missing_ind = entities_ind[entities_ind["wb_country_iso3"].isin(missing_ind)].copy()
    df_missing_ind["wb_country_code"] = 999

    # ----------------------------------------------------------
    # 4. Merge WB countries + indicator entities
    # ----------------------------------------------------------
    entities = (
        pd.concat([entities_wb, df_missing_ind], ignore_index=True)
        .drop_duplicates(subset=["wb_country_iso3"], keep="first")
    )
    entities["wb_country_code"] = pd.to_numeric(entities["wb_country_code"], errors="coerce").fillna(999).astype("Int64")

    # ----------------------------------------------------------
    # 5. Comtrade partner entities
    # ----------------------------------------------------------
    entities_trade = (
        trade[["partnerISO", "partnerCode", "partnerDesc"]]
        .drop_duplicates()
        .rename(columns={
            "partnerISO":  "country_iso3",
            "partnerCode": "country_code",
            "partnerDesc": "country_name",
        })
    )

    # Quality: trade reporter codes not matched to WB entities
    missing_reporters = (
        set(trade["reporterCode"].dropna().astype(int))
        - set(entities["wb_country_code"].dropna().astype(int))
    )
    if missing_reporters:
        logger.warning(
            "Trade reporter codes not matched to WB entities (%d) — "
            "check for code divergence: %s",
            len(missing_reporters), sorted(missing_reporters),
        )

    # Quality: trade partner ISO3 not matched to WB entities
    missing_partners = (
        set(entities_trade["country_iso3"].dropna())
        - set(entities["wb_country_iso3"].dropna())
    )
    if missing_partners:
        logger.warning(
            "Trade partner ISO3 not in WB entities (%d) — "
            "will be added to entities table: %s",
            len(missing_partners), sorted(missing_partners),
        )

    # ----------------------------------------------------------
    # 6. Outer merge: Comtrade partners <-> WB + indicator entities.
    #    Both country_code (Comtrade) and wb_country_code (WB) are kept.
    #    Divergences between the two are surfaced by detect_code_divergence().
    # ----------------------------------------------------------
    entities = pd.merge(
        entities_trade,
        entities,
        left_on="country_iso3",
        right_on="wb_country_iso3",
        how="outer",
    )

    # ----------------------------------------------------------
    # 7. World entity — special case.
    #
    # "World" appears in both WB countries (wb_country_name="World") and
    # Comtrade partners (country_name="World", partnerCode=0).
    # The WB row carries no Comtrade code — it is dropped.
    # The Comtrade row is kept and enriched with a canonical WB identity.
    # ----------------------------------------------------------
    entities = entities[entities["wb_country_name"] != "World"].copy()

    world_mask = entities["country_name"] == "World"
    entities.loc[world_mask, "wb_country_code"] = 0
    entities.loc[world_mask, "wb_country_iso3"] = "WLD"
    entities.loc[world_mask, "wb_country_name"] = "World"

    logger.info("Entities built: %d rows", len(entities))
    return entities


# ============================================================
# CODE DIVERGENCE DETECTION
# ============================================================

def _load_known_divergences() -> set[str]:
    """
    Load the set of ISO3 codes whose WB/Comtrade divergence is known
    and already corrected via a --reporters re-extraction.

    To mark divergences as resolved, add the ISO3 codes to:
        data/config/known_divergences.json
    Example: ["CHE", "FRA", "IND", "NOR", "USA"]
    """
    if not KNOWN_DIVERGENCES_PATH.exists():
        return set()
    try:
        with open(KNOWN_DIVERGENCES_PATH, "r", encoding="utf-8") as f:
            codes = json.load(f)
        return set(codes)
    except Exception:
        logger.warning("Could not read %s -- treating all divergences as new",
                       KNOWN_DIVERGENCES_PATH)
        return set()


def detect_code_divergence(entities: pd.DataFrame) -> pd.DataFrame:
    """
    Detect countries whose Comtrade code (country_code) differs from their
    WB code (wb_country_code) despite sharing the same ISO3 code.

    These countries are missing from the trade data because the extraction
    used the WB code while Comtrade expects a different one. A re-extraction
    is required using the Comtrade codes via the --reporters flag.

    Logs a warning with the divergent rows and the ready-to-use --reporters
    command. Returns the divergent rows DataFrame for reference.
    """
    mask = (
        entities["country_code"].notna()
        & entities["wb_country_code"].notna()
        & ~entities["wb_country_code"].isin([0, 999])
        & (
            pd.to_numeric(entities["country_code"], errors="coerce").astype("Int64")
            != pd.to_numeric(entities["wb_country_code"], errors="coerce").astype("Int64")
        )
    )
    divergent = entities[mask][[
        "country_iso3",
        "country_name",    "country_code",
        "wb_country_name", "wb_country_code",
    ]].copy()

    if divergent.empty:
        logger.info("No code divergence detected between WB and Comtrade")
        return divergent

    # Filter out known/resolved divergences
    known = _load_known_divergences()
    if known:
        resolved   = divergent[divergent["country_iso3"].isin(known)]
        divergent  = divergent[~divergent["country_iso3"].isin(known)]
        if not resolved.empty:
            logger.info(
                "Known divergence (already corrected) -- skipping: %s",
                resolved["country_iso3"].tolist(),
            )

    if divergent.empty:
        logger.info("All divergences are known and resolved")
        return divergent

    comtrade_codes = " ".join(
        divergent["country_code"].dropna().astype(int).astype(str).tolist()
    )
    logger.warning(
        "Code divergence -- %d country/ies have different WB and Comtrade codes.\n"
        "Re-extraction required:\n"
        "  python -m src.bronze.extract_trade_data "
        "--year <year> --product <product> --flows <flows> --reporters %s\n%s",
        len(divergent),
        comtrade_codes,
        divergent.to_string(index=False),
    )
    return divergent


# ============================================================
# CLEAN & NORMALIZE
# ============================================================

def clean_entities(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning and normalizing entities")

    # Strip leading/trailing spaces from string columns
    for col in ["country_iso3", "country_name", "wb_country_iso3", "wb_country_name"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Entity type:
    #   COUNTRY -> valid WB numeric code (not 0 for World, not 999 for aggregates)
    #   ENTITY  -> non-country aggregate (region, income group, etc.) or World
    df["entity_type"] = df["wb_country_code"].apply(
        lambda x: "COUNTRY" if x not in {0, 999} else "ENTITY"
    )

    # Surrogate key — index-based, consistent with products, indicators and
    # trade_flow gold modules. Not guaranteed stable across re-runs: a full
    # gold re-run (all modules) is required after any bronze correction extraction.
    df = df.reset_index(drop=True)
    df["entity_key"] = df.index + 1

    # Final schema
    df = df[[
        "entity_key",
        "country_id",
        "country_code",    "country_iso3",    "country_name",
        "wb_country_code", "wb_country_iso3",  "wb_country_name",
        "entity_type",
    ]]

    return df


# ============================================================
# SAVE
# ============================================================

def save_entities(df: pd.DataFrame, connector=None) -> None:
    GOLD_ENTITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_ENTITIES_PATH, index=False)
    if connector:
        save_to_db(df, table="entities", schema="gold", connector=connector)
    logger.info("Gold entities saved -> %s", GOLD_ENTITIES_PATH)


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def run(connector=None) -> None:
    countries, indicators, trade = load_inputs()
    entities = build_entities(countries, indicators, trade)
    detect_code_divergence(entities)
    entities = clean_entities(entities)
    save_entities(entities, connector=connector)


if __name__ == "__main__":
    run()
