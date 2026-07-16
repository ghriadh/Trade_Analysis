import json
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from src.db.loader import save_to_db
from src.config.settings import BRONZE_COUNTRIES_PATH

logger = logging.getLogger(__name__)


def extract_countries() -> pd.DataFrame:
    """
    Extract countries data from the mledoze/countries GitHub repository.
    Bronze layer: raw fields only, no business transformations.
    Falls back to local bronze cache if the API is unreachable.
    """
    url = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"

    logger.info("Starting countries extraction")

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]
    )))

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        if BRONZE_COUNTRIES_PATH.exists():
            logger.warning("API unreachable (%s) — loading from local bronze cache", e)
            return pd.read_csv(BRONZE_COUNTRIES_PATH)
        raise

    try:
        countries = response.json()
    except json.JSONDecodeError:
        raise ValueError("JSON decoding error — invalid response from REST Countries source")

    records = []
    for c in countries:
        if not isinstance(c, dict) or "cca3" not in c:
            continue

        records.append({
            "country_code":    c.get("ccn3"),
            "countryiso3code": c.get("cca3"),
            "country_name":    c.get("name", {}).get("common"),
            "region":          c.get("region"),
            "subregion":       c.get("subregion"),
            "capital": (c.get("capital")[0] if c.get("capital") else None),
            "population":      c.get("population"),
            "currency":        ", ".join(c.get("currencies", {}).keys()) or None,
            "language":        ", ".join(c.get("languages", {}).values()) or None,
        })

    df = pd.DataFrame(records)
    logger.info("%d countries extracted", len(df))

    return df


def save_countries_to_bronze(df: pd.DataFrame, connector=None) -> None:
    """Save countries DataFrame to the bronze layer."""
    df.to_csv(BRONZE_COUNTRIES_PATH, index=False)
    if connector:
        save_to_db(df, table="countries", schema="bronze", connector=connector)
    logger.info("Countries saved → %s", BRONZE_COUNTRIES_PATH)


def run(connector=None) -> None:
    """Orchestrate countries extraction (extract + save)."""
    df = extract_countries()
    save_countries_to_bronze(df, connector=connector)


if __name__ == "__main__":
    run()
