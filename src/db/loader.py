"""
Database loader — generic DataFrame-to-table save function.

Each module (bronze, silver, gold) calls save_to_db() alongside its CSV save.
The connector is created once in the pipeline entrypoint and passed down.

Usage in a module:
    from src.db.loader import save_to_db

    def save_to_silver(df: pd.DataFrame, connector=None) -> None:
        df.to_csv(SILVER_COUNTRIES_PATH, index=False)
        if connector:
            save_to_db(df, table="countries", schema="silver", connector=connector)

Schemas:
    bronze  — raw extracted data
    silver  — cleaned and validated data
    gold    — star schema dimensions and fact tables
"""

import logging

import pandas as pd

from src.db.connector import BaseConnector

logger = logging.getLogger(__name__)

VALID_SCHEMAS      = {"bronze", "silver", "gold"}
DEFAULT_CHUNKSIZE  = 500
LARGE_CHUNKSIZE    = 5000   # for tables with 1M+ rows

# Backends that support multi-row INSERT (method="multi") and their max bound
# parameters per statement. SQL Server is deliberately excluded: it already
# gets fast, driver-level batching via pyodbc's fast_executemany (set on its
# engine in connector.py) -- method="multi" would replace that with one big
# multi-VALUES statement instead and risks hitting SQL Server's own ~2100
# parameter limit on wide tables.
MULTI_ROW_PARAM_LIMITS = {
    "databricks": 10_000,  # BAD_REQUEST if a single query exceeds this
    "duckdb":     100_000,  # no meaningful limit at our data sizes; generous cap
}


def save_to_db(
    df: pd.DataFrame,
    table: str,
    schema: str,
    connector: BaseConnector,
    if_exists: str = "replace",
    chunksize: int | None = None,
) -> None:
    """
    Save a DataFrame as a database table.

    Args:
        df:         DataFrame to save.
        table:      Table name (without schema prefix).
        schema:     Target schema — "bronze", "silver", or "gold".
        connector:  Connector instance from get_connector().
        if_exists:  SQLAlchemy behavior if table exists — "replace" (default)
                    or "append". "replace" truncates and reloads, consistent
                    with the pipeline's clean-slate philosophy.
    """
    if schema not in VALID_SCHEMAS:
        raise ValueError(
            f"Invalid schema '{schema}'. Valid schemas: {', '.join(sorted(VALID_SCHEMAS))}"
        )

    engine = connector.get_engine()

    logger.info(
        "Saving to DB | backend=%s schema=%s table=%s rows=%d",
        connector.backend, schema, table, len(df),
    )

    try:
        effective_chunksize = chunksize or DEFAULT_CHUNKSIZE
        method = None

        param_limit = MULTI_ROW_PARAM_LIMITS.get(connector.backend)
        if param_limit is not None:
            method = "multi"
            n_cols = max(len(df.columns), 1)
            max_rows_per_statement = max(param_limit // n_cols, 1)
            if effective_chunksize > max_rows_per_statement:
                logger.info(
                    "Capping chunksize for %s | requested=%d -> %d (param limit %d / %d cols)",
                    connector.backend, effective_chunksize, max_rows_per_statement,
                    param_limit, n_cols,
                )
                effective_chunksize = max_rows_per_statement

        df.to_sql(
            name=table,
            schema=schema,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=effective_chunksize,
            method=method,
        )
        logger.info(
            "Saved -> %s.%s | %d rows", schema, table, len(df)
        )

    except Exception:
        logger.exception(
            "Failed to save %s.%s to %s", schema, table, connector.backend
        )
        raise
