"""
Bootstrap — Create bronze, silver and gold schemas in the database.

Run once after create_database.py, before the first pipeline execution.

Usage:
    python -m src.bootstrap.create_schemas
"""

import logging

from sqlalchemy import text

from src.db.connector import get_connector

logger = logging.getLogger(__name__)

SCHEMAS = ["bronze", "silver", "gold"]


def create_schemas(db: str | None = None) -> None:
    connector = get_connector(backend=db)
    engine    = connector.get_engine()

    logger.info("Creating schemas: %s | backend=%s", SCHEMAS, connector.backend)

    with engine.begin() as conn:
        for schema in SCHEMAS:
            if connector.backend == "duckdb":
                # DuckDB uses standard SQL CREATE SCHEMA IF NOT EXISTS
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            else:
                # SQL Server requires sys.schemas check + EXEC
                conn.execute(text(f"""
                    IF NOT EXISTS (
                        SELECT 1 FROM sys.schemas WHERE name = '{schema}'
                    )
                    EXEC('CREATE SCHEMA {schema}')
                """))
            logger.info("Schema '%s' is ready", schema)

    logger.info("All schemas created successfully")


if __name__ == "__main__":
    import argparse
    from src.config.logging_config import setup_logger
    setup_logger("src")

    parser = argparse.ArgumentParser(description="Bootstrap — create medallion schemas")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        choices=["duckdb", "sqlserver", "databricks"],
        help="Backend override (default: reads DB_BACKEND from .env)",
    )
    args = parser.parse_args()
    create_schemas(db=args.db)
