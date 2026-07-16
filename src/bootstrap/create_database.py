"""
Bootstrap — Create SQL Server database.

Run once before the first pipeline execution.
Connects to the SQL Server master database and creates the target
database if it does not already exist.

Usage:
    python -m src.bootstrap.create_database
"""

import logging
import os

import pyodbc
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def create_database_if_not_exists() -> None:
    driver   = os.getenv("DB_DRIVER")
    server   = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")

    missing = [k for k, v in {"DB_DRIVER": driver, "DB_SERVER": server, "DB_NAME": database}.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            f"Check your .env file."
        )

    logger.info("Checking SQL Server database existence | server=%s database=%s", server, database)

    # Connect to master (target DB may not exist yet)
    connection_string = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        "DATABASE=master;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    sql = f"""
    IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = '{database}')
    BEGIN
        CREATE DATABASE {database};
    END
    """

    cnxn = None
    cursor = None

    try:
        cnxn   = pyodbc.connect(connection_string, autocommit=True)
        cursor = cnxn.cursor()
        cursor.execute(sql)
        logger.info("Database '%s' is ready", database)

    except pyodbc.Error:
        logger.exception("Error while creating database '%s'", database)
        raise

    finally:
        if cursor:
            cursor.close()
        if cnxn:
            cnxn.close()
        logger.info("SQL Server connection closed")


if __name__ == "__main__":
    from src.config.logging_config import setup_logger
    setup_logger("src")
    create_database_if_not_exists()
