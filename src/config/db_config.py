"""
Database configuration — reads and validates connection parameters from .env.

Required variables in .env:
    DB_BACKEND    : target backend — "sqlserver" or "databricks"

SQL Server:
    DB_DRIVER     : ODBC driver name (e.g. "ODBC Driver 17 for SQL Server")
    DB_SERVER     : server name or IP (e.g. "localhost" or "myserver\\SQLEXPRESS")
    DB_NAME       : database name

Databricks (future):
    DB_HOST       : workspace hostname
    DB_HTTP_PATH  : SQL warehouse HTTP path
    DB_TOKEN      : personal access token
    DB_CATALOG    : Unity Catalog name
    DB_SCHEMA_PREFIX: optional schema prefix (default: empty)

Note: DB_DRIVER, DB_SERVER, DB_NAME use Trusted_Connection (Windows auth).
      No password is stored — authentication is handled by the OS.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_BACKENDS = {"sqlserver", "databricks", "duckdb"}


def get_db_config(backend: str | None = None) -> dict:
    """
    Read, validate and return database connection parameters from environment.
    Raises EnvironmentError with a clear message if any required variable is missing.
    """
    backend = (backend or os.getenv("DB_BACKEND", "sqlserver")).lower()

    if backend not in SUPPORTED_BACKENDS:
        raise EnvironmentError(
            f"Unsupported DB_BACKEND '{backend}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    if backend == "sqlserver":
        return _get_sqlserver_config()

    if backend == "databricks":
        return _get_databricks_config()

    if backend == "duckdb":
        return _get_duckdb_config()


def _get_sqlserver_config() -> dict:
    required = {
        "driver":   os.getenv("DB_DRIVER"),
        "server":   os.getenv("DB_SERVER"),
        "database": os.getenv("DB_NAME"),
    }
    _validate(required, backend="sqlserver")
    return {"backend": "sqlserver", **required}


def _get_duckdb_config() -> dict:
    """
    DuckDB is an embedded database — no server or credentials required.
    DB_PATH points to the .duckdb file. Use ':memory:' for an in-memory database.
    """
    path = os.getenv("DB_PATH", "data/trade_analysis.duckdb")
    return {"backend": "duckdb", "path": path}


def _get_databricks_config() -> dict:
    required = {
        "host":      os.getenv("DB_HOST"),
        "http_path": os.getenv("DB_HTTP_PATH"),
        "token":     os.getenv("DB_TOKEN"),
        "catalog":   os.getenv("DB_CATALOG"),
    }
    _validate(required, backend="databricks")
    return {"backend": "databricks", **required}


def _validate(params: dict, backend: str) -> None:
    missing = [key for key, val in params.items() if not val]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s) for backend '{backend}': "
            f"{', '.join(missing.upper() if isinstance(missing, str) else [f'DB_{k.upper()}' for k in missing])}.\n"
            f"Check your .env file."
        )
