"""
Database connector factory.

Usage
-----
# From pipeline entrypoint (backend read from DB_BACKEND env var):
    from src.db.connector import get_connector
    connector = get_connector()
    engine = connector.get_engine()

# Explicit backend override:
    connector = get_connector(backend="sqlserver")
"""

import logging
import urllib
from abc import ABC, abstractmethod

from sqlalchemy import create_engine, Engine

from src.config.db_config import get_db_config

logger = logging.getLogger(__name__)


# ============================================================
# BASE CLASS
# ============================================================

class BaseConnector(ABC):
    """Abstract connector — all backends implement get_engine()."""

    @abstractmethod
    def get_engine(self) -> Engine:
        """Return a SQLAlchemy engine for this backend."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend identifier string."""

    def test_connection(self) -> dict:
        """
        Verify the database connection is reachable.
        Returns a status dict suitable for ETL dashboard display.

        Returns:
            {"status": "ok",    "backend": str}  on success
            {"status": "error", "backend": str, "detail": str}  on failure
        """
        from sqlalchemy import text
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connection test OK | backend=%s", self.backend)
            return {"status": "ok", "backend": self.backend}
        except Exception as e:
            logger.warning("Connection test FAILED | backend=%s | %s", self.backend, e)
            return {"status": "error", "backend": self.backend, "detail": str(e)}


# ============================================================
# SQL SERVER
# ============================================================

class SQLServerConnector(BaseConnector):
    """
    SQL Server connector using Windows Trusted Authentication (no password).
    Relies on pyodbc + SQLAlchemy mssql dialect.
    fast_executemany=True significantly speeds up bulk inserts (df.to_sql).
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._engine: Engine | None = None

    @property
    def backend(self) -> str:
        return "sqlserver"

    def get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine

        cfg = self._config
        conn_str = (
            f"DRIVER={cfg['driver']};"
            f"SERVER={cfg['server']};"
            f"DATABASE={cfg['database']};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
            "Connection Timeout=30;"
        )
        url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
        self._engine = create_engine(url, fast_executemany=True)

        logger.info(
            "SQL Server engine created | server=%s database=%s",
            cfg["server"], cfg["database"],
        )
        return self._engine


# ============================================================
# DATABRICKS (stub — to be implemented)
# ============================================================

class DatabricksConnector(BaseConnector):
    """
    Databricks SQL connector (Unity Catalog).
    Requires: databricks-sqlalchemy or databricks-sql-connector.

    To implement: pip install sqlalchemy-databricks
    Connection URL: databricks+connector://token:<token>@<host>/<catalog>
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._engine: Engine | None = None

    @property
    def backend(self) -> str:
        return "databricks"

    def get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine

        cfg = self._config
        url = (
            f"databricks+connector://token:{cfg['token']}"
            f"@{cfg['host']}/{cfg['catalog']}"
            f"?http_path={urllib.parse.quote_plus(cfg['http_path'])}"
        )
        self._engine = create_engine(url)

        logger.info(
            "Databricks engine created | host=%s catalog=%s",
            cfg["host"], cfg["catalog"],
        )
        return self._engine


# ============================================================
# DUCKDB
# ============================================================

class DuckDBConnector(BaseConnector):
    """
    DuckDB embedded connector.
    No server or credentials required — connects to a local .duckdb file.
    Schemas (bronze, silver, gold) are created by bootstrap/create_schemas.py.

    Requires: pip install duckdb duckdb-engine
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._engine: Engine | None = None

    @property
    def backend(self) -> str:
        return "duckdb"

    def get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine

        import warnings
        import duckdb_engine  # noqa: F401 — registers the duckdb dialect with SQLAlchemy
        from duckdb_engine import DuckDBEngineWarning
        warnings.filterwarnings("ignore", category=DuckDBEngineWarning)
        path = self._config["path"]
        url  = f"duckdb:///{path}"
        self._engine = create_engine(url)

        logger.info("DuckDB engine created | path=%s", path)
        return self._engine


# ============================================================
# FACTORY
# ============================================================

_CONNECTORS: dict[str, type[BaseConnector]] = {
    "sqlserver":  SQLServerConnector,
    "databricks": DatabricksConnector,
    "duckdb":     DuckDBConnector,
}


def get_connector(backend: str | None = None) -> BaseConnector:
    """
    Instantiate and return a connector for the given backend.

    Args:
        backend: Optional override. If None, reads DB_BACKEND from env
                 via db_config (default: "sqlserver").

    Returns:
        Configured BaseConnector instance.
    """
    config = get_db_config(backend)

    resolved_backend = config["backend"]
    if resolved_backend not in _CONNECTORS:
        raise ValueError(
            f"Unknown backend '{resolved_backend}'. "
            f"Available: {', '.join(_CONNECTORS.keys())}"
        )

    connector_cls = _CONNECTORS[resolved_backend]
    logger.info("Initialising connector | backend=%s", resolved_backend)
    return connector_cls(config)
