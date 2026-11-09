-- SQL pipeline -- bronze -> silver -> gold for products (Option B: zero Python).
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_products.sql
-- .read is a DuckDB CLI shell command, not standard SQL -- this file only
-- runs through the `duckdb` binary, not the Python API.

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Persistent DuckDB backend -- mirrors the pandas pipeline's
-- `if connector: save_to_db(...)`, done once here rather than per layer
-- file. Separate database from data/trade_analysis.duckdb (the real
-- pandas-fed one) so SQL-pipeline experiments never conflict with it.
ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/products.sql
.read src/sql/silver/products.sql
.read src/sql/gold/products.sql

DETACH persist;
