-- SQL pipeline -- bronze -> silver -> gold for trade_flow. Zero Python.
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_trade_flow.sql

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Persistent DuckDB backend -- separate database from data/trade_analysis.duckdb
-- (the real pandas-fed one) so SQL-pipeline experiments never conflict with it.
ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/trade_flow.sql
.read src/sql/silver/trade_flow.sql
.read src/sql/gold/trade_flow.sql

DETACH persist;
