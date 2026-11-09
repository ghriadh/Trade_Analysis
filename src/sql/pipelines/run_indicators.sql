-- SQL pipeline -- bronze -> silver -> gold for indicators. Zero Python.
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_indicators.sql

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Persistent DuckDB backend -- separate database from data/trade_analysis.duckdb
-- (the real pandas-fed one) so SQL-pipeline experiments never conflict with it.
ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/indicators.sql
.read src/sql/silver/indicators.sql
.read src/sql/gold/indicators.sql

DETACH persist;
