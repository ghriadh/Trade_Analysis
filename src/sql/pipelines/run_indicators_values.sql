-- SQL pipeline -- bronze -> silver -> gold for indicators_values. Zero Python.
-- Requires run_countries.sql and run_indicators.sql to have already been run
-- (gold.entities / gold.indicators must exist in the persisted database).
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_indicators_values.sql

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Persistent DuckDB backend -- separate database from data/trade_analysis.duckdb
-- (the real pandas-fed one) so SQL-pipeline experiments never conflict with it.
ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/indicators_values.sql
.read src/sql/silver/indicators_values.sql
.read src/sql/gold/indicators_values.sql

DETACH persist;
