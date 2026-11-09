-- SQL pipeline -- silver layer, all tables. Zero Python.
-- Requires run_bronze.sql (and optionally trade_data_correction.sql) to
-- have already been run -- each silver/*.sql file reads persist.bronze.*.
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_silver.sql

INSTALL encodings;
LOAD encodings;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/silver/products.sql
.read src/sql/silver/trade_flow.sql
.read src/sql/silver/indicators.sql
.read src/sql/silver/countries.sql
.read src/sql/silver/indicators_values.sql
.read src/sql/silver/trade_data.sql

DETACH persist;
