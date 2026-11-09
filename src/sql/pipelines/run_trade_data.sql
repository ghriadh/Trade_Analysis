-- SQL pipeline -- bronze -> silver -> gold for trade_data. Zero Python.
-- Requires run_products.sql, run_trade_flow.sql and run_countries.sql to
-- have already been run (gold.products / gold.trade_flow / gold.entities
-- must exist in the persisted database).
--
-- bronze/trade_data.sql deliberately excludes the flows=M_X batch (reporters
-- with a divergent Comtrade/WB code) -- this run is EXPECTED TO FAIL at the
-- gold referential integrity gate, demonstrating the divergence detection.
-- To fix it, run: duckdb :memory: < src/sql/pipelines/run_trade_data_correction.sql
--
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_trade_data.sql

-- cp1252 support (trade_data CSVs are cp1252-encoded) -- not loaded by
-- default in the DuckDB CLI, unlike the Python API.
INSTALL encodings;
LOAD encodings;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/trade_data.sql
.read src/sql/silver/trade_data.sql
.read src/sql/gold/trade_data.sql

DETACH persist;
