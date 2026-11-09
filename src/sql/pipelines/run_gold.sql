-- SQL pipeline -- gold layer, all tables. Zero Python.
-- Requires run_silver.sql to have already been run -- each gold/*.sql file
-- reads persist.silver.*. Order matters here (unlike bronze/silver): entities
-- and indicators/products/trade_flow must be built before indicators_values
-- and trade_data, which join against their persisted gold output.
--
-- trade_data's gate is EXPECTED TO FAIL until trade_data_correction.sql has
-- been run (see run_trade_data.sql) -- everything before it in this script
-- still completes and persists normally.
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_gold.sql

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/gold/products.sql
.read src/sql/gold/trade_flow.sql
.read src/sql/gold/indicators.sql
.read src/sql/gold/entities.sql
.read src/sql/gold/indicators_values.sql
.read src/sql/gold/trade_data.sql

DETACH persist;
