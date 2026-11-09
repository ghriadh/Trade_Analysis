-- SQL pipeline -- bronze layer, all tables. Zero Python.
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_bronze.sql
--
-- Note: trade_data here excludes the flows=M_X batch (WB/Comtrade divergent
-- reporters) -- run trade_data_correction.sql separately afterward, same as
-- the manual `--reporters` correction step in the pandas pipeline.

INSTALL encodings;
LOAD encodings;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

ATTACH 'data/sql/trade_analysis_sql.duckdb' AS persist;
CREATE SCHEMA IF NOT EXISTS persist.bronze;
CREATE SCHEMA IF NOT EXISTS persist.silver;
CREATE SCHEMA IF NOT EXISTS persist.gold;

.read src/sql/bronze/products.sql
.read src/sql/bronze/trade_flow.sql
.read src/sql/bronze/indicators.sql
.read src/sql/bronze/countries.sql
.read src/sql/bronze/indicators_values.sql
.read src/sql/bronze/trade_data.sql

DETACH persist;
