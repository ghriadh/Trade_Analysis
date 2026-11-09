-- SQL pipeline -- apply the trade_data WB/Comtrade divergence correction.
-- Run AFTER run_trade_data.sql has failed at the gold referential integrity
-- gate. Appends the flows=M_X batch (divergent reporters) into the already
-- written bronze Parquet dataset, rebuilds bronze.trade_data, then re-runs
-- silver and gold -- which should now pass.
--
-- Run from the project root with the DuckDB CLI:
--     duckdb :memory: < src/sql/pipelines/run_trade_data_correction.sql

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

.read src/sql/bronze/trade_data_correction.sql
.read src/sql/silver/trade_data.sql
.read src/sql/gold/trade_data.sql

DETACH persist;
