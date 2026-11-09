-- Bronze correction: the flows=M_X batch holds reporters whose Comtrade
-- reporterCode diverges from their WB country_code (extracted separately via
-- `python -m src.bronze.extract_trade_data --reporters <codes> --flows M X`).
-- gold/trade_data.sql's referential integrity gates detect this divergence
-- (reporterCode/partnerCode not found in gold.entities) when only
-- src/sql/bronze/trade_data.sql (flows=M and flows=X) has been run.
--
-- This script appends the M_X batch into the existing partitioned Parquet
-- dataset (APPEND true -- does not touch the files already written by
-- trade_data.sql) and rebuilds bronze.trade_data from the now-complete set.
-- Run this, then re-run silver/trade_data.sql and gold/trade_data.sql --
-- see src/sql/pipelines/run_trade_data_correction.sql.
COPY (
    SELECT * FROM read_csv_auto(
        'data/bronze/trade_data/year=*/product=*/flows=M_X/*.csv',
        union_by_name = true,
        encoding = 'cp1252'
    )
) TO 'data/sql/bronze/trade_data' (FORMAT PARQUET, PARTITION_BY (year, cmdCode, flowCode), APPEND true);

CREATE OR REPLACE TABLE bronze.trade_data AS
SELECT * FROM read_parquet('data/sql/bronze/trade_data/*/*/*/*.parquet', hive_partitioning = true);

-- Persist (DuckDB backend -- `persist` attached by run_trade_data_correction.sql)
CREATE OR REPLACE TABLE persist.bronze.trade_data AS SELECT * FROM bronze.trade_data;
