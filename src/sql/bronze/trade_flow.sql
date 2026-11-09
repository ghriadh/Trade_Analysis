-- Bronze: load the raw trade flow reference CSV into a table.
CREATE OR REPLACE TABLE bronze.trade_flow AS
SELECT * FROM read_csv_auto('data/bronze/trade_flow/trade_flow.csv', header = true);

-- Persist (DuckDB backend -- `persist` attached by run_trade_flow.sql)
CREATE OR REPLACE TABLE persist.bronze.trade_flow AS SELECT * FROM bronze.trade_flow;
