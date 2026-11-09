-- Bronze: load the raw indicators reference CSV into a table.
CREATE OR REPLACE TABLE bronze.indicators AS
SELECT * FROM read_csv_auto('data/bronze/indicators/indicators.csv', header = true);

-- Persist (DuckDB backend -- `persist` attached by run_indicators.sql)
CREATE OR REPLACE TABLE persist.bronze.indicators AS SELECT * FROM bronze.indicators;
