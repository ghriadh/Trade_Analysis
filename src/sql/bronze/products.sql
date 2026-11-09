-- Bronze: load the raw products CSV into a table.
-- Chapter is forced to VARCHAR here (bronze CSV stores it unpadded, e.g. "1"),
-- so leading zeros can be safely applied downstream in silver.
CREATE OR REPLACE TABLE bronze.products AS
SELECT * FROM read_csv_auto('data/bronze/products/products.csv', header = true, types = {'chapter': 'VARCHAR'});

-- Persist (DuckDB backend -- `persist` attached by run_products.sql)
CREATE OR REPLACE TABLE persist.bronze.products AS SELECT * FROM bronze.products;
