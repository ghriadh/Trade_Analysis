-- Bronze: load the raw countries reference CSV into a table.
CREATE OR REPLACE TABLE bronze.countries AS
SELECT * FROM read_csv_auto('data/bronze/countries/countries.csv', header = true);

-- Persist (DuckDB backend -- `persist` attached by run_countries.sql)
CREATE OR REPLACE TABLE persist.bronze.countries AS SELECT * FROM bronze.countries;
