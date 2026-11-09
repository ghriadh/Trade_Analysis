-- Bronze -> Silver: restore Namibia's country_id, synthesize ISO3 codes for
-- non-country aggregates, cast types. Mirrors src/silver/transform_indicators_values.py.
-- Reads from persist.bronze.indicators_values -- not the raw files again.
-- Read via `persist` so this runs correctly whether run standalone, as part
-- of the per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.indicators_values_quality`)
CREATE OR REPLACE TABLE silver.indicators_values_quality AS
SELECT
    COUNT(*)                                                              AS row_count,
    COUNT(*) FILTER (WHERE countryiso3code IS NULL)                       AS null_countryiso3code,
    COUNT(*) FILTER (WHERE date IS NULL)                                  AS null_date,
    COUNT(*) FILTER (WHERE value IS NULL)                                 AS null_value,
    COUNT(*) FILTER (WHERE country_id IS NULL)                            AS null_country_id,
    COUNT(*) FILTER (WHERE country_name IS NULL)                          AS null_country_name,
    COUNT(*) FILTER (WHERE indicator_code IS NULL)                        AS null_indicator_code,
    COUNT(*) FILTER (WHERE indicator_name IS NULL)                        AS null_indicator_name,
    COUNT(*) FILTER (WHERE TRY_CAST(date AS INTEGER) < 1950 OR TRY_CAST(date AS INTEGER) > 2025) AS invalid_years,
    COUNT(*) FILTER (WHERE value = 0)                                     AS zero_values,
    COUNT(*) FILTER (WHERE countryiso3code IS NOT NULL AND countryiso3code <> TRIM(countryiso3code)) AS untrimmed_countryiso3code,
    COUNT(*) FILTER (WHERE country_id IS NOT NULL AND country_id <> TRIM(country_id))                 AS untrimmed_country_id,
    COUNT(*) FILTER (WHERE country_name IS NOT NULL AND country_name <> TRIM(country_name))           AS untrimmed_country_name,
    COUNT(*) FILTER (WHERE indicator_code IS NOT NULL AND indicator_code <> TRIM(indicator_code))     AS untrimmed_indicator_code,
    COUNT(*) FILTER (WHERE indicator_name IS NOT NULL AND indicator_name <> TRIM(indicator_name))     AS untrimmed_indicator_name
FROM persist.bronze.indicators_values;

-- 2. Transform
CREATE OR REPLACE TABLE silver.indicators_values AS
SELECT
    -- Non-country aggregates (regions, income groups, etc.) have no ISO3 code
    -- in the World Bank API -- synthesize one as "X" + country_id.
    COALESCE(countryiso3code, 'X' || COALESCE(country_id, 'NA')) AS countryiso3code,
    CAST(date AS INTEGER)  AS date,
    CAST(value AS DOUBLE)  AS value,
    -- Namibia's country_id ("NA") reads back correctly in DuckDB (unlike
    -- pandas, which treats the string "NA" as null on read) -- this only
    -- fires for genuinely missing values, kept for parity with pandas.
    COALESCE(country_id, 'NA') AS country_id,
    country_name,
    indicator_code,
    indicator_name
FROM persist.bronze.indicators_values;

-- 3. Persist (DuckDB backend -- `persist` attached by run_indicators_values.sql).
--    No CSV/Parquet export: verification is done by querying this table directly.
CREATE OR REPLACE TABLE persist.silver.indicators_values AS SELECT * FROM silver.indicators_values;
