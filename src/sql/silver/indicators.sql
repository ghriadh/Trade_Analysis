-- Bronze -> Silver: quality check only. No transform is applied -- silver
-- indicators is a passthrough of bronze (matches src/silver/transform_indicators.py).
-- Reads from persist.bronze.indicators -- not the raw CSV again. Read via
-- `persist` so this runs correctly whether run standalone, as part of the
-- per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.indicators_quality`)
CREATE OR REPLACE TABLE silver.indicators_quality AS
SELECT
    COUNT(*)                                                                    AS row_count,
    COUNT(*) FILTER (WHERE indicator_code IS NULL)                              AS null_indicator_code,
    COUNT(*) FILTER (WHERE indicator_name IS NULL)                              AS null_indicator_name,
    COUNT(*) FILTER (WHERE indicator_description IS NULL)                       AS null_indicator_description,
    COUNT(*) FILTER (WHERE indicator_unit IS NULL)                              AS null_indicator_unit,
    COUNT(*) - COUNT(DISTINCT indicator_code)                                   AS duplicate_indicator_codes,
    COUNT(*) FILTER (WHERE indicator_code <> TRIM(indicator_code))              AS untrimmed_indicator_code,
    COUNT(*) FILTER (WHERE indicator_name <> TRIM(indicator_name))              AS untrimmed_indicator_name,
    COUNT(*) FILTER (WHERE indicator_description <> TRIM(indicator_description)) AS untrimmed_indicator_description,
    COUNT(*) FILTER (WHERE indicator_unit <> TRIM(indicator_unit))              AS untrimmed_indicator_unit
FROM persist.bronze.indicators;

-- 2. Transform (passthrough)
CREATE OR REPLACE TABLE silver.indicators AS
SELECT indicator_code, indicator_name, indicator_description, indicator_unit
FROM persist.bronze.indicators;

-- 3. Persist (DuckDB backend -- `persist` attached by run_indicators.sql).
--    No CSV export: verification is done by querying this table directly.
CREATE OR REPLACE TABLE persist.silver.indicators AS SELECT * FROM silver.indicators;
