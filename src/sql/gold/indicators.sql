-- Silver -> Gold: referential validation + indicators dimension with surrogate key.
-- Reads from persist.silver.indicators and persist.silver.indicators_values
-- (both persisted by their own silver pipelines) -- read via `persist` so
-- this runs correctly whether gold is run standalone, as part of the
-- per-table pipeline, or as part of run_gold.sql.

-- 1. Indicator codes/names referenced by values but absent from the reference.
--    Two independent checks, matching pandas (both code and name are validated).
CREATE OR REPLACE TABLE gold.indicators_missing_codes AS
SELECT DISTINCT v.indicator_code
FROM persist.silver.indicators_values v
WHERE v.indicator_code NOT IN (SELECT indicator_code FROM persist.silver.indicators);

CREATE OR REPLACE TABLE gold.indicators_missing_names AS
SELECT DISTINCT v.indicator_name
FROM persist.silver.indicators_values v
WHERE v.indicator_name NOT IN (SELECT indicator_name FROM persist.silver.indicators);

-- 2. Self-consistency: the reference itself must not have duplicate codes.
CREATE OR REPLACE TABLE gold.indicators_duplicate_codes AS
SELECT indicator_code
FROM persist.silver.indicators
GROUP BY indicator_code
HAVING COUNT(*) > 1;

-- 3. Hard-fail the script if any of the three checks above found rows.
SELECT error('Indicator codes in values not found in reference: ' || (SELECT string_agg(indicator_code, ', ') FROM gold.indicators_missing_codes))
WHERE EXISTS (SELECT 1 FROM gold.indicators_missing_codes);

SELECT error('Indicator names in values not found in reference: ' || (SELECT string_agg(indicator_name, ', ') FROM gold.indicators_missing_names))
WHERE EXISTS (SELECT 1 FROM gold.indicators_missing_names);

SELECT error('Duplicate indicator_code found in indicators reference: ' || (SELECT string_agg(indicator_code, ', ') FROM gold.indicators_duplicate_codes))
WHERE EXISTS (SELECT 1 FROM gold.indicators_duplicate_codes);

-- 4. Transform. Sorted by indicator_code before key assignment, matching pandas.
CREATE OR REPLACE TABLE gold.indicators AS
SELECT
    ROW_NUMBER() OVER (ORDER BY indicator_code) AS indicator_key,
    indicator_code,
    indicator_name,
    indicator_description,
    indicator_unit
FROM persist.silver.indicators
ORDER BY indicator_code;

-- 5. Persist (DuckDB backend -- `persist` attached by run_indicators.sql).
CREATE OR REPLACE TABLE persist.gold.indicators AS SELECT * FROM gold.indicators;
