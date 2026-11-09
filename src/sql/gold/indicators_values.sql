-- Silver -> Gold: referential validation + indicators_values fact table.
-- Mirrors src/gold/transform_indicators_values.py.
--
-- Depends on gold.entities and gold.indicators already being persisted in
-- data/sql/trade_analysis_sql.duckdb (run run_countries.sql and
-- run_indicators.sql first) -- read from `persist`, not rebuilt here. Same
-- dependency the pandas version has on GOLD_ENTITIES_PATH / GOLD_INDICATORS_PATH
-- already existing on disk.

-- 1. Indicator codes in values but absent from the gold indicators reference.
CREATE OR REPLACE TABLE gold.indicators_values_missing_indicators AS
SELECT DISTINCT indicator_code
FROM persist.silver.indicators_values
WHERE indicator_code NOT IN (SELECT indicator_code FROM persist.gold.indicators);

-- 2. Country ISO3 codes in values but absent from the gold entities reference.
CREATE OR REPLACE TABLE gold.indicators_values_missing_entities AS
SELECT DISTINCT countryiso3code
FROM persist.silver.indicators_values
WHERE countryiso3code NOT IN (SELECT wb_country_iso3 FROM persist.gold.entities);

-- 3. Hard-fail the script if either check found rows.
SELECT error('Missing indicator_code(s) in gold indicators: ' || (SELECT string_agg(indicator_code, ', ') FROM gold.indicators_values_missing_indicators))
WHERE EXISTS (SELECT 1 FROM gold.indicators_values_missing_indicators);

SELECT error('Missing countryiso3code(s) in gold entities: ' || (SELECT string_agg(countryiso3code, ', ') FROM gold.indicators_values_missing_entities))
WHERE EXISTS (SELECT 1 FROM gold.indicators_values_missing_entities);

-- 4. Transform: join surrogate keys, sort, assign fact key.
--    INNER JOIN is equivalent to pandas' LEFT JOIN + null-check-and-raise
--    here -- the two gates above already guarantee every row matches.
CREATE OR REPLACE TABLE gold.indicators_values AS
SELECT
    ROW_NUMBER() OVER (ORDER BY ind.indicator_key, ent.entity_key, v.date) AS indicator_value_key,
    ind.indicator_key,
    ent.entity_key,
    v.date AS year,
    v.value
FROM persist.silver.indicators_values v
JOIN persist.gold.entities   ent ON v.countryiso3code = ent.wb_country_iso3
JOIN persist.gold.indicators ind ON v.indicator_code  = ind.indicator_code;

-- 5. Persist (DuckDB backend -- `persist` attached by run_indicators_values.sql).
CREATE OR REPLACE TABLE persist.gold.indicators_values AS SELECT * FROM gold.indicators_values;
