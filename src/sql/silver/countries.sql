-- Bronze -> Silver: Kosovo code fix, drop unused population column, correct
-- mojibake-corrupted country names. Mirrors src/silver/transform_countries.py.
-- Reads from persist.bronze.countries -- not the raw CSV again. Read via
-- `persist` so this runs correctly whether run standalone, as part of the
-- per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.countries_quality`)
-- country_code is read as VARCHAR (bronze has leading zeros, e.g. "004"),
-- cast to INTEGER for the numeric range check.
CREATE OR REPLACE TABLE silver.countries_quality AS
SELECT
    COUNT(*)                                                          AS row_count,
    COUNT(*) FILTER (WHERE country_code IS NULL)                      AS null_country_code,
    COUNT(*) FILTER (WHERE countryiso3code IS NULL)                   AS null_countryiso3code,
    COUNT(*) FILTER (WHERE country_name IS NULL)                      AS null_country_name,
    COUNT(*) - COUNT(DISTINCT country_code)                           AS duplicate_country_codes,
    COUNT(*) FILTER (
        WHERE TRY_CAST(country_code AS INTEGER) IS NOT NULL
          AND (TRY_CAST(country_code AS INTEGER) <= 0 OR TRY_CAST(country_code AS INTEGER) >= 1000)
    )                                                                  AS invalid_country_codes,
    COUNT(*) FILTER (WHERE countryiso3code <> TRIM(countryiso3code))  AS untrimmed_countryiso3code,
    COUNT(*) FILTER (WHERE country_name <> TRIM(country_name))        AS untrimmed_country_name
FROM persist.bronze.countries;

-- 2. Transform
CREATE OR REPLACE TABLE silver.countries AS
SELECT
    -- Kosovo (XKX) has no numeric country_code in the source -- assign 990.
    -- Verified upstream: Kosovo is the only country with a null country_code.
    -- country_code is VARCHAR in bronze (leading zeros, e.g. "004").
    COALESCE(TRY_CAST(country_code AS INTEGER), 990) AS country_code,
    CASE WHEN COALESCE(TRY_CAST(country_code AS INTEGER), 990) = 990 THEN 'XKX' ELSE countryiso3code END AS countryiso3code,
    -- Standardize country names with mojibake-corrupted special characters
    CASE country_name
        WHEN 'Côte d''Ivoire'             THEN 'Cote d''Ivoire'
        WHEN 'CÃƒÂ´te d''Ivoire'          THEN 'Cote d''Ivoire'
        WHEN 'Curaçao'                    THEN 'Curacao'
        WHEN 'CuraÃƒÂ§ao'                 THEN 'Curacao'
        WHEN 'Türkiye'                    THEN 'Turkiye'
        WHEN 'TÃ¼rkiye'                   THEN 'Turkiye'
        WHEN 'TÃƒÂ¼rkiye'                 THEN 'Turkiye'
        WHEN 'São Tomé and Príncipe'      THEN 'Sao Tome and Principe'
        ELSE country_name
    END AS country_name,
    -- population dropped: always 0 in the source, unused downstream
    region, subregion, capital, currency, language
FROM persist.bronze.countries;

-- 3. Persist (DuckDB backend -- `persist` attached by run_countries.sql).
--    No CSV export: verification is done by querying this table directly.
CREATE OR REPLACE TABLE persist.silver.countries AS SELECT * FROM silver.countries;
