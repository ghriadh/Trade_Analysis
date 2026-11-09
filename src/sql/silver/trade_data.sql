-- Bronze -> Silver: dedup on natural key, type casting, mojibake name fixes.
-- Mirrors src/silver/transform_trade_data.py.
-- Reads from persist.bronze.trade_data (written by bronze/trade_data.sql,
-- optionally extended by trade_data_correction.sql) -- not the raw CSVs
-- again. Read via `persist` so this runs correctly whether run standalone,
-- as part of the per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.trade_data_quality`)
-- partnerISO is intentionally not checked for spaces -- Comtrade uses padded
-- codes (_X, XX, etc.) for aggregated partner groups (matches pandas).
CREATE OR REPLACE TABLE silver.trade_data_quality AS
SELECT
    COUNT(*)                                                                AS row_count,
    COUNT(*) FILTER (WHERE reporterCode IS NULL)                            AS null_reporter_code,
    COUNT(*) FILTER (WHERE partnerCode IS NULL)                             AS null_partner_code,
    COUNT(*) FILTER (WHERE year IS NULL)                                    AS null_year,
    COUNT(*) FILTER (WHERE flowCode IS NULL)                                AS null_flow_code,
    COUNT(*) FILTER (WHERE cmdCode IS NULL)                                 AS null_cmd_code,
    COUNT(*) FILTER (WHERE primaryValue IS NULL)                            AS null_primary_value,
    COUNT(*) FILTER (WHERE primaryValue < 0)                                AS negative_primary_value,
    COUNT(*) FILTER (WHERE primaryValue = 0)                                AS zero_primary_value,
    COUNT(*) FILTER (WHERE year < 1950 OR year > 2026)                      AS invalid_years,
    COUNT(*) FILTER (WHERE cmdCode IS NOT NULL AND cmdCode <> TRIM(cmdCode))             AS untrimmed_cmd_code,
    COUNT(*) FILTER (WHERE flowCode IS NOT NULL AND flowCode <> TRIM(flowCode))          AS untrimmed_flow_code,
    COUNT(*) FILTER (WHERE reporterISO IS NOT NULL AND reporterISO <> TRIM(reporterISO)) AS untrimmed_reporter_iso,
    COUNT(*) FILTER (WHERE reporterDesc IS NOT NULL AND reporterDesc <> TRIM(reporterDesc)) AS untrimmed_reporter_desc,
    COUNT(*) FILTER (WHERE partnerDesc IS NOT NULL AND partnerDesc <> TRIM(partnerDesc)) AS untrimmed_partner_desc,
    (SELECT COUNT(*) FROM (
        SELECT reporterCode, partnerCode, year, flowCode, cmdCode, COUNT(*) AS n
        FROM persist.bronze.trade_data
        GROUP BY reporterCode, partnerCode, year, flowCode, cmdCode
        HAVING COUNT(*) > 1
    )) AS duplicate_natural_keys
FROM persist.bronze.trade_data;

-- 2. Transform
CREATE OR REPLACE TABLE silver.trade_data AS
SELECT
    typeCode,
    reporterCode, reporterISO,
    -- Includes two extra single-mojibake variants (Côte d'Ivoire, Curaçao)
    -- and Saint Barthélemy, absent from pandas' own NAME_CORRECTIONS: pandas
    -- only ever sees the DOUBLE-mojibake form of these names, because its
    -- consolidate() step writes the consolidated CSV as UTF-8 then re-reads
    -- it forcing cp1252 -- corrupting the text a second time. Reading the
    -- per-reporter files directly (as done here) skips that bug entirely,
    -- so the single-mojibake form needs its own fix.
    CASE reporterDesc
        WHEN 'Côte d''Ivoire'             THEN 'Cote d''Ivoire'
        WHEN 'CÃ´te d''Ivoire'            THEN 'Cote d''Ivoire'
        WHEN 'CÃƒÂ´te d''Ivoire'          THEN 'Cote d''Ivoire'
        WHEN 'Curaçao'                    THEN 'Curacao'
        WHEN 'CuraÃ§ao'                   THEN 'Curacao'
        WHEN 'CuraÃƒÂ§ao'                 THEN 'Curacao'
        WHEN 'Türkiye'                    THEN 'Turkiye'
        WHEN 'TÃ¼rkiye'                   THEN 'Turkiye'
        WHEN 'TÃƒÂ¼rkiye'                 THEN 'Turkiye'
        WHEN 'São Tomé and Príncipe'      THEN 'Sao Tome and Principe'
        WHEN 'Saint Barthélemy'          THEN 'Saint Barthelemy'
        WHEN 'Saint BarthÃ©lemy'         THEN 'Saint Barthelemy'
        ELSE reporterDesc
    END AS reporterDesc,
    flowCode,
    partnerCode, partnerISO,
    CASE partnerDesc
        WHEN 'Côte d''Ivoire'             THEN 'Cote d''Ivoire'
        WHEN 'CÃ´te d''Ivoire'            THEN 'Cote d''Ivoire'
        WHEN 'CÃƒÂ´te d''Ivoire'          THEN 'Cote d''Ivoire'
        WHEN 'Curaçao'                    THEN 'Curacao'
        WHEN 'CuraÃ§ao'                   THEN 'Curacao'
        WHEN 'CuraÃƒÂ§ao'                 THEN 'Curacao'
        WHEN 'Türkiye'                    THEN 'Turkiye'
        WHEN 'TÃ¼rkiye'                   THEN 'Turkiye'
        WHEN 'TÃƒÂ¼rkiye'                 THEN 'Turkiye'
        WHEN 'São Tomé and Príncipe'      THEN 'Sao Tome and Principe'
        WHEN 'Saint Barthélemy'          THEN 'Saint Barthelemy'
        WHEN 'Saint BarthÃ©lemy'         THEN 'Saint Barthelemy'
        ELSE partnerDesc
    END AS partnerDesc,
    cmdCode,
    cifvalue, fobvalue,
    CAST(primaryValue AS DOUBLE)  AS primaryValue,
    CAST(year AS INTEGER)         AS year
FROM persist.bronze.trade_data
-- Dedup on natural key -- overlapping extraction batches can produce
-- duplicate rows. "First" here is arbitrary row order, same as pandas'
-- own drop_duplicates(keep="first") on a concatenation whose row order
-- depends on file-read order -- not a meaningful business rule either way.
QUALIFY ROW_NUMBER() OVER (PARTITION BY reporterCode, partnerCode, year, flowCode, cmdCode) = 1;

-- 3. Persist (DuckDB backend -- `persist` attached by the pipeline orchestrator).
--    No CSV export: verification is done by querying this table directly.
CREATE OR REPLACE TABLE persist.silver.trade_data AS SELECT * FROM silver.trade_data;
