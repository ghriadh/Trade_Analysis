-- Silver -> Gold: referential validation + trade_data fact table.
-- Mirrors src/gold/transform_trade_data.py.
--
-- Depends on gold.products, gold.trade_flow and gold.entities already being
-- persisted in data/sql/trade_analysis_sql.duckdb (run run_products.sql,
-- run_trade_flow.sql and run_countries.sql first) -- read from `persist`,
-- not rebuilt here. Same dependency the pandas version has on
-- GOLD_PRODUCTS_PATH / GOLD_TRADE_FLOW_PATH / GOLD_ENTITIES_PATH already
-- existing on disk.

-- 1. Product codes in trade data but absent from the gold products reference.
CREATE OR REPLACE TABLE gold.trade_data_missing_products AS
SELECT DISTINCT cmdCode
FROM persist.silver.trade_data
WHERE cmdCode NOT IN (SELECT cmdCode FROM persist.gold.products);

-- 2. Flow codes in trade data but absent from the gold trade_flow reference.
CREATE OR REPLACE TABLE gold.trade_data_missing_flows AS
SELECT DISTINCT flowCode
FROM persist.silver.trade_data
WHERE flowCode NOT IN (SELECT flow_code FROM persist.gold.trade_flow);

-- 3. Reporter/partner Comtrade codes not found in gold entities. Mirrors
-- pandas' validate_references() exactly -- but note this does NOT catch
-- WB/Comtrade divergence: entities.country_code is itself populated from
-- trade_data's own observed partnerCode/partnerISO (see gold/entities.sql's
-- entities_trade CTE), so any code appearing as a partner anywhere already
-- "counts" as known, regardless of whether it appears as a reporter. Check
-- 5 below is what actually detects unresolved divergence.
CREATE OR REPLACE TABLE gold.trade_data_missing_reporters AS
SELECT DISTINCT reporterCode
FROM persist.silver.trade_data
WHERE reporterCode NOT IN (SELECT country_code FROM persist.gold.entities WHERE country_code IS NOT NULL);

CREATE OR REPLACE TABLE gold.trade_data_missing_partners AS
SELECT DISTINCT partnerCode
FROM persist.silver.trade_data
WHERE partnerCode NOT IN (SELECT country_code FROM persist.gold.entities WHERE country_code IS NOT NULL);

-- 4. Hard-fail the script if any of the checks above found rows.
SELECT error('cmdCode values not in gold products: ' || (SELECT string_agg(cmdCode, ', ') FROM gold.trade_data_missing_products))
WHERE EXISTS (SELECT 1 FROM gold.trade_data_missing_products);

SELECT error('flowCode values not in gold trade_flow: ' || (SELECT string_agg(flowCode, ', ') FROM gold.trade_data_missing_flows))
WHERE EXISTS (SELECT 1 FROM gold.trade_data_missing_flows);

SELECT error('reporterCode values not in gold entities: ' || (SELECT string_agg(CAST(reporterCode AS VARCHAR), ', ') FROM gold.trade_data_missing_reporters))
WHERE EXISTS (SELECT 1 FROM gold.trade_data_missing_reporters);

SELECT error('partnerCode values not in gold entities: ' || (SELECT string_agg(CAST(partnerCode AS VARCHAR), ', ') FROM gold.trade_data_missing_partners))
WHERE EXISTS (SELECT 1 FROM gold.trade_data_missing_partners);

-- 5. Unresolved WB/Comtrade code divergence. wb_country_code (840, 250, ...)
-- and country_code (842, 251, ...) differ permanently for these entities --
-- that fact never changes. What DOES change is whether the correct Comtrade
-- code has any reporter-side trade data yet: before the flows=M_X
-- correction, 842/251/579/699/757 never appear as reporterCode at all
-- (the standard extraction queried Comtrade using the WB codes, which
-- Comtrade doesn't recognize as those reporters). This is additional to
-- pandas -- src/gold/transform_countries.py's detect_code_divergence() logs
-- this but never gates on it; check 3/4 above can't catch it either, for
-- the reason explained there.
CREATE OR REPLACE TABLE gold.trade_data_unresolved_divergence AS
SELECT wb_country_iso3, wb_country_name, country_code, wb_country_code
FROM persist.gold.entities
WHERE country_code IS NOT NULL
  AND wb_country_code IS NOT NULL
  AND wb_country_code NOT IN (0, 999)
  AND country_code != wb_country_code
  AND country_code NOT IN (SELECT DISTINCT reporterCode FROM persist.silver.trade_data);

SELECT error('Unresolved WB/Comtrade divergence -- correct Comtrade code has no reporter data yet: ' ||
    (SELECT string_agg(wb_country_iso3 || ' (comtrade=' || country_code || ', wb=' || wb_country_code || ')', ', ')
     FROM gold.trade_data_unresolved_divergence))
WHERE EXISTS (SELECT 1 FROM gold.trade_data_unresolved_divergence);

-- 6. Transform: join surrogate keys (entities joined twice: reporter + partner).
--    INNER JOIN is equivalent to pandas' LEFT JOIN + null-check-and-raise --
--    the gates above already guarantee every row matches.
CREATE OR REPLACE TABLE gold.trade_data AS
SELECT
    ROW_NUMBER() OVER () AS trade_key,
    t.typeCode,
    rep.entity_key  AS reporter_key,
    par.entity_key  AS partner_key,
    p.product_key,
    f.flow_key,
    t.year,
    t.primaryValue
FROM persist.silver.trade_data t
JOIN persist.gold.products    p   ON t.cmdCode    = p.cmdCode
JOIN persist.gold.trade_flow  f   ON t.flowCode   = f.flow_code
JOIN persist.gold.entities    rep ON t.reporterCode = rep.country_code
JOIN persist.gold.entities    par ON t.partnerCode  = par.country_code;

-- 6. Persist (DuckDB backend -- `persist` attached by the pipeline orchestrator).
CREATE OR REPLACE TABLE persist.gold.trade_data AS SELECT * FROM gold.trade_data;
