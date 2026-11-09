-- Bronze -> Silver: quality check only. No transform is applied -- silver
-- trade_flow is a passthrough of bronze (matches src/silver/transform_trade_flow.py).
-- Reads from persist.bronze.trade_flow -- not the raw CSV again. Read via
-- `persist` so this runs correctly whether run standalone, as part of the
-- per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.trade_flow_quality`)
CREATE OR REPLACE TABLE silver.trade_flow_quality AS
SELECT
    COUNT(*)                                                          AS row_count,
    COUNT(*) FILTER (WHERE flow_code IS NULL)                         AS null_flow_code,
    COUNT(*) FILTER (WHERE flow_description IS NULL)                  AS null_flow_description,
    COUNT(*) - COUNT(DISTINCT flow_code)                              AS duplicate_flow_codes,
    COUNT(*) FILTER (WHERE flow_description <> TRIM(flow_description)) AS untrimmed_flow_description
FROM persist.bronze.trade_flow;

-- 2. Transform (passthrough)
CREATE OR REPLACE TABLE silver.trade_flow AS
SELECT flow_code, flow_description
FROM persist.bronze.trade_flow;

-- 3. Persist (DuckDB backend -- `persist` attached by run_trade_flow.sql).
--    No CSV export: verification is done by querying this table directly.
CREATE OR REPLACE TABLE persist.silver.trade_flow AS SELECT * FROM silver.trade_flow;
