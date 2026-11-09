-- Silver -> Gold: referential validation + trade flow dimension with surrogate key.
-- Reads from persist.silver.trade_flow and persist.silver.trade_data (both
-- persisted by their own silver pipelines) -- read via `persist` so this
-- runs correctly whether gold is run standalone, as part of the per-table
-- pipeline, or as part of run_gold.sql.

-- 1. Flow codes referenced by trade data but absent from the reference.
--    Unlike products, there's no exempt code here -- any mismatch fails.
CREATE OR REPLACE TABLE gold.trade_flow_missing_codes AS
SELECT DISTINCT t.flowCode
FROM persist.silver.trade_data t
WHERE t.flowCode NOT IN (SELECT flow_code FROM persist.silver.trade_flow);

-- 2. Hard-fail the script if any missing codes exist.
SELECT error('Flow codes in trade data not found in reference: ' || (SELECT string_agg(flowCode, ', ') FROM gold.trade_flow_missing_codes))
WHERE EXISTS (SELECT 1 FROM gold.trade_flow_missing_codes);

-- 3. Transform. Key order follows the reference's own row order (matches
--    pandas, which assigns flow_key without sorting) -- not alphabetical.
CREATE OR REPLACE TABLE gold.trade_flow AS
SELECT
    ROW_NUMBER() OVER () AS flow_key,
    flow_code,
    flow_description
FROM persist.silver.trade_flow;

-- 4. Persist (DuckDB backend -- `persist` attached by run_trade_flow.sql).
CREATE OR REPLACE TABLE persist.gold.trade_flow AS SELECT * FROM gold.trade_flow;
