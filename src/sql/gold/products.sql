-- Silver -> Gold: referential validation + products dimension with surrogate key.
-- Reads from persist.silver.products and persist.silver.trade_data (both
-- persisted by their own silver pipelines) -- read via `persist` so this
-- runs correctly whether gold is run standalone, as part of the per-table
-- pipeline, or as part of run_gold.sql.

-- 1. Codes referenced by trade data but absent from products.
--    TOTAL is a valid aggregation code, handled separately below -- not a
--    referential integrity failure.
CREATE OR REPLACE TABLE gold.products_missing_codes AS
SELECT DISTINCT t.cmdCode
FROM persist.silver.trade_data t
WHERE t.cmdCode NOT IN (SELECT chapter FROM persist.silver.products)
  AND t.cmdCode != 'TOTAL';

-- 2. Hard-fail the script if any unexpected missing codes exist.
-- error() only evaluates for rows the WHERE clause lets through, so this
-- is a no-op (zero rows) when products_missing_codes is empty.
SELECT error('Unexpected missing product codes in trade data: ' || (SELECT string_agg(cmdCode, ', ') FROM gold.products_missing_codes))
WHERE EXISTS (SELECT 1 FROM gold.products_missing_codes);

-- 3. Transform
CREATE OR REPLACE TABLE gold.products AS
WITH trade_codes AS (
    SELECT DISTINCT cmdCode FROM persist.silver.trade_data
),
products_with_total AS (
    SELECT chapter, description, sector FROM persist.silver.products
    UNION ALL
    SELECT 'TOTAL', 'AGGREGATION ALL PRODUCTS', 'AGGREGATION ALL SECTORS'
    WHERE 'TOTAL' NOT IN (SELECT chapter FROM persist.silver.products)
      AND 'TOTAL' IN (SELECT cmdCode FROM trade_codes)
)
SELECT
    ROW_NUMBER() OVER (ORDER BY chapter) AS product_key,
    chapter     AS cmdCode,
    description AS product_description,
    sector
FROM products_with_total
ORDER BY chapter;

-- 4. Persist (DuckDB backend -- `persist` attached by run_products.sql).
--    No CSV export: verification is done by querying this table directly
--    (e.g. `SELECT * FROM gold.products` in trade_analysis_sql.duckdb).
CREATE OR REPLACE TABLE persist.gold.products AS SELECT * FROM gold.products;
