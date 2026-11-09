-- Bronze -> Silver: quality check + zero-pad HS chapter codes to 3 characters.
-- Reads from persist.bronze.products -- bronze.products.sql's persisted
-- output, not the raw CSV again. Read via `persist` (not a local in-memory
-- table) so this runs correctly whether silver is run standalone, as part
-- of the per-table pipeline, or as part of run_silver.sql.

-- 1. Quality metrics (informational -- inspect with `SELECT * FROM silver.products_quality`)
CREATE OR REPLACE TABLE silver.products_quality AS
SELECT
    COUNT(*)                                                     AS row_count,
    COUNT(*) FILTER (WHERE chapter IS NULL)                      AS null_chapter,
    COUNT(*) FILTER (WHERE description IS NULL)                  AS null_description,
    COUNT(*) FILTER (WHERE sector IS NULL)                       AS null_sector,
    COUNT(*) - COUNT(DISTINCT chapter)                           AS duplicate_chapters,
    COUNT(*) FILTER (
        WHERE TRY_CAST(chapter AS INTEGER) IS NULL
           OR TRY_CAST(chapter AS INTEGER) <= 0
           OR TRY_CAST(chapter AS INTEGER) >= 100
    )                                                             AS invalid_chapters,
    COUNT(*) FILTER (WHERE description <> TRIM(description))     AS untrimmed_description,
    COUNT(*) FILTER (WHERE sector <> TRIM(sector))                AS untrimmed_sector
FROM persist.bronze.products;

-- 2. Transform
CREATE OR REPLACE TABLE silver.products AS
SELECT
    LPAD(CAST(chapter AS VARCHAR), 3, '0') AS chapter,
    description,
    sector
FROM persist.bronze.products;

-- 3. Persist (DuckDB backend -- `persist` attached by run_products.sql).
--    No CSV export: verification is done by querying this table directly
--    (e.g. `SELECT * FROM silver.products` in trade_analysis_sql.duckdb).
CREATE OR REPLACE TABLE persist.silver.products AS SELECT * FROM silver.products;
