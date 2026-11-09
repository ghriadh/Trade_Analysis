-- Bronze: read the 16 per-indicator checkpoint CSVs individually and write
-- them out as Parquet partitioned by indicator_code (one file per
-- indicator) -- then reunite the partitioned files into a single table.
-- Preserves the same per-indicator granularity as the extraction's own
-- checkpoint mechanism (src/bronze/extract_indicators_values.py), so a
-- future new indicator could be added without reprocessing the others.

-- 1. CSV (one per indicator) -> partitioned Parquet
COPY (
    SELECT * FROM read_csv_auto('data/bronze/indicators_values/.checkpoints/*.csv', union_by_name = true)
) TO 'data/sql/bronze/indicators_values' (FORMAT PARQUET, PARTITION_BY (indicator_code), OVERWRITE_OR_IGNORE true);

-- 2. Partitioned Parquet -> table
CREATE OR REPLACE TABLE bronze.indicators_values AS
SELECT * FROM read_parquet('data/sql/bronze/indicators_values/*/*.parquet', hive_partitioning = true);

-- 3. Persist (DuckDB backend -- `persist` attached by run_indicators_values.sql)
CREATE OR REPLACE TABLE persist.bronze.indicators_values AS SELECT * FROM bronze.indicators_values;
