-- Bronze: read per-(reporter,flow) trade CSVs -- flows=M and flows=X only
-- (the "flows=?" glob matches a single character, excluding "flows=M_X").
-- The M_X batch holds reporters with a divergent Comtrade/WB code and is
-- deliberately left out here -- see src/sql/bronze/trade_data_correction.sql.
-- Files are cp1252-encoded (Comtrade API output on Windows).
--
-- Written to Parquet partitioned by (year, cmdCode, flowCode) -- content
-- columns, not path segments -- then reunited into a table.
COPY (
    SELECT * FROM read_csv_auto(
        'data/bronze/trade_data/year=*/product=*/flows=?/*.csv',
        union_by_name = true,
        encoding = 'cp1252'
    )
) TO 'data/sql/bronze/trade_data' (FORMAT PARQUET, PARTITION_BY (year, cmdCode, flowCode), OVERWRITE_OR_IGNORE true);

CREATE OR REPLACE TABLE bronze.trade_data AS
SELECT * FROM read_parquet('data/sql/bronze/trade_data/*/*/*/*.parquet', hive_partitioning = true);

-- Persist (DuckDB backend -- `persist` attached by run_trade_data.sql)
CREATE OR REPLACE TABLE persist.bronze.trade_data AS SELECT * FROM bronze.trade_data;
