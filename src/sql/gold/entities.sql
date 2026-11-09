-- Silver -> Gold: entities dimension, merging WB countries, WB indicator
-- entities and Comtrade trade partners into one canonical reference.
-- Mirrors src/gold/transform_countries.py (build_entities + clean_entities).
--
-- Reads persist.silver.countries, persist.silver.indicators_values and
-- persist.silver.trade_data (all persisted by their own silver pipelines) --
-- read via `persist` so this runs correctly whether gold is run standalone,
-- as part of the per-table pipeline, or as part of run_gold.sql.
--
-- The pandas version also logs warnings for: reporter codes not matched to
-- WB entities, partner ISO3 not matched to WB entities, and WB/Comtrade
-- code divergence (checked against data/config/known_divergences.json).
-- None of these affect the resulting table -- they're informational for a
-- human operator -- so they're not replicated here, just noted.
--
-- entity_key is a plain ROW_NUMBER() over whatever order the joins produce,
-- same as pandas (index-based, "not guaranteed stable across re-runs" per
-- the original code's own comment) -- it is not expected to match pandas'
-- values row for row, only the underlying data is expected to match.

CREATE OR REPLACE TABLE gold.entities AS
WITH entities_ind AS (
    -- All entities appearing in indicator values (includes non-country
    -- aggregates: regions, income groups, etc.), deduplicated by ISO3.
    SELECT
        countryiso3code AS wb_country_iso3,
        ANY_VALUE(country_name) AS wb_country_name,
        ANY_VALUE(country_id) AS country_id
    FROM persist.silver.indicators_values
    GROUP BY countryiso3code
),
entities_wb AS (
    -- WB countries enriched with their indicator country_id (WB 2-letter code)
    SELECT
        c.country_code AS wb_country_code,
        c.countryiso3code AS wb_country_iso3,
        c.country_name AS wb_country_name,
        ind.country_id
    FROM persist.silver.countries c
    LEFT JOIN entities_ind ind ON c.countryiso3code = ind.wb_country_iso3
),
missing_ind AS (
    -- Indicator-only entities absent from WB countries -- assigned code 999.
    -- Column order matches entities_wb exactly: the UNION ALL below combines
    -- them with SELECT * (positional), not by column name.
    SELECT 999 AS wb_country_code, wb_country_iso3, wb_country_name, country_id
    FROM entities_ind
    WHERE wb_country_iso3 NOT IN (SELECT wb_country_iso3 FROM entities_wb WHERE wb_country_iso3 IS NOT NULL)
),
entities_base AS (
    -- WB countries + indicator-only entities, deduplicated by ISO3 (WB wins).
    -- Explicit column lists in the UNION ALL -- never rely on positional
    -- matching via SELECT * across CTEs with independently defined columns.
    SELECT wb_country_code, wb_country_iso3, wb_country_name, country_id
    FROM (
        SELECT wb_country_code, wb_country_iso3, wb_country_name, country_id, 1 AS priority FROM entities_wb
        UNION ALL
        SELECT wb_country_code, wb_country_iso3, wb_country_name, country_id, 2 AS priority FROM missing_ind
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY wb_country_iso3 ORDER BY priority) = 1
),
entities_trade AS (
    -- Comtrade partner entities (raw Comtrade identifiers)
    SELECT DISTINCT
        partnerISO AS country_iso3,
        partnerCode AS country_code,
        partnerDesc AS country_name
    FROM persist.silver.trade_data
),
entities_joined AS (
    -- Outer join: every Comtrade partner matched to a WB/indicator entity
    -- by ISO3 where possible; unmatched rows on either side are kept.
    SELECT
        t.country_iso3, t.country_code, t.country_name,
        e.wb_country_code, e.wb_country_iso3, e.wb_country_name, e.country_id
    FROM entities_trade t
    FULL OUTER JOIN entities_base e ON t.country_iso3 = e.wb_country_iso3
),
entities_world AS (
    -- "World" special case: the WB-side World row (no Comtrade code) is
    -- dropped; the Comtrade-side World row is enriched with a canonical
    -- WB identity (code 0, ISO3 WLD).
    SELECT
        country_iso3, country_code, country_name, country_id,
        CASE WHEN country_name = 'World' THEN 0       ELSE wb_country_code END AS wb_country_code,
        CASE WHEN country_name = 'World' THEN 'WLD'    ELSE wb_country_iso3 END AS wb_country_iso3,
        CASE WHEN country_name = 'World' THEN 'World'  ELSE wb_country_name END AS wb_country_name
    FROM entities_joined
    WHERE wb_country_name IS DISTINCT FROM 'World'
)
SELECT
    ROW_NUMBER() OVER () AS entity_key,
    country_id,
    country_code,
    TRIM(COALESCE(country_iso3, ''))    AS country_iso3,
    TRIM(COALESCE(country_name, ''))    AS country_name,
    wb_country_code,
    TRIM(COALESCE(wb_country_iso3, '')) AS wb_country_iso3,
    TRIM(COALESCE(wb_country_name, '')) AS wb_country_name,
    -- NULL wb_country_code mirrors pandas' NaN behavior here: `NaN not in
    -- {0, 999}` is True in Python, so unmatched entities are also COUNTRY.
    CASE WHEN wb_country_code IS NULL OR wb_country_code NOT IN (0, 999)
         THEN 'COUNTRY' ELSE 'ENTITY' END AS entity_type
FROM entities_world;

-- Persist (DuckDB backend -- `persist` attached by run_countries.sql).
CREATE OR REPLACE TABLE persist.gold.entities AS SELECT * FROM gold.entities;
