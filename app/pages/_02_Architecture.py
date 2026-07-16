"""
Page 2 — Architecture & Infrastructure  (v4)
Updated with new styles_v3 palette.
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from utils.styles import (
    inject_styles, kpi, section_label, thin_divider, finding
)

inject_styles()


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">ETL · Page 2 of 2</div>', unsafe_allow_html=True)
st.title("Architecture & Infrastructure")
st.caption(
    "Medallion pipeline design · multi-backend abstraction · "
    "star schema · 187 countries with data · 316 gold entities"
)


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.6rem;'>", unsafe_allow_html=True)
cols = st.columns(4)
for col, (label, value, color) in zip(cols, [
    ("Layers",        "3",   "teal"),
    ("Backends",      "3",   "blue"),
    ("Gold tables",   "6",   ""),
    ("Gold entities", "316", "amber"),
]):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MEDALLION ARCHITECTURE
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.9rem;'>", unsafe_allow_html=True)
section_label("Medallion Architecture")

col1, arr1, col2, arr2, col3 = st.columns([10, 1, 10, 1, 10])

with col1:
    st.markdown("""
    <div class="med-card med-bronze">
      <div class="med-title">🟡 Bronze — Raw</div>
      <div class="med-files">
        countries.csv<br>
        products.csv<br>
        trade_flow.csv<br>
        indicators.csv<br>
        indicators_values.csv<br>
        trade_data/<span class="med-tag"> partitioned</span>
      </div>
      <div class="med-footer">
        250 source countries<br>
        ~5,000 Comtrade API calls<br>
        Comtrade · World Bank · GitHub
      </div>
    </div>
    """, unsafe_allow_html=True)

with arr1:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;font-size:22px;color:#304870;padding-top:60px;'>→</div>",
        unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="med-card med-silver">
      <div class="med-title">🔵 Silver — Cleaned</div>
      <div class="med-files">
        countries.csv<br>
        products.csv<br>
        trade_flow.csv<br>
        indicators.csv<br>
        indicators_values.csv<br>
        trade_data.csv
      </div>
      <div class="med-footer">
        Null checks · deduplication<br>
        Type casting · validation<br>
        6 transform modules
      </div>
    </div>
    """, unsafe_allow_html=True)

with arr2:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;font-size:22px;color:#304870;padding-top:60px;'>→</div>",
        unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="med-card med-gold">
      <div class="med-title">🟢 Gold — Star Schema</div>
      <div class="med-files">
        entities.csv<span class="med-tag"> dim</span><br>
        products.csv<span class="med-tag"> dim</span><br>
        trade_flow.csv<span class="med-tag"> dim</span><br>
        indicators.csv<span class="med-tag"> dim</span><br>
        indicators_values.csv<span class="med-tag"> fact</span><br>
        trade_data.csv<span class="med-tag"> fact</span>
      </div>
      <div class="med-footer">
        316 entities (250 countries + aggregates)<br>
        Surrogate keys · referential integrity<br>
        Divergence detection · dependency check
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DATABASE BACKEND
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.75rem;'>", unsafe_allow_html=True)
section_label("Database Backend")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:6px;">

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">SQL Server</span>
        <span class="backend-desc">
          Local relational DB · Windows Auth (no password) ·
          ODBC driver · <code style="font-size:12px;background:#253B5C;color:#93C5FD;
          padding:1px 5px;border-radius:3px;">fast_executemany</code> for bulk inserts
        </span>
        <span class="badge badge-active">active</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-partial"></div>
        <span class="backend-name">Databricks</span>
        <span class="backend-desc">
          Cloud data lakehouse · Unity Catalog · token auth ·
          SQL warehouse · planned for production deployment
        </span>
        <span class="badge badge-planned">planned</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-pending"></div>
        <span class="backend-name">DuckDB</span>
        <span class="backend-desc">
          Embedded file-based DB · zero config · single file ·
          ideal for local testing and portfolio demo
        </span>
        <span class="badge badge-ext">extensible</span>
      </div>

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1A2C48;border:1px solid #304870;border-radius:12px;
                padding:1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#818CF8;
                  text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;">
        Design Pattern
      </div>
      <div style="font-size:13px;color:#A8C0D8;line-height:1.9;
                  font-family:'IBM Plex Mono',monospace;">
        Abstract Base Class<br>
        Factory function<br>
        SQLAlchemy Engine<br>
        <span style="color:#93C5FD;">DB_BACKEND=sqlserver</span><br>
        <span style="color:#6890B0;">→ switch without code change</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# KEY DESIGN DECISIONS — order: 3-2-4-1
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.75rem;'>", unsafe_allow_html=True)
section_label("Key Design Decisions")

finding(
    "<strong>Dual save — CSV + database</strong>: every transform writes both a CSV file "
    "and a database table. The pipeline runs fully standalone (files only) or integrated "
    "with a database — making it portable from local development to cloud deployment "
    "without changing a single line of pipeline code"
)
finding(
    "<strong>Medallion schemas at database level</strong>: Bronze, Silver, and Gold are real "
    "database schemas (not just folder names), enforced at creation time. "
    "Data structure is consistent whether you read from files or query the database directly"
)
finding(
    "<strong>Partitioned storage + consolidation</strong>: Comtrade data is extracted in "
    "partitions (year / product / flows) enabling parallel batches across sessions. "
    "World Bank uses per-indicator checkpoints for the same reason. "
    "A final consolidation step (<code>rglob</code> + deduplication) merges all partitions "
    "into one clean table — combining batch flexibility with query simplicity"
)
finding(
    "<strong>Backend abstraction (ABC + Factory)</strong>: switching databases is a one-line "
    "config change (<code>DB_BACKEND=databricks</code>). Adding a new backend requires one class "
    "and one line in the registry — no pipeline code changes. "
    "The same generic loader handles all tables across all backends"
)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# EXPANDERS — order: connection → DB creation → connector → loader → DuckDB demo
# ════════════════════════════════════════════════════════════════

thin_divider()

with st.expander("🔌  Test live connection"):
    st.markdown(
        "<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        "Tests all database backends and displays the ones that are reachable "
        "in the current environment.</div>",
        unsafe_allow_html=True)
    if st.button("Test connections", key="test_conn"):
        from src.db.connector import get_connector
        connected = []
        with st.spinner("Testing backends…"):
            for backend in ("sqlserver", "duckdb", "databricks"):
                try:
                    connector = get_connector(backend=backend)
                    result    = connector.test_connection()
                    if result["status"] == "ok":
                        connected.append(backend)
                except Exception:
                    pass
        if connected:
            for b in connected:
                st.success(f"✅  {b} — connected")
        else:
            st.info("No backends reachable in this environment.")

with st.expander("🗄️  Database & schema creation"):
    st.code("""
# 1 — Create the database (SQL Server)
def create_database() -> None:
    engine = create_engine(master_url)   # connects to 'master'
    with engine.connect() as conn:
        conn.execute(text(
            "IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name='Trade_Analysis') "
            "CREATE DATABASE Trade_Analysis"
        ))

# 2 — Create medallion schemas inside the database
SCHEMAS = ["bronze", "silver", "gold"]

def create_schemas() -> None:
    engine = get_connector().get_engine()
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.execute(text(f\"\"\"
                IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name='{schema}')
                EXEC('CREATE SCHEMA {schema}')
            \"\"\"))
    """, language="python")

with st.expander("📄  Connector — Abstract Base Class + Factory"):
    st.code("""
# BaseConnector enforces the interface — every backend must implement get_engine()
class BaseConnector(ABC):
    @abstractmethod
    def get_engine(self) -> Engine: ...

    @property
    @abstractmethod
    def backend(self) -> str: ...

    def test_connection(self) -> dict:
        try:
            with self.get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "backend": self.backend}
        except Exception as e:
            return {"status": "error", "backend": self.backend, "detail": str(e)}

# Registry — adding a new backend = one class + one line
_CONNECTORS = {
    "sqlserver":  SQLServerConnector,
    "databricks": DatabricksConnector,
    "duckdb":     DuckDBConnector,
}

def get_connector(backend: str | None = None) -> BaseConnector:
    config   = get_db_config()        # reads DB_BACKEND from .env
    resolved = backend or config["backend"]
    return _CONNECTORS[resolved](config)
    """, language="python")

with st.expander("💾  Loader — one function, all tables, all backends"):
    st.code("""
def save_to_db(
    df:        pd.DataFrame,
    table:     str,
    schema:    str,           # "bronze" | "silver" | "gold"
    connector: BaseConnector,
    if_exists: str = "replace",
    chunksize: int | None = None,
) -> None:
    df.to_sql(
        name=table, schema=schema,
        con=connector.get_engine(),
        if_exists=if_exists,
        index=False,
        chunksize=chunksize or DEFAULT_CHUNKSIZE,
    )
# Same call for every table and every backend — zero backend-specific code
    """, language="python")

with st.expander("🦆  Live DuckDB demo — in-memory pipeline simulation"):
    st.markdown(
        "<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        "Illustrates the medallion pattern on synthetic data: two tables per layer, "
        "one dirty row filtered at Silver, surrogate keys assigned at Gold.</div>",
        unsafe_allow_html=True)

    if st.button("▶  Run DuckDB demo", key="duckdb_demo"):
        try:
            import duckdb, pandas as pd

            with st.spinner("Running in-memory pipeline…"):
                con = duckdb.connect(":memory:")
                for schema in ["bronze", "silver", "gold"]:
                    con.execute(f"CREATE SCHEMA {schema}")

                # ── Bronze — raw, multi-table, dirty rows ─────────────
                df_b_countries = pd.DataFrame({
                    "wb_code": [250,      276,       842,             156,    392,   None],
                    "iso3":    ["FRA",    "DEU",     "USA",           "CHN",  "JPN", "???"],
                    "name":    ["France", "Germany", "United States", "China","Japan","Unknown"],
                    "region":  ["Europe", "Europe",  "Americas",      "Asia", "Asia", None],
                })
                df_b_trade = pd.DataFrame({
                    "reporter_code": [250,        250,         276,        None,       250,        276],
                    "partner_code":  [276,        842,         250,        276,        276,        250],
                    "year":          [2023,       2023,        2023,       2023,       2023,       2023],
                    "flow":          ["M",        "M",         "X",        "M",        "M",        "X"],
                    "value_usd":     [85_000_000, 120_000_000, 95_000_000, 50_000_000, 85_000_000, -1],
                })
                con.execute("CREATE TABLE bronze.countries  AS SELECT * FROM df_b_countries")
                con.execute("CREATE TABLE bronze.trade_data AS SELECT * FROM df_b_trade")

                # ── Silver — clean, deduplicate, reject invalids ───────
                df_s_countries = con.execute("""
                    SELECT CAST(wb_code AS INTEGER) AS wb_code,
                           TRIM(iso3)   AS iso3,
                           TRIM(name)   AS name,
                           TRIM(region) AS region
                    FROM bronze.countries
                    WHERE wb_code IS NOT NULL AND iso3 != '???'
                """).df()
                df_s_trade = con.execute("""
                    SELECT DISTINCT
                           CAST(reporter_code AS INTEGER) AS reporter_code,
                           CAST(partner_code  AS INTEGER) AS partner_code,
                           year, flow,
                           CAST(value_usd AS BIGINT) AS value_usd
                    FROM bronze.trade_data
                    WHERE reporter_code IS NOT NULL AND value_usd > 0
                """).df()
                con.execute("CREATE TABLE silver.countries  AS SELECT * FROM df_s_countries")
                con.execute("CREATE TABLE silver.trade_data AS SELECT * FROM df_s_trade")

                # ── Gold — dimension + fact, surrogate keys ────────────
                # entities: countries + "World" aggregate (entity_type=ENTITY)
                df_g_entities = con.execute("""
                    SELECT ROW_NUMBER() OVER (ORDER BY wb_code) AS entity_key,
                           wb_code, iso3, name, 'COUNTRY' AS entity_type
                    FROM silver.countries
                    UNION ALL
                    SELECT 99 AS entity_key, 0 AS wb_code,
                           'WLD' AS iso3, 'World' AS name, 'ENTITY' AS entity_type
                """).df()
                con.execute("CREATE TABLE gold.entities AS SELECT * FROM df_g_entities")

                df_g_trade = con.execute("""
                    SELECT e_rep.entity_key AS reporter_key,
                           e_par.entity_key AS partner_key,
                           t.year, t.flow, t.value_usd
                    FROM silver.trade_data t
                    JOIN gold.entities e_rep ON t.reporter_code = e_rep.wb_code
                    JOIN gold.entities e_par ON t.partner_code  = e_par.wb_code
                """).df()
                con.execute("CREATE TABLE gold.trade_data AS SELECT * FROM df_g_trade")

                # code divergence: simulate France having different WB vs Comtrade codes
                df_divergence = pd.DataFrame({
                    "iso3":           ["FRA"],
                    "wb_code":        [250],
                    "comtrade_code":  [251],
                })

            st.success("✅  In-memory pipeline complete — Bronze → Silver → Gold")

            def _layer_title(label, color, note="", note_color="#EF4444"):
                note_html = (f" <span style='color:{note_color};font-weight:500;"
                             f"font-size:11px;'>{note}</span>") if note else ""
                st.markdown(
                    f"<div style='font-size:12px;font-weight:700;color:{color};"
                    f"text-transform:uppercase;letter-spacing:.07em;margin:1rem 0 .3rem 0;'>"
                    f"{label}{note_html}</div>",
                    unsafe_allow_html=True,
                )

            _layer_title("🟡 Bronze — Raw", "#F59E0B",
                         "· dirty rows: NULL codes, invalid values, duplicates")
            cb1, cb2 = st.columns(2)
            with cb1:
                st.caption("countries — includes rows with missing or invalid codes")
                st.dataframe(df_b_countries, use_container_width=True, height=220)
            with cb2:
                st.caption("trade_data — includes NULL reporter, negative value, duplicate")
                st.dataframe(df_b_trade, use_container_width=True, height=220)

            _layer_title("🔵 Silver — Cleaned", "#93C5FD",
                         "· rows rejected: NULLs, invalid values, duplicates removed")
            cs1, cs2 = st.columns(2)
            with cs1:
                st.caption("countries — NULL wb_code and invalid iso3 filtered out")
                st.dataframe(df_s_countries, use_container_width=True, height=210)
            with cs2:
                st.caption("trade_data — NULL reporter, negative value and duplicate removed")
                st.dataframe(df_s_trade, use_container_width=True, height=210)

            _layer_title("🟢 Gold — Star Schema", "#34D399",
                         "· surrogate keys · entity classification · divergence check",
                         "#34D399")
            cg1, cg2 = st.columns(2)
            with cg1:
                st.caption("entities — dimension · surrogate key · COUNTRY vs ENTITY type")
                st.dataframe(df_g_entities, use_container_width=True, height=230)
            with cg2:
                st.caption("trade_data — fact · reporter_key / partner_key replace raw codes")
                st.dataframe(df_g_trade, use_container_width=True, height=180)
                st.markdown(
                    "<div style='margin-top:.5rem;background:#1C1407;border-left:3px solid #F59E0B;"
                    "border-radius:0 6px 6px 0;padding:.5rem .8rem;font-size:12px;color:#D97706;'>"
                    "⚠ <strong>Code divergence detected</strong> — WB code ≠ Comtrade code "
                    "for some reporters. Re-extraction required with Comtrade-specific codes.</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df_divergence, use_container_width=True, height=80)

            con.close()

        except ImportError:
            st.error("duckdb not installed. Run: `pip install duckdb`")
        except Exception as e:
            st.error(f"Demo error: {e}")
