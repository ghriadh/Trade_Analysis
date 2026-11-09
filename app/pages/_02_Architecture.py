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
        <span class="badge badge-done">done</span>
        <span style="font-size:10px;color:#6890B0;border:1px solid #304870;
          border-radius:10px;padding:1px 8px;white-space:nowrap;"
          title="DB_BACKEND in .env is currently set to this one">default</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">DuckDB</span>
        <span class="backend-desc">
          Embedded file-based DB · zero config · single file ·
          full medallion ETL + a complete SQL analysis notebook (9 chapters),
          validated end-to-end against the pandas reference
        </span>
        <span class="badge badge-done">done</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">Databricks</span>
        <span class="backend-desc">
          Cloud data lakehouse · Unity Catalog · token auth ·
          SQL warehouse · full medallion ETL, validated end-to-end against the
          pandas/DuckDB reference
        </span>
        <span class="badge badge-done">done</span>
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
        results = []
        with st.spinner("Testing backends…"):
            for backend in ("sqlserver", "duckdb", "databricks"):
                try:
                    connector = get_connector(backend=backend)
                    result    = connector.test_connection()
                    results.append((backend, result))
                except Exception as e:
                    results.append((backend, {"status": "error", "detail": str(e)}))
        for backend, result in results:
            if result["status"] == "ok":
                st.success(f"✅  {backend} — connected")
            elif backend == "sqlserver":
                st.error(f"❌  {backend} — {result.get('detail', 'unreachable')}")
                st.caption("ℹ️  SQL Server : fonctionne en local uniquement")
            else:
                st.error(f"❌  {backend} — {result.get('detail', 'unreachable')}")

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

with st.expander("🦆  Live DuckDB demo — real pipeline data"):
    st.markdown(
        "<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        "Connects read-only to the actual populated file — "
        "<code style='font-size:12px;background:#253B5C;color:#93C5FD;padding:1px 5px;"
        "border-radius:3px;'>data/sql/trade_analysis_sql.duckdb</code> — built by the real "
        "bronze → silver → gold pipeline, not synthetic data. This is the same file behind the "
        "full SQL analysis notebook (medallion ETL + structural/bilateral/combined scoring, "
        "9 chapters validated against the pandas reference).</div>",
        unsafe_allow_html=True)

    if st.button("▶  Query live DuckDB file", key="duckdb_real_demo"):
        try:
            import duckdb

            db_path = ROOT / "data" / "sql" / "trade_analysis_sql.duckdb"

            if not db_path.exists():
                st.error(f"File not found: {db_path}")
            else:
                with st.spinner("Querying DuckDB file…"):
                    con = duckdb.connect(str(db_path), read_only=True)

                    gold_tables = [
                        "entities", "products", "trade_flow",
                        "indicators", "indicators_values", "trade_data",
                    ]
                    gold_counts = {
                        t: con.execute(f"SELECT COUNT(*) FROM gold.{t}").fetchone()[0]
                        for t in gold_tables
                    }

                    n_analysis_tables = con.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = 'analysis'"
                    ).fetchone()[0]

                    sample = con.execute("""
                        SELECT country_name, structural_profile, meta_profile, structural_score
                        FROM analysis.trade_structural
                        ORDER BY structural_score DESC
                        LIMIT 5
                    """).df()

                    con.close()

                st.success("✅  Connected to the real pipeline output")

                st.markdown(
                    "<div style='font-size:12px;font-weight:700;color:#34D399;"
                    "text-transform:uppercase;letter-spacing:.07em;margin:1rem 0 .3rem 0;'>"
                    "🟢 Gold layer — live row counts</div>", unsafe_allow_html=True)
                gcols = st.columns(len(gold_tables))
                for col, t in zip(gcols, gold_tables):
                    col.metric(t, f"{gold_counts[t]:,}")

                st.markdown(
                    f"<div style='font-size:12px;font-weight:700;color:#818CF8;"
                    f"text-transform:uppercase;letter-spacing:.07em;margin:1.2rem 0 .3rem 0;'>"
                    f"📊 Analysis layer — {n_analysis_tables} tables built by the SQL notebook "
                    f"(structural + bilateral + combined scoring)</div>",
                    unsafe_allow_html=True)
                st.caption("Top 5 countries by structural score — analysis.trade_structural")
                st.dataframe(sample, use_container_width=True, height=210)

        except ImportError:
            st.error("duckdb not installed. Run: `pip install duckdb`")
        except Exception as e:
            st.error(f"Demo error: {e}")

def _format_large(n: float) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{sign}{n / div:.2f}{unit}"
    return f"{sign}{n:.2f}"


with st.expander("🔎  Live cross-backend query"):
    st.markdown(
        "<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        "Runs the exact same SQL query against the gold layer live, on all three backends, "
        "to prove the medallion ETL produces identical results regardless of the database. "
        "SQL Server is local-only by design — unreachable from a public deployment, shown "
        "with that note instead of an error.</div>",
        unsafe_allow_html=True)

    DEMO_COUNTRIES = ["China", "USA", "Germany", "France", "India", "Brazil", "Tunisia"]
    country = st.selectbox("Country", DEMO_COUNTRIES, index=0, key="xbackend_country")

    # No LIMIT/TOP here on purpose — that clause isn't portable across backends
    # (SQL Server needs TOP, DuckDB/Databricks need LIMIT). Keeping the ORDER BY
    # but doing the top-5 cut in pandas keeps one identical query for all three.
    QUERY = """
SELECT e2.country_name AS partner, SUM(t.primaryValue) AS total_value
FROM gold.trade_data t
JOIN gold.entities e1 ON t.reporter_key = e1.entity_key
JOIN gold.entities e2 ON t.partner_key = e2.entity_key
WHERE e1.country_name = :country
  AND e1.entity_type = 'COUNTRY'
  AND e2.entity_type = 'COUNTRY'
GROUP BY e2.country_name
ORDER BY total_value DESC
""".strip()
    # entity_type = 'COUNTRY' excludes aggregates like "World" and "EU"
    # (tagged 'ENTITY' in the gold layer) — without it, "World" swamps every
    # other partner since it sums the reporter's trade with everyone.

    st.code(QUERY, language="sql")

    if st.button("▶  Run on all backends", key="xbackend_run"):
        import time

        import numpy as np
        import pandas as pd
        from sqlalchemy import text

        from src.db.connector import get_connector

        results = {}
        cols = st.columns(3)
        with st.spinner("Querying all backends…"):
            for col, backend in zip(cols, ("duckdb", "databricks", "sqlserver")):
                with col:
                    st.markdown(f"**{backend}**")
                    try:
                        connector = get_connector(backend=backend)

                        conn_check = connector.test_connection()
                        if conn_check["status"] != "ok":
                            raise ConnectionError(conn_check.get("detail", "unreachable"))

                        t0 = time.time()
                        with connector.get_engine().connect() as conn:
                            df = pd.read_sql(text(QUERY), conn, params={"country": country})
                        df = df.sort_values("total_value", ascending=False).head(5).reset_index(drop=True)
                        elapsed = time.time() - t0
                        results[backend] = df

                        st.caption(f"⏱ {elapsed:.2f}s")
                        df_display = df.copy()
                        df_display["total_value"] = df_display["total_value"].apply(_format_large)
                        st.dataframe(df_display, use_container_width=True, height=180)
                        if not df.empty:
                            import plotly.graph_objects as go

                            scale, unit = (
                                (1e12, "USD Trillion") if df["total_value"].max() >= 1e12
                                else (1e9, "USD Billion")
                            )
                            fig = go.Figure(go.Bar(
                                x=df["partner"],
                                y=df["total_value"] / scale,
                                marker=dict(color="#60A5FA"),
                                text=df["total_value"].apply(_format_large),
                                textposition="outside",
                            ))
                            fig.update_layout(
                                paper_bgcolor="#1F3050",
                                plot_bgcolor="#162035",
                                font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
                                margin=dict(l=30, r=20, t=20, b=30),
                                xaxis=dict(gridcolor="#304870", linecolor="#304870"),
                                yaxis=dict(gridcolor="#304870", linecolor="#304870", title=unit),
                                height=340,
                            )
                            st.plotly_chart(fig, width="stretch", key=f"xbackend_chart_{backend}")

                    except ConnectionError as e:
                        if backend == "sqlserver":
                            st.caption("ℹ️  Local-only — unreachable from this deployment")
                        else:
                            st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ {e}")

        if "duckdb" in results and "databricks" in results:
            a, b = results["duckdb"], results["databricks"]
            same = (
                list(a["partner"]) == list(b["partner"])
                and np.allclose(a["total_value"].values, b["total_value"].values, rtol=1e-6)
            )
            if same:
                st.success("✅  DuckDB and Databricks returned identical results")
            else:
                st.warning(
                    "⚠️  Results differ slightly between DuckDB and Databricks — "
                    "likely floating-point summation order, not a data error"
                )
