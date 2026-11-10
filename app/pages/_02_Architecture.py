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
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "ETL · Page 2 of 2",
        "title": "Architecture & Infrastructure",
        "caption": (
            "Medallion pipeline design · multi-backend abstraction · "
            "star schema · 187 countries with data · 316 gold entities"
        ),
        "kpi_layers": "Layers",
        "kpi_backends": "Backends",
        "kpi_gold_tables": "Gold tables",
        "kpi_gold_entities": "Gold entities",
        "medallion_architecture": "Medallion Architecture",
        "tag_partitioned": "partitioned",
        "tag_dim": "dim",
        "tag_fact": "fact",
        "bronze_title": "🟡 Bronze — Raw",
        "bronze_footer": (
            "250 source countries<br>"
            "~5,000 Comtrade API calls<br>"
            "Comtrade · World Bank · GitHub"
        ),
        "silver_title": "🔵 Silver — Cleaned",
        "silver_footer": (
            "Null checks · deduplication<br>"
            "Type casting · validation<br>"
            "6 transform modules"
        ),
        "gold_title": "🟢 Gold — Star Schema",
        "gold_footer": (
            "316 entities (250 countries + aggregates)<br>"
            "Surrogate keys · referential integrity<br>"
            "Divergence detection · dependency check"
        ),
        "database_backend": "Database Backend",
        "backend_sqlserver_desc": (
            "Local relational DB · Windows Auth (no password) · "
            "ODBC driver · <code style=\"font-size:12px;background:#253B5C;color:#93C5FD;"
            "padding:1px 5px;border-radius:3px;\">fast_executemany</code> for bulk inserts"
        ),
        "backend_duckdb_desc": (
            "Embedded file-based DB · zero config · single file · "
            "full medallion ETL + a complete SQL analysis notebook (9 chapters), "
            "validated end-to-end against the pandas reference"
        ),
        "backend_databricks_desc": (
            "Cloud data lakehouse · Unity Catalog · token auth · "
            "SQL warehouse · full medallion ETL, validated end-to-end against the "
            "pandas/DuckDB reference"
        ),
        "default_badge": "default",
        "default_tooltip": "DB_BACKEND in .env is currently set to this one",
        "design_pattern": "Design Pattern",
        "design_pattern_lines": (
            "Abstract Base Class<br>"
            "Factory function<br>"
            "SQLAlchemy Engine<br>"
            "<span style=\"color:#93C5FD;\">DB_BACKEND=sqlserver</span><br>"
            "<span style=\"color:#6890B0;\">→ switch without code change</span>"
        ),
        "key_design_decisions": "Key Design Decisions",
        "finding_dual_save": (
            "<strong>Dual save — CSV + database</strong>: every transform writes both a CSV file "
            "and a database table. The pipeline runs fully standalone (files only) or integrated "
            "with a database — making it portable from local development to cloud deployment "
            "without changing a single line of pipeline code"
        ),
        "finding_medallion_schemas": (
            "<strong>Medallion schemas at database level</strong>: Bronze, Silver, and Gold are real "
            "database schemas (not just folder names), enforced at creation time. "
            "Data structure is consistent whether you read from files or query the database directly"
        ),
        "finding_partitioned_storage": (
            "<strong>Partitioned storage + consolidation</strong>: Comtrade data is extracted in "
            "partitions (year / product / flows) enabling parallel batches across sessions. "
            "World Bank uses per-indicator checkpoints for the same reason. "
            "A final consolidation step (<code>rglob</code> + deduplication) merges all partitions "
            "into one clean table — combining batch flexibility with query simplicity"
        ),
        "finding_backend_abstraction": (
            "<strong>Backend abstraction (ABC + Factory)</strong>: switching databases is a one-line "
            "config change (<code>DB_BACKEND=databricks</code>). Adding a new backend requires one class "
            "and one line in the registry — no pipeline code changes. "
            "The same generic loader handles all tables across all backends"
        ),
        "exp_test_connection": "🔌  Test live connection",
        "exp_test_connection_body": (
            "Tests all database backends and displays the ones that are reachable "
            "in the current environment."
        ),
        "btn_test_connections": "Test connections",
        "spinner_testing_backends": "Testing backends…",
        "connected": "✅  {backend} — connected",
        "sqlserver_local_note": "ℹ️  SQL Server : runs locally only",
        "exp_db_creation": "🗄️  Database & schema creation",
        "exp_connector": "📄  Connector — Abstract Base Class + Factory",
        "exp_loader": "💾  Loader — one function, all tables, all backends",
        "exp_duckdb_demo": "🦆  Live DuckDB demo — real pipeline data",
        "exp_duckdb_demo_body": (
            "Connects read-only to the actual populated file — "
            "<code style='font-size:12px;background:#253B5C;color:#93C5FD;padding:1px 5px;"
            "border-radius:3px;'>data/sql/trade_analysis_sql.duckdb</code> — built by the real "
            "bronze → silver → gold pipeline, not synthetic data. This is the same file behind the "
            "full SQL analysis notebook (medallion ETL + structural/bilateral/combined scoring, "
            "9 chapters validated against the pandas reference)."
        ),
        "btn_query_duckdb": "▶  Query live DuckDB file",
        "file_not_found": "File not found: {path}",
        "spinner_querying_duckdb": "Querying DuckDB file…",
        "connected_real_pipeline": "✅  Connected to the real pipeline output",
        "gold_layer_counts": "🟢 Gold layer — live row counts",
        "analysis_layer_tables": (
            "📊 Analysis layer — {n} tables built by the SQL notebook "
            "(structural + bilateral + combined scoring)"
        ),
        "top5_structural_caption": "Top 5 countries by structural score — analysis.trade_structural",
        "duckdb_not_installed": "duckdb not installed. Run: `pip install duckdb`",
        "demo_error": "Demo error: {e}",
        "exp_cross_backend": "🔎  Live cross-backend query",
        "exp_cross_backend_body": (
            "Runs the exact same SQL query against the gold layer live, on all three backends, "
            "to prove the medallion ETL produces identical results regardless of the database. "
            "SQL Server is local-only by design — unreachable from a public deployment, shown "
            "with that note instead of an error."
        ),
        "country_label": "Country",
        "btn_run_all_backends": "▶  Run on all backends",
        "elapsed_caption": "⏱ {elapsed:.2f}s",
        "local_only_note": "ℹ️  Local-only — unreachable from this deployment",
        "identical_results": "✅  DuckDB and Databricks returned identical results",
        "results_differ": (
            "⚠️  Results differ slightly between DuckDB and Databricks — "
            "likely floating-point summation order, not a data error"
        ),
        "usd_trillion": "USD Trillion",
        "usd_billion": "USD Billion",
        "unreachable": "unreachable",
        "spinner_run_all_backends": "Querying all backends…",
    },
    "fr": {
        "eyebrow": "ETL · Page 2 sur 2",
        "title": "Architecture & Infrastructure",
        "caption": (
            "Conception du pipeline medallion · abstraction multi-moteur · "
            "schéma en étoile · 187 pays avec données · 316 entités Gold"
        ),
        "kpi_layers": "Couches",
        "kpi_backends": "Backends",
        "kpi_gold_tables": "Tables Gold",
        "kpi_gold_entities": "Entités Gold",
        "medallion_architecture": "Architecture Medallion",
        "tag_partitioned": "partitionné",
        "tag_dim": "dim",
        "tag_fact": "fait",
        "bronze_title": "🟡 Bronze — Brut",
        "bronze_footer": (
            "250 pays source<br>"
            "~5 000 appels à l'API Comtrade<br>"
            "Comtrade · World Bank · GitHub"
        ),
        "silver_title": "🔵 Silver — Nettoyé",
        "silver_footer": (
            "Contrôles de nullité · déduplication<br>"
            "Typage · validation<br>"
            "6 modules de transformation"
        ),
        "gold_title": "🟢 Gold — Schéma en étoile",
        "gold_footer": (
            "316 entités (250 pays + agrégats)<br>"
            "Clés de substitution · intégrité référentielle<br>"
            "Détection de divergence · vérification des dépendances"
        ),
        "database_backend": "Moteur de base de données",
        "backend_sqlserver_desc": (
            "Base relationnelle locale · authentification Windows (sans mot de passe) · "
            "pilote ODBC · <code style=\"font-size:12px;background:#253B5C;color:#93C5FD;"
            "padding:1px 5px;border-radius:3px;\">fast_executemany</code> pour les insertions en masse"
        ),
        "backend_duckdb_desc": (
            "Base embarquée basée sur fichier · configuration nulle · fichier unique · "
            "ETL medallion complet + un notebook d'analyse SQL complet (9 chapitres), "
            "validé de bout en bout par rapport à la référence pandas"
        ),
        "backend_databricks_desc": (
            "Lakehouse cloud · Unity Catalog · authentification par jeton · "
            "entrepôt SQL · ETL medallion complet, validé de bout en bout par rapport à la "
            "référence pandas/DuckDB"
        ),
        "default_badge": "par défaut",
        "default_tooltip": "DB_BACKEND dans .env est actuellement réglé sur celui-ci",
        "design_pattern": "Patron de conception",
        "design_pattern_lines": (
            "Classe de base abstraite<br>"
            "Fonction fabrique<br>"
            "Moteur SQLAlchemy<br>"
            "<span style=\"color:#93C5FD;\">DB_BACKEND=sqlserver</span><br>"
            "<span style=\"color:#6890B0;\">→ changement sans modifier le code</span>"
        ),
        "key_design_decisions": "Décisions de conception clés",
        "finding_dual_save": (
            "<strong>Double écriture — CSV + base de données</strong> : chaque transformation écrit à la fois "
            "un fichier CSV et une table en base. Le pipeline fonctionne entièrement en autonomie (fichiers seuls) "
            "ou intégré à une base de données — ce qui le rend portable du développement local au déploiement "
            "cloud sans changer une seule ligne de code du pipeline"
        ),
        "finding_medallion_schemas": (
            "<strong>Schémas medallion au niveau base de données</strong> : Bronze, Silver et Gold sont de vrais "
            "schémas de base de données (pas seulement des noms de dossiers), imposés à la création. "
            "La structure des données est cohérente, que l'on lise depuis les fichiers ou interroge directement la base"
        ),
        "finding_partitioned_storage": (
            "<strong>Stockage partitionné + consolidation</strong> : les données Comtrade sont extraites en "
            "partitions (année / produit / flux) permettant des lots parallèles entre sessions. "
            "World Bank utilise des points de contrôle par indicateur pour la même raison. "
            "Une étape de consolidation finale (<code>rglob</code> + déduplication) fusionne toutes les partitions "
            "en une seule table propre — combinant flexibilité par lot et simplicité des requêtes"
        ),
        "finding_backend_abstraction": (
            "<strong>Abstraction des moteurs (ABC + Fabrique)</strong> : changer de base de données est un "
            "changement de configuration d'une ligne (<code>DB_BACKEND=databricks</code>). Ajouter un nouveau moteur "
            "ne nécessite qu'une classe et une ligne dans le registre — aucun changement au code du pipeline. "
            "Le même chargeur générique gère toutes les tables sur tous les moteurs"
        ),
        "exp_test_connection": "🔌  Tester la connexion en direct",
        "exp_test_connection_body": (
            "Teste tous les moteurs de base de données et affiche ceux qui sont accessibles "
            "dans l'environnement actuel."
        ),
        "btn_test_connections": "Tester les connexions",
        "spinner_testing_backends": "Test des moteurs…",
        "connected": "✅  {backend} — connecté",
        "sqlserver_local_note": "ℹ️  SQL Server : fonctionne en local uniquement",
        "exp_db_creation": "🗄️  Création de la base & des schémas",
        "exp_connector": "📄  Connecteur — Classe de base abstraite + Fabrique",
        "exp_loader": "💾  Chargeur — une fonction, toutes les tables, tous les moteurs",
        "exp_duckdb_demo": "🦆  Démo DuckDB en direct — données réelles du pipeline",
        "exp_duckdb_demo_body": (
            "Se connecte en lecture seule au fichier réellement peuplé — "
            "<code style='font-size:12px;background:#253B5C;color:#93C5FD;padding:1px 5px;"
            "border-radius:3px;'>data/sql/trade_analysis_sql.duckdb</code> — construit par le véritable "
            "pipeline bronze → silver → gold, pas des données synthétiques. C'est le même fichier qui alimente le "
            "notebook d'analyse SQL complet (ETL medallion + scoring structurel/bilatéral/combiné, "
            "9 chapitres validés par rapport à la référence pandas)."
        ),
        "btn_query_duckdb": "▶  Interroger le fichier DuckDB en direct",
        "file_not_found": "Fichier introuvable : {path}",
        "spinner_querying_duckdb": "Interrogation du fichier DuckDB…",
        "connected_real_pipeline": "✅  Connecté à la sortie réelle du pipeline",
        "gold_layer_counts": "🟢 Couche Gold — nombre de lignes en direct",
        "analysis_layer_tables": (
            "📊 Couche d'analyse — {n} tables construites par le notebook SQL "
            "(scoring structurel + bilatéral + combiné)"
        ),
        "top5_structural_caption": "Top 5 des pays par score structurel — analysis.trade_structural",
        "duckdb_not_installed": "duckdb n'est pas installé. Exécutez : `pip install duckdb`",
        "demo_error": "Erreur de démo : {e}",
        "exp_cross_backend": "🔎  Requête en direct multi-moteur",
        "exp_cross_backend_body": (
            "Exécute exactement la même requête SQL sur la couche gold en direct, sur les trois moteurs, "
            "pour prouver que l'ETL medallion produit des résultats identiques quelle que soit la base de données. "
            "SQL Server est local-only par conception — inaccessible depuis un déploiement public, affiché "
            "avec cette note plutôt qu'une erreur."
        ),
        "country_label": "Pays",
        "btn_run_all_backends": "▶  Exécuter sur tous les moteurs",
        "elapsed_caption": "⏱ {elapsed:.2f}s",
        "local_only_note": "ℹ️  Local uniquement — inaccessible depuis ce déploiement",
        "identical_results": "✅  DuckDB et Databricks ont renvoyé des résultats identiques",
        "results_differ": (
            "⚠️  Les résultats diffèrent légèrement entre DuckDB et Databricks — "
            "probablement l'ordre de sommation en virgule flottante, pas une erreur de données"
        ),
        "usd_trillion": "Billions USD",
        "usd_billion": "Milliards USD",
        "unreachable": "inaccessible",
        "spinner_run_all_backends": "Interrogation de tous les moteurs…",
    },
}


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.6rem;'>", unsafe_allow_html=True)
cols = st.columns(4)
for col, (label, value, color) in zip(cols, [
    (t(STRINGS, "kpi_layers"),        "3",   "teal"),
    (t(STRINGS, "kpi_backends"),      "3",   "blue"),
    (t(STRINGS, "kpi_gold_tables"),   "6",   ""),
    (t(STRINGS, "kpi_gold_entities"), "316", "amber"),
]):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MEDALLION ARCHITECTURE
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.9rem;'>", unsafe_allow_html=True)
section_label(t(STRINGS, "medallion_architecture"))

col1, arr1, col2, arr2, col3 = st.columns([10, 1, 10, 1, 10])

with col1:
    st.markdown(f"""
    <div class="med-card med-bronze">
      <div class="med-title">{t(STRINGS, "bronze_title")}</div>
      <div class="med-files">
        countries.csv<br>
        products.csv<br>
        trade_flow.csv<br>
        indicators.csv<br>
        indicators_values.csv<br>
        trade_data/<span class="med-tag"> {t(STRINGS, "tag_partitioned")}</span>
      </div>
      <div class="med-footer">
        {t(STRINGS, "bronze_footer")}
      </div>
    </div>
    """, unsafe_allow_html=True)

with arr1:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;font-size:22px;color:#304870;padding-top:60px;'>→</div>",
        unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="med-card med-silver">
      <div class="med-title">{t(STRINGS, "silver_title")}</div>
      <div class="med-files">
        countries.csv<br>
        products.csv<br>
        trade_flow.csv<br>
        indicators.csv<br>
        indicators_values.csv<br>
        trade_data.csv
      </div>
      <div class="med-footer">
        {t(STRINGS, "silver_footer")}
      </div>
    </div>
    """, unsafe_allow_html=True)

with arr2:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;font-size:22px;color:#304870;padding-top:60px;'>→</div>",
        unsafe_allow_html=True)

with col3:
    _dim = t(STRINGS, "tag_dim")
    _fact = t(STRINGS, "tag_fact")
    st.markdown(f"""
    <div class="med-card med-gold">
      <div class="med-title">{t(STRINGS, "gold_title")}</div>
      <div class="med-files">
        entities.csv<span class="med-tag"> {_dim}</span><br>
        products.csv<span class="med-tag"> {_dim}</span><br>
        trade_flow.csv<span class="med-tag"> {_dim}</span><br>
        indicators.csv<span class="med-tag"> {_dim}</span><br>
        indicators_values.csv<span class="med-tag"> {_fact}</span><br>
        trade_data.csv<span class="med-tag"> {_fact}</span>
      </div>
      <div class="med-footer">
        {t(STRINGS, "gold_footer")}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DATABASE BACKEND
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.75rem;'>", unsafe_allow_html=True)
section_label(t(STRINGS, "database_backend"))

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:6px;">

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">SQL Server</span>
        <span class="backend-desc">
          {t(STRINGS, "backend_sqlserver_desc")}
        </span>
        <span class="badge badge-done">done</span>
        <span style="font-size:10px;color:#6890B0;border:1px solid #304870;
          border-radius:10px;padding:1px 8px;white-space:nowrap;"
          title="{t(STRINGS, "default_tooltip")}">{t(STRINGS, "default_badge")}</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">DuckDB</span>
        <span class="backend-desc">
          {t(STRINGS, "backend_duckdb_desc")}
        </span>
        <span class="badge badge-done">done</span>
      </div>

      <div class="backend-row">
        <div class="dot dot-done"></div>
        <span class="backend-name">Databricks</span>
        <span class="backend-desc">
          {t(STRINGS, "backend_databricks_desc")}
        </span>
        <span class="badge badge-done">done</span>
      </div>

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#1A2C48;border:1px solid #304870;border-radius:12px;
                padding:1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#818CF8;
                  text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;">
        {t(STRINGS, "design_pattern")}
      </div>
      <div style="font-size:13px;color:#A8C0D8;line-height:1.9;
                  font-family:'IBM Plex Mono',monospace;">
        {t(STRINGS, "design_pattern_lines")}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# KEY DESIGN DECISIONS — order: 3-2-4-1
# ════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:0.75rem;'>", unsafe_allow_html=True)
section_label(t(STRINGS, "key_design_decisions"))

finding(t(STRINGS, "finding_dual_save"))
finding(t(STRINGS, "finding_medallion_schemas"))
finding(t(STRINGS, "finding_partitioned_storage"))
finding(t(STRINGS, "finding_backend_abstraction"))

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# EXPANDERS — order: connection → DB creation → connector → loader → DuckDB demo
# ════════════════════════════════════════════════════════════════

thin_divider()

with st.expander(t(STRINGS, "exp_test_connection")):
    st.markdown(
        f"<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        f"{t(STRINGS, 'exp_test_connection_body')}</div>",
        unsafe_allow_html=True)
    if st.button(t(STRINGS, "btn_test_connections"), key="test_conn"):
        from src.db.connector import get_connector
        results = []
        with st.spinner(t(STRINGS, "spinner_testing_backends")):
            for backend in ("sqlserver", "duckdb", "databricks"):
                try:
                    connector = get_connector(backend=backend)
                    result    = connector.test_connection()
                    results.append((backend, result))
                except Exception as e:
                    results.append((backend, {"status": "error", "detail": str(e)}))
        for backend, result in results:
            if result["status"] == "ok":
                st.success(t(STRINGS, "connected").format(backend=backend))
            elif backend == "sqlserver":
                st.error(f"❌  {backend} — {result.get('detail', t(STRINGS, 'unreachable'))}")
                st.caption(t(STRINGS, "sqlserver_local_note"))
            else:
                st.error(f"❌  {backend} — {result.get('detail', t(STRINGS, 'unreachable'))}")

with st.expander(t(STRINGS, "exp_db_creation")):
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

with st.expander(t(STRINGS, "exp_connector")):
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

with st.expander(t(STRINGS, "exp_loader")):
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

with st.expander(t(STRINGS, "exp_duckdb_demo")):
    st.markdown(
        f"<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        f"{t(STRINGS, 'exp_duckdb_demo_body')}</div>",
        unsafe_allow_html=True)

    if st.button(t(STRINGS, "btn_query_duckdb"), key="duckdb_real_demo"):
        try:
            import duckdb

            db_path = ROOT / "data" / "sql" / "trade_analysis_sql.duckdb"

            if not db_path.exists():
                st.error(t(STRINGS, "file_not_found").format(path=db_path))
            else:
                with st.spinner(t(STRINGS, "spinner_querying_duckdb")):
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

                st.success(t(STRINGS, "connected_real_pipeline"))

                st.markdown(
                    f"<div style='font-size:12px;font-weight:700;color:#34D399;"
                    f"text-transform:uppercase;letter-spacing:.07em;margin:1rem 0 .3rem 0;'>"
                    f"{t(STRINGS, 'gold_layer_counts')}</div>", unsafe_allow_html=True)
                gcols = st.columns(len(gold_tables))
                for col, gt in zip(gcols, gold_tables):
                    col.metric(gt, f"{gold_counts[gt]:,}")

                st.markdown(
                    f"<div style='font-size:12px;font-weight:700;color:#818CF8;"
                    f"text-transform:uppercase;letter-spacing:.07em;margin:1.2rem 0 .3rem 0;'>"
                    f"{t(STRINGS, 'analysis_layer_tables').format(n=n_analysis_tables)}</div>",
                    unsafe_allow_html=True)
                st.caption(t(STRINGS, "top5_structural_caption"))
                st.dataframe(sample, use_container_width=True, height=210)

        except ImportError:
            st.error(t(STRINGS, "duckdb_not_installed"))
        except Exception as e:
            st.error(t(STRINGS, "demo_error").format(e=e))

def _format_large(n: float) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{sign}{n / div:.2f}{unit}"
    return f"{sign}{n:.2f}"


with st.expander(t(STRINGS, "exp_cross_backend")):
    st.markdown(
        f"<div style='font-size:14px;color:#A8C0D8;margin-bottom:12px;'>"
        f"{t(STRINGS, 'exp_cross_backend_body')}</div>",
        unsafe_allow_html=True)

    DEMO_COUNTRIES = ["China", "USA", "Germany", "France", "India", "Brazil", "Tunisia"]
    country = st.selectbox(t(STRINGS, "country_label"), DEMO_COUNTRIES, index=0, key="xbackend_country")

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

    if st.button(t(STRINGS, "btn_run_all_backends"), key="xbackend_run"):
        import time

        import numpy as np
        import pandas as pd
        from sqlalchemy import text

        from src.db.connector import get_connector

        results = {}
        cols = st.columns(3)
        with st.spinner(t(STRINGS, "spinner_run_all_backends")):
            for col, backend in zip(cols, ("duckdb", "databricks", "sqlserver")):
                with col:
                    st.markdown(f"**{backend}**")
                    try:
                        connector = get_connector(backend=backend)

                        conn_check = connector.test_connection()
                        if conn_check["status"] != "ok":
                            raise ConnectionError(conn_check.get("detail", t(STRINGS, "unreachable")))

                        t0 = time.time()
                        with connector.get_engine().connect() as conn:
                            df = pd.read_sql(text(QUERY), conn, params={"country": country})
                        df = df.sort_values("total_value", ascending=False).head(5).reset_index(drop=True)
                        elapsed = time.time() - t0
                        results[backend] = df

                        st.caption(t(STRINGS, "elapsed_caption").format(elapsed=elapsed))
                        df_display = df.copy()
                        df_display["total_value"] = df_display["total_value"].apply(_format_large)
                        st.dataframe(df_display, use_container_width=True, height=180)
                        if not df.empty:
                            import plotly.graph_objects as go

                            scale, unit = (
                                (1e12, t(STRINGS, "usd_trillion")) if df["total_value"].max() >= 1e12
                                else (1e9, t(STRINGS, "usd_billion"))
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
                            st.caption(t(STRINGS, "local_only_note"))
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
                st.success(t(STRINGS, "identical_results"))
            else:
                st.warning(t(STRINGS, "results_differ"))
