"""
Page 1 — Pipeline & Extraction  (v4)
Layout fixes + all accumulated corrections:
- Bronze/Silver/Gold horizontal layout (6/2/2)
- Arrow removed from expander labels
- min-height on sub-boxes
- st.container() for demo sections
- Status limited to ~10 lines + Show all expander
- Order: KPIs → Pipeline Flow → Challenges → Status → Demo → Run
- 16 indicators (not 17)
- Code snippets added to challenges B and C
- Font: body 15px, labels 12px
- Contrast fixes
"""

import os
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from utils.styles import (
    inject_styles, kpi, section_label, thin_divider,
    challenge_solution, decision, finding, dot
)
from utils.etl_status_reader import load_etl_status, ETLStatus
from utils.i18n import t

inject_styles()

etl: ETLStatus = load_etl_status()

STRINGS = {
    "en": {
        "eyebrow": "ETL · Page 1 of 2",
        "title": "Pipeline & Extraction",
        "caption": (
            "Async API extraction · thread pool · rate-limit handling · "
            "checkpointing · missing country detection · resumable runs"
        ),
        "lbl_challenge": "The challenge",
        "lbl_solution": "The solution",
        "lbl_key_decision": "Key decision",
        "kpi_bronze_rows": "Bronze rows",
        "kpi_silver_rows": "Silver rows",
        "kpi_gold_rows": "Gold rows",
        "kpi_last_run": "Last run",
        "not_run_yet": "Not run yet",
        "pipeline_flow": "Pipeline Flow",
        "bronze_banner": "🟡 Bronze Layer — Raw Extraction",
        "sub_apis_title": "APIs",
        "sub_apis_lines": ["UN Comtrade", "World Bank API"],
        "sub_apis_footer": "250 reporters · 16 indicators",
        "sub_async_title": "Async + Threads",
        "sub_async_lines": ["asyncio event loop", "asyncio.to_thread()", "Configurable workers"],
        "sub_async_footer": "~5,000 safe calls",
        "sub_checkpoints_title": "Checkpoints",
        "sub_checkpoints_lines": [".done per (reporter,flow)", "Resume after crash", "Per-indicator WB"],
        "sub_checkpoints_footer": "Zero re-work on re-run",
        "sub_validation_title": "Validation",
        "sub_validation_lines": ["failed/empty/with_data", "Code divergence", "Auto re-run cmd"],
        "sub_validation_footer": "Partitioned CSV + DB",
        "silver_banner": "🔵 Silver",
        "silver_lines": ["6 transform modules", "Null checks", "Deduplication", "Type casting"],
        "silver_footer": "Module registry<br>→ targeted re-runs",
        "gold_banner": "🟢 Gold",
        "gold_lines": [
            "4 dimension tables", "2 fact tables", "Surrogate keys",
            "Dependency checker", "Divergence detection",
        ],
        "gold_footer": "316 entities",
        "section_a": "A · Managing API Calls — Comtrade",
        "a_challenge": """
Comtrade requires one API call per <strong>(reporter × flow × year)</strong> combination.
Starting from <strong>250 countries</strong>, 2 flows, and 10 years that's
<strong>~5,000 calls</strong> per product. Sequential execution would take
days and guarantee hitting API quota bans
(<em>"Quota will be replenished in 00:36:18"</em>).
""",
        "a_solution": """
<strong>Async event loop + thread pool</strong> via <code>asyncio.to_thread()</code>:
each synchronous API call runs in a dedicated thread while the async event loop
coordinates concurrency. Workers are configurable via <code>--workers</code>.
When the API returns <strong>429 (rate limit)</strong>, only that worker pauses —
others continue. A 1-second courtesy pause avoids triggering bans.
""",
        "a_decision": (
            "Use <code>asyncio.to_thread()</code> to run blocking API calls in a thread pool — "
            "combines async coordination (event loop) with true parallelism (threads), "
            "without rewriting the synchronous Comtrade SDK"
        ),
        "a_expander": "See the async + thread pool code",
        "section_b": "B · Checkpoint & Resumability",
        "b_challenge": """
With <strong>~5,000 API calls</strong> split across multiple sessions
(API quota: <em>"Quota will be replenished in 00:36:18"</em>),
a crash or quota ban mid-run means losing all progress.
The pipeline must resume from exactly where it stopped —
across sessions, days, or machines.
""",
        "b_solution": """
Each successful <strong>(reporter, flow)</strong> pair writes a lightweight
<code>.done</code> marker file. On re-run, any pair with a marker is silently skipped.
<strong>Empty responses</strong> (countries with no trade data) are tracked separately
but <em>not</em> marked done — they remain eligible for future re-extraction.
<code>failed_reporters.json</code> captures only true API errors for targeted re-run.
""",
        "b_decision": (
            "Three distinct outcomes per call: "
            "<code>with_data</code> (saved + marked done), "
            "<code>empty</code> (valid — not marked done, re-eligible), "
            "<code>failed</code> (API error — in failed_reporters.json). "
            "Only failures need a re-run"
        ),
        "b_expander": "See the checkpoint logic",
        "section_c": "C · Missing Country Detection & Correction",
        "c_challenge": """
Some countries use a <strong>different code in Comtrade</strong> vs the ISO standard
used by World Bank. A standard extraction using ISO codes returns no data
for these countries — they appear as <em>silently missing</em> with no error raised.
Additional runs may also produce new API failures requiring targeted correction.
""",
        "c_solution": """
The Gold pipeline runs a <strong>code divergence check</strong> comparing reporter codes
between World Bank and Comtrade. Divergent countries are flagged and an
<strong>exact re-extraction command is auto-generated</strong>.
API failures produce a <code>failed_reporters.json</code> with the same correction pattern.
The <strong>module registry</strong> then allows only the affected Silver/Gold modules
to re-run — not the full pipeline.
""",
        "c_decision": (
            "Two failure modes, two corrections — both auto-generate the exact CLI command needed: "
            "<strong>divergent codes</strong> → re-extract with Comtrade-specific codes; "
            "<strong>API failures</strong> → re-extract from <code>failed_reporters.json</code>. "
            "After correction: <code>run_silver --module trade_data</code> + "
            "<code>run_gold --module trade_data</code>"
        ),
        "c_expander": "See the divergence detection & correction flow",
        "section_d": "D · World Bank — Pagination & Resumability",
        "d_challenge": """
The World Bank API paginates results, but the <strong>total number of pages is unknown
before the first call</strong>. A naive implementation either fetches only page 1
(incomplete data) or over-fetches with unnecessary calls.
With <strong>16 indicators</strong> to extract, a mid-run failure
means re-fetching already-completed indicators from scratch.
""",
        "d_solution": """
The first API call reads <code>data[0]["pages"]</code> to discover the page count,
then loops through the remaining pages automatically — no hardcoded limit.
Each indicator is saved to a <strong>per-indicator checkpoint</strong> on success,
giving the same crash-resilience as Comtrade.
A <strong>session with automatic HTTP retry</strong> handles transient server
errors (5xx) with exponential backoff.
""",
        "d_decision": (
            "Discover pagination dynamically from the first response — no hardcoded page count. "
            "Per-indicator checkpoints make re-runs incremental: "
            "already-extracted indicators are skipped with no redundant API calls"
        ),
        "d_expander": "See the World Bank pagination + retry logic",
        "pipeline_run_status": "Pipeline Run Status",
        "pipeline_not_run": "Pipeline has not been run yet — metrics will appear after the first run.",
        "stage_countries": "Countries",
        "stage_products": "Products",
        "stage_trade_flow": "Trade flows",
        "stage_indicators": "Indicators",
        "stage_indicators_values": "Indicator values",
        "stage_trade_data": "Trade data",
        "duration_total_suffix": " total",
        "rows_suffix": " rows",
        "show_more_checkpoints": "Show {n} more checkpoint(s)",
        "ck_reporters": "{done}/{total} reporters",
        "ck_failed": " · {n} failed",
        "ck_empty": " · {n} empty",
        "ck_indicators": "{done}/{total} indicators · {pct}%",
        "live_api_demo": "Live API Demo",
        "live_api_demo_caption": "Single-call extractions — real-time API connectivity, no pipeline run required.",
        "tab_comtrade": "🌍  Comtrade — Trade Data",
        "tab_worldbank": "📈  World Bank — Indicator",
        "lbl_reporter_country": "Reporter country (Comtrade code)",
        "lbl_reporter_code": "Reporter code (Comtrade)",
        "lbl_year": "Year",
        "lbl_flow": "Flow",
        "flow_imports": "M (Imports)",
        "flow_exports": "X (Exports)",
        "btn_extract_trade": "▶  Extract trade data",
        "spinner_comtrade_call": "Calling Comtrade API — reporter {rc}, {year}…",
        "rows_fetched": "✅  {n} rows fetched in {elapsed}s",
        "no_data_country_year": "No data returned — try a different country or year.",
        "comtradeapicall_missing": "comtradeapicall not installed.",
        "api_error": "API error: {e}",
        "lbl_indicator": "Indicator",
        "lbl_indicator_code": "Indicator code",
        "lbl_period": "Period",
        "btn_extract_indicator": "▶  Extract indicator",
        "spinner_wb_call": "Calling World Bank API — {code}…",
        "wb_rows_summary": "✅  {n} rows in {elapsed}s · {countries} countries · {pages} page(s)",
        "no_data_indicator": "No data returned — check indicator code.",
        "preview_first_10": "Preview — first 10 rows",
        "col_year": "Year",
        "run_pipeline_local": "Run Pipeline — Local Development",
        "prod_note": (
            "Pipeline runs locally — metrics above reflect the last execution. "
            "Trade data extraction runs separately via CLI."
        ),
        "dev_note": (
            "⚠️  <strong style=\"color:#F59E0B;\">Local execution only</strong> "
            "— Silver and Gold complete in seconds. "
            "Bronze skips already-extracted data (only new indicator values may take a few minutes). "
            "Trade data extraction runs separately via CLI and is not triggered here."
        ),
        "cloud_note": (
            "⚠️  <strong style=\"color:#F59E0B;\">Live execution</strong> "
            "— a configured backend was found in this deployment's secrets. "
            "Silver and Gold complete in seconds. Bronze calls the World Bank API for real "
            "(a few minutes on the first run per container; instant afterwards via checkpoints) "
            "and writes to the configured backend. "
            "Trade data extraction runs separately via CLI and is not triggered here."
        ),
        "btn_run_bronze": "▶  Run Bronze",
        "btn_run_silver": "▶  Run Silver",
        "btn_run_gold": "▶  Run Gold",
        "running": "⏳ Running…",
        "spinner_silver": "Running Silver pipeline…",
        "spinner_gold": "Running Gold pipeline…",
        "bronze_completed": "✅  Bronze completed",
        "silver_completed": "✅  Silver completed",
        "gold_completed": "✅  Gold completed",
        "bronze_failed": "❌  Bronze failed (exit code {code}).",
        "silver_failed_exit": "❌  Silver failed (exit code {code}).",
        "gold_failed_exit": "❌  Gold failed (exit code {code}).",
        "silver_failed": "Silver failed:\n{err}",
        "gold_failed": "Gold failed:\n{err}",
    },
    "fr": {
        "eyebrow": "ETL · Page 1 sur 2",
        "title": "Pipeline & extraction",
        "caption": (
            "Extraction API asynchrone · pool de threads · gestion des limites de débit · "
            "points de contrôle · détection des pays manquants · relances reprenables"
        ),
        "lbl_challenge": "Le défi",
        "lbl_solution": "La solution",
        "lbl_key_decision": "Décision clé",
        "kpi_bronze_rows": "Lignes Bronze",
        "kpi_silver_rows": "Lignes Silver",
        "kpi_gold_rows": "Lignes Gold",
        "kpi_last_run": "Dernière exécution",
        "not_run_yet": "Pas encore exécuté",
        "pipeline_flow": "Flux du pipeline",
        "bronze_banner": "🟡 Couche Bronze — Extraction brute",
        "sub_apis_title": "API",
        "sub_apis_lines": ["UN Comtrade", "API World Bank"],
        "sub_apis_footer": "250 déclarants · 16 indicateurs",
        "sub_async_title": "Async + Threads",
        "sub_async_lines": ["Boucle d'événements asyncio", "asyncio.to_thread()", "Workers configurables"],
        "sub_async_footer": "~5 000 appels sécurisés",
        "sub_checkpoints_title": "Points de contrôle",
        "sub_checkpoints_lines": [".done par (déclarant, flux)", "Reprise après crash", "Par indicateur WB"],
        "sub_checkpoints_footer": "Aucune reprise de travail à la relance",
        "sub_validation_title": "Validation",
        "sub_validation_lines": ["failed/empty/with_data", "Divergence de code", "Commande de relance auto"],
        "sub_validation_footer": "CSV partitionné + DB",
        "silver_banner": "🔵 Silver",
        "silver_lines": ["6 modules de transformation", "Contrôles de nullité", "Déduplication", "Typage"],
        "silver_footer": "Registre de modules<br>→ relances ciblées",
        "gold_banner": "🟢 Gold",
        "gold_lines": [
            "4 tables de dimension", "2 tables de faits", "Clés de substitution",
            "Vérificateur de dépendances", "Détection de divergence",
        ],
        "gold_footer": "316 entités",
        "section_a": "A · Gestion des appels API — Comtrade",
        "a_challenge": """
Comtrade exige un appel API par combinaison <strong>(déclarant × flux × année)</strong>.
En partant de <strong>250 pays</strong>, 2 flux et 10 ans, cela fait
<strong>~5 000 appels</strong> par produit. Une exécution séquentielle prendrait
des jours et garantirait de heurter les quotas de l'API
(<em>"Quota will be replenished in 00:36:18"</em>).
""",
        "a_solution": """
<strong>Boucle d'événements async + pool de threads</strong> via <code>asyncio.to_thread()</code> :
chaque appel API synchrone s'exécute dans un thread dédié tandis que la boucle async
coordonne la concurrence. Les workers sont configurables via <code>--workers</code>.
Quand l'API renvoie <strong>429 (limite de débit)</strong>, seul ce worker se met en pause —
les autres continuent. Une pause de courtoisie d'1 seconde évite de déclencher des bans.
""",
        "a_decision": (
            "Utiliser <code>asyncio.to_thread()</code> pour exécuter les appels API bloquants dans un pool de threads — "
            "combine la coordination async (boucle d'événements) avec un vrai parallélisme (threads), "
            "sans réécrire le SDK Comtrade synchrone"
        ),
        "a_expander": "Voir le code du pool async + threads",
        "section_b": "B · Points de contrôle & reprise",
        "b_challenge": """
Avec <strong>~5 000 appels API</strong> répartis sur plusieurs sessions
(quota API : <em>"Quota will be replenished in 00:36:18"</em>),
un crash ou un ban de quota en cours d'exécution signifie perdre toute la progression.
Le pipeline doit reprendre exactement où il s'est arrêté —
entre sessions, jours ou machines.
""",
        "b_solution": """
Chaque paire <strong>(déclarant, flux)</strong> réussie écrit un fichier marqueur léger
<code>.done</code>. À la relance, toute paire marquée est silencieusement ignorée.
Les <strong>réponses vides</strong> (pays sans données commerciales) sont suivies séparément
mais <em>pas</em> marquées comme terminées — elles restent éligibles à une réextraction future.
<code>failed_reporters.json</code> ne capture que les véritables erreurs API pour une relance ciblée.
""",
        "b_decision": (
            "Trois résultats distincts par appel : "
            "<code>with_data</code> (sauvegardé + marqué terminé), "
            "<code>empty</code> (valide — pas marqué terminé, re-éligible), "
            "<code>failed</code> (erreur API — dans failed_reporters.json). "
            "Seuls les échecs nécessitent une relance"
        ),
        "b_expander": "Voir la logique de points de contrôle",
        "section_c": "C · Détection & correction des pays manquants",
        "c_challenge": """
Certains pays utilisent un <strong>code différent dans Comtrade</strong> par rapport à la norme ISO
utilisée par la World Bank. Une extraction standard avec les codes ISO ne renvoie aucune donnée
pour ces pays — ils apparaissent comme <em>silencieusement manquants</em> sans aucune erreur levée.
Des exécutions supplémentaires peuvent aussi produire de nouveaux échecs API nécessitant une correction ciblée.
""",
        "c_solution": """
Le pipeline Gold exécute une <strong>vérification de divergence de code</strong> comparant les codes déclarants
entre World Bank et Comtrade. Les pays divergents sont signalés et une
<strong>commande de réextraction exacte est générée automatiquement</strong>.
Les échecs API produisent un <code>failed_reporters.json</code> avec le même schéma de correction.
Le <strong>registre de modules</strong> permet alors de ne relancer que les modules Silver/Gold concernés
— pas tout le pipeline.
""",
        "c_decision": (
            "Deux modes d'échec, deux corrections — les deux génèrent automatiquement la commande CLI exacte nécessaire : "
            "<strong>codes divergents</strong> → réextraction avec les codes spécifiques à Comtrade ; "
            "<strong>échecs API</strong> → réextraction depuis <code>failed_reporters.json</code>. "
            "Après correction : <code>run_silver --module trade_data</code> + "
            "<code>run_gold --module trade_data</code>"
        ),
        "c_expander": "Voir le flux de détection & correction des divergences",
        "section_d": "D · World Bank — Pagination & reprise",
        "d_challenge": """
L'API de la World Bank pagine les résultats, mais le <strong>nombre total de pages est inconnu
avant le premier appel</strong>. Une implémentation naïve ne récupère que la page 1
(données incomplètes) ou sur-sollicite l'API avec des appels inutiles.
Avec <strong>16 indicateurs</strong> à extraire, un échec en cours d'exécution
signifie réextraire depuis zéro des indicateurs déjà terminés.
""",
        "d_solution": """
Le premier appel API lit <code>data[0]["pages"]</code> pour découvrir le nombre de pages,
puis boucle automatiquement sur les pages restantes — sans limite codée en dur.
Chaque indicateur est sauvegardé dans un <strong>point de contrôle par indicateur</strong> en cas de succès,
offrant la même résilience aux crashs que pour Comtrade.
Une <strong>session avec retry HTTP automatique</strong> gère les erreurs serveur transitoires
(5xx) avec un backoff exponentiel.
""",
        "d_decision": (
            "Découvrir la pagination dynamiquement depuis la première réponse — aucun nombre de pages codé en dur. "
            "Les points de contrôle par indicateur rendent les relances incrémentales : "
            "les indicateurs déjà extraits sont ignorés sans appel API redondant"
        ),
        "d_expander": "Voir la logique de pagination + retry World Bank",
        "pipeline_run_status": "État d'exécution du pipeline",
        "pipeline_not_run": "Le pipeline n'a pas encore été exécuté — les métriques apparaîtront après la première exécution.",
        "stage_countries": "Pays",
        "stage_products": "Produits",
        "stage_trade_flow": "Flux commerciaux",
        "stage_indicators": "Indicateurs",
        "stage_indicators_values": "Valeurs des indicateurs",
        "stage_trade_data": "Données commerciales",
        "duration_total_suffix": " au total",
        "rows_suffix": " lignes",
        "show_more_checkpoints": "Afficher {n} point(s) de contrôle supplémentaire(s)",
        "ck_reporters": "{done}/{total} déclarants",
        "ck_failed": " · {n} échoué(s)",
        "ck_empty": " · {n} vide(s)",
        "ck_indicators": "{done}/{total} indicateurs · {pct}%",
        "live_api_demo": "Démo API en direct",
        "live_api_demo_caption": "Extractions à appel unique — connectivité API en temps réel, aucune exécution de pipeline requise.",
        "tab_comtrade": "🌍  Comtrade — Données commerciales",
        "tab_worldbank": "📈  World Bank — Indicateur",
        "lbl_reporter_country": "Pays déclarant (code Comtrade)",
        "lbl_reporter_code": "Code déclarant (Comtrade)",
        "lbl_year": "Année",
        "lbl_flow": "Flux",
        "flow_imports": "M (Importations)",
        "flow_exports": "X (Exportations)",
        "btn_extract_trade": "▶  Extraire les données commerciales",
        "spinner_comtrade_call": "Appel à l'API Comtrade — déclarant {rc}, {year}…",
        "rows_fetched": "✅  {n} lignes récupérées en {elapsed}s",
        "no_data_country_year": "Aucune donnée renvoyée — essayez un autre pays ou une autre année.",
        "comtradeapicall_missing": "comtradeapicall n'est pas installé.",
        "api_error": "Erreur API : {e}",
        "lbl_indicator": "Indicateur",
        "lbl_indicator_code": "Code indicateur",
        "lbl_period": "Période",
        "btn_extract_indicator": "▶  Extraire l'indicateur",
        "spinner_wb_call": "Appel à l'API World Bank — {code}…",
        "wb_rows_summary": "✅  {n} lignes en {elapsed}s · {countries} pays · {pages} page(s)",
        "no_data_indicator": "Aucune donnée renvoyée — vérifiez le code indicateur.",
        "preview_first_10": "Aperçu — 10 premières lignes",
        "col_year": "Année",
        "run_pipeline_local": "Exécuter le pipeline — Développement local",
        "prod_note": (
            "Le pipeline s'exécute en local — les métriques ci-dessus reflètent la dernière exécution. "
            "L'extraction des données commerciales s'exécute séparément via la CLI."
        ),
        "dev_note": (
            "⚠️  <strong style=\"color:#F59E0B;\">Exécution locale uniquement</strong> "
            "— Silver et Gold se terminent en quelques secondes. "
            "Bronze ignore les données déjà extraites (seules les nouvelles valeurs d'indicateurs peuvent prendre quelques minutes). "
            "L'extraction des données commerciales s'exécute séparément via la CLI et n'est pas déclenchée ici."
        ),
        "cloud_note": (
            "⚠️  <strong style=\"color:#F59E0B;\">Exécution en direct</strong> "
            "— un backend valide a été trouvé dans les secrets de ce déploiement. "
            "Silver et Gold se terminent en quelques secondes. Bronze appelle réellement l'API World Bank "
            "(quelques minutes au premier lancement sur ce conteneur, instantané ensuite grâce aux checkpoints) "
            "et écrit dans le backend configuré. "
            "L'extraction des données commerciales s'exécute séparément via la CLI et n'est pas déclenchée ici."
        ),
        "btn_run_bronze": "▶  Exécuter Bronze",
        "btn_run_silver": "▶  Exécuter Silver",
        "btn_run_gold": "▶  Exécuter Gold",
        "running": "⏳ En cours…",
        "spinner_silver": "Exécution du pipeline Silver…",
        "spinner_gold": "Exécution du pipeline Gold…",
        "bronze_completed": "✅  Bronze terminé",
        "silver_completed": "✅  Silver terminé",
        "gold_completed": "✅  Gold terminé",
        "bronze_failed": "❌  Bronze a échoué (code de sortie {code}).",
        "silver_failed_exit": "❌  Silver a échoué (code de sortie {code}).",
        "gold_failed_exit": "❌  Gold a échoué (code de sortie {code}).",
        "silver_failed": "Silver a échoué :\n{err}",
        "gold_failed": "Gold a échoué :\n{err}",
    },
}

STAGE_LABELS = {
    "countries":         t(STRINGS, "stage_countries"),
    "products":          t(STRINGS, "stage_products"),
    "trade_flow":        t(STRINGS, "stage_trade_flow"),
    "indicators":        t(STRINGS, "stage_indicators"),
    "indicators_values": t(STRINGS, "stage_indicators_values"),
    "trade_data":        t(STRINGS, "stage_trade_data"),
}


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

bronze = etl.pipelines.get("bronze")
silver = etl.pipelines.get("silver")
gold   = etl.pipelines.get("gold")


def _fmt(pipeline, exclude_zero: bool = False) -> str:
    if not pipeline:
        return "—"
    rows = sum(
        s.rows for s in pipeline.stages
        if not (exclude_zero and s.rows == 0)
    )
    if rows == 0:
        return "—"
    return (f"{rows/1_000_000:.1f}M" if rows >= 1_000_000
            else f"{rows/1_000:.0f}K" if rows >= 1_000 else str(rows))


last_run = (etl.last_updated or t(STRINGS, "not_run_yet")).replace(" UTC", "")

cols = st.columns(4)
cols[0].markdown(kpi(t(STRINGS, "kpi_bronze_rows"), _fmt(bronze, exclude_zero=True), "amber"), unsafe_allow_html=True)
cols[1].markdown(kpi(t(STRINGS, "kpi_silver_rows"), _fmt(silver), "blue"),  unsafe_allow_html=True)
cols[2].markdown(kpi(t(STRINGS, "kpi_gold_rows"),   _fmt(gold),   "teal"),  unsafe_allow_html=True)
cols[3].markdown(kpi(t(STRINGS, "kpi_last_run"),    last_run, ""),           unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# PIPELINE FLOW — Bronze large + Silver/Gold stacked on right
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "pipeline_flow"))

col_bronze, col_silver, col_gold = st.columns([6, 2, 2])

# ── Bronze ────────────────────────────────────────────────────
with col_bronze:
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #92400E55;
                border-radius:14px; padding:.85rem 1rem 1rem 1rem;">
      <div style="font-size:11px; font-weight:700; color:#F59E0B;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        {t(STRINGS, "bronze_banner")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    def _sub(icon, title, lines, footer):
        items = "".join(f"<div style='margin-bottom:2px;'>· {l}</div>" for l in lines)
        return f"""
        <div style="background:#1A2840; border:1px solid #2C3F58;
                    border-radius:10px; padding:.8rem .9rem;
                    height:160px; display:flex; flex-direction:column;">
          <div style="font-size:12px; font-weight:600; color:#FCD34D;
                      margin-bottom:8px; flex-shrink:0;">{icon} {title}</div>
          <div style="font-size:13px; color:#A8C0D8; line-height:1.75; flex:1;">{items}</div>
          <div style="margin-top:6px; font-size:11px; color:#D97706;
                      font-family:'IBM Plex Mono',monospace; flex-shrink:0;">{footer}</div>
        </div>"""

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_sub("🌐", t(STRINGS, "sub_apis_title"),
        t(STRINGS, "sub_apis_lines"),
        t(STRINGS, "sub_apis_footer")), unsafe_allow_html=True)

    s2.markdown(_sub("⚡", t(STRINGS, "sub_async_title"),
        t(STRINGS, "sub_async_lines"),
        t(STRINGS, "sub_async_footer")), unsafe_allow_html=True)

    s3.markdown(_sub("📍", t(STRINGS, "sub_checkpoints_title"),
        t(STRINGS, "sub_checkpoints_lines"),
        t(STRINGS, "sub_checkpoints_footer")), unsafe_allow_html=True)

    s4.markdown(_sub("🔍", t(STRINGS, "sub_validation_title"),
        t(STRINGS, "sub_validation_lines"),
        t(STRINGS, "sub_validation_footer")), unsafe_allow_html=True)

# ── Silver ────────────────────────────────────────────────────
with col_silver:
    _silver_lines = "".join(f"<div>· {l}</div>" for l in t(STRINGS, "silver_lines"))
    st.markdown(f"""
    <div style="background:#0F2240; border:1px solid #3B82F655;
                border-radius:14px; padding:.85rem .9rem;
                height:100%; min-height:260px;">
      <div style="font-size:11px; font-weight:700; color:#93C5FD;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        {t(STRINGS, "silver_banner")}
      </div>
      <div style="font-size:13px; color:#93C5FD; line-height:1.9; opacity:.85;">
        {_silver_lines}
      </div>
      <div style="margin-top:10px; font-size:11px; color:#60A5FA;
                  font-family:'IBM Plex Mono',monospace; opacity:.8;">
        {t(STRINGS, "silver_footer")}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Gold ──────────────────────────────────────────────────────
with col_gold:
    _gold_lines = "".join(f"<div>· {l}</div>" for l in t(STRINGS, "gold_lines"))
    st.markdown(f"""
    <div style="background:#082E18; border:1px solid #10B98155;
                border-radius:14px; padding:.85rem .9rem;
                height:100%; min-height:260px;">
      <div style="font-size:11px; font-weight:700; color:#34D399;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        {t(STRINGS, "gold_banner")}
      </div>
      <div style="font-size:13px; color:#6EE7B7; line-height:1.9; opacity:.85;">
        {_gold_lines}
      </div>
      <div style="margin-top:10px; font-size:11px; color:#34D399;
                  font-family:'IBM Plex Mono',monospace; opacity:.8;">
        {t(STRINGS, "gold_footer")}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# A — API CALL MANAGEMENT
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "section_a"))

challenge_solution(
    challenge=t(STRINGS, "a_challenge"),
    solution=t(STRINGS, "a_solution"),
    challenge_label=t(STRINGS, "lbl_challenge"),
    solution_label=t(STRINGS, "lbl_solution"),
)

decision(t(STRINGS, "a_decision"), label=t(STRINGS, "lbl_key_decision"))

with st.expander(t(STRINGS, "a_expander")):
    st.code("""
async def run_task(semaphore, flow, year, reporter, product, partition_dir):
    if is_done(partition_dir, reporter, flow):   # skip checkpointed
        return None

    async with semaphore:
        # asyncio.to_thread() dispatches to ThreadPoolExecutor
        df, status = await asyncio.to_thread(
            fetch_comtrade, flow, year, reporter, product
        )
        await asyncio.sleep(1)             # courtesy pause — non-blocking

        for delay in RETRY_DELAYS:         # [10, 20] seconds
            if status != 429:
                break
            await asyncio.sleep(delay)     # only THIS thread waits
            df, status = await asyncio.to_thread(fetch_comtrade, ...)

        if df is None:
            return {"reporter": reporter, "flow": flow, "status": "failed"}
        if df.empty:
            return {"reporter": reporter, "flow": flow, "status": "empty"}

        save_reporter(df, partition_dir, reporter, flow)
        return {"reporter": reporter, "flow": flow, "status": "with_data"}

# Bounded concurrency — configurable via --workers
semaphore = asyncio.Semaphore(workers)
results   = await asyncio.gather(*tasks)
    """, language="python")


# ════════════════════════════════════════════════════════════════
# B — CHECKPOINT & RESUMABILITY
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "section_b"))

challenge_solution(
    challenge=t(STRINGS, "b_challenge"),
    solution=t(STRINGS, "b_solution"),
    challenge_label=t(STRINGS, "lbl_challenge"),
    solution_label=t(STRINGS, "lbl_solution"),
)

decision(t(STRINGS, "b_decision"), label=t(STRINGS, "lbl_key_decision"))

with st.expander(t(STRINGS, "b_expander")):
    st.code("""
# Partition layout
data/bronze/trade_data/
    year=2022/product=TOTAL/flows=M_X/
        134_M.csv        ← data for reporter 134, import
        134_M.done       ← marker: extracted with data
        failed_reporters.json  ← true API failures only
        .checkpoints/    ← marker files directory

def is_done(partition_dir, reporter, flow) -> bool:
    return (partition_dir / ".checkpoints" / f"{reporter}_{flow}.done").exists()

def mark_done(partition_dir, reporter, flow) -> None:
    path = partition_dir / ".checkpoints" / f"{reporter}_{flow}.done"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

# On re-run — only pending pairs are fetched
all_pairs = [(f, r) for f in flows for r in reporters]
pending   = [(f, r) for f, r in all_pairs
             if not is_done(partition_dir, r, f)]
logger.info("Skipping %d already-done pairs", len(all_pairs) - len(pending))
    """, language="python")


# ════════════════════════════════════════════════════════════════
# C — MISSING COUNTRY DETECTION
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "section_c"))

challenge_solution(
    challenge=t(STRINGS, "c_challenge"),
    solution=t(STRINGS, "c_solution"),
    challenge_label=t(STRINGS, "lbl_challenge"),
    solution_label=t(STRINGS, "lbl_solution"),
)

decision(t(STRINGS, "c_decision"), label=t(STRINGS, "lbl_key_decision"))

with st.expander(t(STRINGS, "c_expander")):
    st.code("""
# Gold — detect code divergence after merging WB and Comtrade
divergent = df_merged[df_merged["wb_code"] != df_merged["comtrade_code"]]

if not divergent.empty:
    codes = " ".join(divergent["country_code"].astype(int).astype(str).tolist())
    print(f"WARNING: {len(divergent)} countries with divergent codes")
    print(f"  → python -m src.bronze.extract_trade_data \\")
    print(f"         --year 2022 --product TOTAL --flows M X \\")
    print(f"         --reporters {codes}")

# API failures → re-run with failed_reporters.json
#   python -m src.bronze.extract_trade_data --reporters 134 276 512

# After any correction — only re-run affected modules
#   python -m src.pipelines.run_silver --module trade_data
#   python -m src.pipelines.run_gold   --module trade_data
    """, language="python")


# ════════════════════════════════════════════════════════════════
# D — WORLD BANK PAGINATION
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "section_d"))

challenge_solution(
    challenge=t(STRINGS, "d_challenge"),
    solution=t(STRINGS, "d_solution"),
    challenge_label=t(STRINGS, "lbl_challenge"),
    solution_label=t(STRINGS, "lbl_solution"),
)

decision(t(STRINGS, "d_decision"), label=t(STRINGS, "lbl_key_decision"))

with st.expander(t(STRINGS, "d_expander")):
    st.code("""
# HTTP session — automatic retry on 5xx errors
session = requests.Session()
retry   = Retry(total=3, backoff_factor=2,      # 2s → 4s → 8s
                status_forcelist=[500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retry))

def extract_indicator(session, indicator, start_year, end_year):
    # First call — discover total page count from response metadata
    df_all, pages = fetch_page(session, indicator, start_year, end_year, page=1)

    for page in range(2, pages + 1):             # loop only known pages
        time.sleep(1)
        df_page, _ = fetch_page(session, indicator, start_year, end_year, page)
        df_all = pd.concat([df_all, df_page])
    return df_all

# Per-indicator checkpoint — skip already-done on re-run
for indicator in all_indicators:
    if is_done(indicator):
        df_final = pd.concat([df_final, load_from_checkpoint(indicator)])
        continue
    df = extract_indicator(session, indicator, start_year, end_year)
    mark_done(indicator, df)
    df_final = pd.concat([df_final, df])
    """, language="python")


# ════════════════════════════════════════════════════════════════
# PIPELINE RUN STATUS
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label(t(STRINGS, "pipeline_run_status"))

if not etl.has_data:
    st.info(t(STRINGS, "pipeline_not_run"))
else:
    for pipeline_name, pipeline in etl.pipelines.items():
        d       = dot(pipeline.status)
        bdg_cls = {"done": "done", "partial": "partial", "partial_errors": "partial",
                   "failed": "failed", "running": "partial"}.get(pipeline.status, "pending")

        st.markdown(f"""
        <div class="status-row">
          <div class="dot dot-{d}"></div>
          <span style="font-weight:600;font-size:15px;color:#F1F5F9;">{pipeline.label}</span>
          <span class="badge badge-{bdg_cls}">{pipeline.status}</span>
          {"<span style='margin-left:auto;font-size:13px;color:#8AABB0;"
           "font-family:IBM Plex Mono,monospace;'>"
           + pipeline.total_duration_fmt + t(STRINGS, "duration_total_suffix") + "</span>"
           if pipeline.total_duration_s > 0 else ""}
        </div>""", unsafe_allow_html=True)

        for stage in pipeline.stages:
            ds  = dot(stage.status)
            lbl = STAGE_LABELS.get(stage.name, stage.label)
            st.markdown(f"""
            <div class="stage-row">
              <div class="dot dot-{ds}" style="width:7px;height:7px;"></div>
              <span style="min-width:170px;color:#C4D5E8;">{lbl}</span>
              <span style="color:#8AABB0;min-width:90px;text-align:right;
                           font-family:'IBM Plex Mono',monospace;">{stage.rows_fmt}{t(STRINGS, "rows_suffix")}</span>
              <span style="color:#6890B0;min-width:70px;text-align:right;
                           font-family:'IBM Plex Mono',monospace;">{stage.duration_fmt}</span>
              {"<span style='color:#EF4444;margin-left:10px;font-size:12px;'>"
               + (stage.error or "") + "</span>" if stage.error else ""}
            </div>""", unsafe_allow_html=True)

        # Checkpoints — limited display
        if pipeline_name == "bronze":
            for stage in pipeline.stages:
                if not stage.checkpoints:
                    continue

                if stage.name == "trade_data":
                    runs = {k: v for k, v in stage.checkpoints.items()
                            if isinstance(v, dict) and "total" in v}
                    if not runs:
                        continue

                    sorted_runs = sorted(runs.items())
                    preview     = sorted_runs[:10]
                    remainder   = sorted_runs[10:]

                    def _ck_row(run_key, counts):
                        partition  = run_key.split("#")[0]
                        timestamp  = run_key.split("#")[1] if "#" in run_key else ""
                        with_data_n = counts.get("with_data", counts.get("done", 0))
                        empty_n    = counts.get("empty", 0)
                        done_n     = with_data_n + empty_n
                        total_n    = counts.get("total", 0)
                        failed_n   = counts.get("failed", 0)
                        run_type   = counts.get("type", "standard")
                        color      = "#10B981" if failed_n == 0 else "#F59E0B"
                        bdg        = "corr" if run_type == "correction" else "pending"
                        return f"""
                        <div class="ck-line">
                          <span style="min-width:180px;color:#8AABB0;">{partition}</span>
                          <span style="color:{color};font-weight:500;">{t(STRINGS, "ck_reporters").format(done=done_n, total=total_n)}</span>
                          {"<span class='ck-fail'>&nbsp;" + t(STRINGS, "ck_failed").format(n=failed_n) + "</span>" if failed_n else ""}
                          {"<span style='color:#8AABB0;'>&nbsp;" + t(STRINGS, "ck_empty").format(n=empty_n) + "</span>" if empty_n else ""}
                          {"<span style='color:#64748B;margin-left:6px;'>" + timestamp + "</span>" if timestamp else ""}
                          <span class="badge badge-{bdg}" style="margin-left:6px;">{run_type}</span>
                        </div>"""

                    for rk, rc in preview:
                        st.markdown(_ck_row(rk, rc), unsafe_allow_html=True)

                    if remainder:
                        with st.expander(t(STRINGS, "show_more_checkpoints").format(n=len(remainder))):
                            for rk, rc in remainder:
                                st.markdown(_ck_row(rk, rc), unsafe_allow_html=True)

                elif stage.name == "indicators_values":
                    ck  = stage.checkpoints
                    pct = ck.get("progress_pct", 0)
                    st.progress(pct / 100)
                    failed_n = ck.get("failed", 0)
                    st.markdown(
                        f'<div class="ck-line">'
                        f'{t(STRINGS, "ck_indicators").format(done=ck.get("done",0), total=ck.get("total",0), pct=pct)}'
                        + (f'<span class="ck-fail">{t(STRINGS, "ck_failed").format(n=failed_n)}</span>' if failed_n else "")
                        + "</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LIVE API DEMO
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label(t(STRINGS, "live_api_demo"))
st.caption(t(STRINGS, "live_api_demo_caption"))

tab1, tab2 = st.tabs([t(STRINGS, "tab_comtrade"), t(STRINGS, "tab_worldbank")])

with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # Priority: gold entities (Comtrade codes) > bronze countries (ISO codes)
        gold_ent = ROOT / "data" / "gold" / "entities" / "entities.csv"
        bronze_c = ROOT / "data" / "bronze" / "countries" / "countries.csv"
        src_path = gold_ent if gold_ent.exists() else bronze_c

        if src_path.exists():
            df_c    = pd.read_csv(src_path)
            # Find best name/code columns available
            name_c  = next((c for c in ["country_name","reporter_desc","name"]        if c in df_c.columns), None)
            code_c  = next((c for c in ["reporter_code","comtrade_code","country_code"] if c in df_c.columns), None)
            if name_c and code_c:
                opts = (df_c[[name_c, code_c]].dropna()
                        .assign(label=lambda d: d[name_c] + " (" + d[code_c].astype(int).astype(str) + ")")
                        .sort_values(name_c))
                default = next((l for l in opts["label"].tolist() if "France" in l), None)
                sel = st.selectbox(t(STRINGS, "lbl_reporter_country"), opts["label"].tolist(),
                                   index=opts["label"].tolist().index(default) if default else 0,
                                   key="trade_country")
                rc = int(opts.loc[opts["label"] == sel, code_c].iloc[0])
            else:
                rc = st.number_input(t(STRINGS, "lbl_reporter_code"), value=251, step=1)
        else:
            rc = st.number_input(t(STRINGS, "lbl_reporter_code"), value=251, step=1)
    with c2:
        year = st.selectbox(t(STRINGS, "lbl_year"), list(range(2024, 2014, -1)), index=0, key="trade_year")
    with c3:
        flow      = st.selectbox(t(STRINGS, "lbl_flow"),
                                  [t(STRINGS, "flow_imports"), t(STRINGS, "flow_exports")],
                                  index=0, key="trade_flow")
        flow_code = flow.split()[0]

    if st.button(t(STRINGS, "btn_extract_trade"), key="btn_trade"):
        try:
            import comtradeapicall
            with st.spinner(t(STRINGS, "spinner_comtrade_call").format(rc=rc, year=year)):
                t0     = time.time()
                df_out = comtradeapicall.previewFinalData(
                    typeCode="C", freqCode="A", clCode="HS",
                    period=str(year), reporterCode=str(rc),
                    cmdCode="TOTAL", flowCode=flow_code,
                    partnerCode=None, partner2Code=0,
                    customsCode=None, motCode=0,
                    maxRecords=500, format_output="JSON", includeDesc=True,
                )
                elapsed = round(time.time() - t0, 2)
            if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                st.success(t(STRINGS, "rows_fetched").format(n=len(df_out), elapsed=elapsed))
                show = [c for c in ["reporterDesc", "partnerDesc", "flowCode",
                                    "cmdCode", "primaryValue"] if c in df_out.columns]
                st.dataframe(df_out[show].head(10), width='stretch')
            else:
                st.warning(t(STRINGS, "no_data_country_year"))
        except ImportError:
            st.error(t(STRINGS, "comtradeapicall_missing"))
        except Exception as e:
            st.error(t(STRINGS, "api_error").format(e=e))

with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        ip = ROOT / "data" / "resources" / "indicators.csv"
        if ip.exists():
            df_i    = pd.read_csv(ip)
            col_n   = "indicator_name" if "indicator_name" in df_i.columns else "indicator_code"
            ind_opts = (df_i["indicator_code"] + " — " + df_i[col_n]).tolist()
            sel_ind  = st.selectbox(t(STRINGS, "lbl_indicator"), ind_opts, index=0, key="wb_indicator")
            ind_code = sel_ind.split(" — ")[0]
        else:
            ind_code = st.text_input(t(STRINGS, "lbl_indicator_code"), value="NY.GDP.MKTP.KD.ZG")
    with c2:
        yr_range = st.selectbox(t(STRINGS, "lbl_period"), ["2015–2024", "2015–2019", "2020–2024"],
                                index=0, key="wb_period")
        sy, ey = (int(y) for y in yr_range.split("–"))

    if st.button(t(STRINGS, "btn_extract_indicator"), key="btn_ind"):
        try:
            import requests as req
            url = (f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}"
                   f"?format=json&per_page=500&date={sy}:{ey}&page=1")
            with st.spinner(t(STRINGS, "spinner_wb_call").format(code=ind_code)):
                t0      = time.time()
                r       = req.get(url, timeout=30)
                r.raise_for_status()
                dat     = r.json()
                elapsed = round(time.time() - t0, 2)
            if dat and len(dat) > 1 and dat[1]:
                df_wb = pd.DataFrame(dat[1])
                df_wb["country_name"] = df_wb["country"].apply(
                    lambda x: x["value"] if isinstance(x, dict) else x)
                df_wb["date"]  = pd.to_numeric(df_wb["date"],  errors="coerce")
                df_wb["value"] = pd.to_numeric(df_wb["value"], errors="coerce")
                df_wb = df_wb.dropna(subset=["value"])
                total_pages = dat[0].get("pages", 1)
                st.success(t(STRINGS, "wb_rows_summary").format(
                    n=len(df_wb), elapsed=elapsed,
                    countries=df_wb['country_name'].nunique(), pages=total_pages))
                sample = next(
                    (c for c in ["France", "Germany", "United States", "China", "Tunisia"]
                     if c in df_wb["country_name"].values),
                    df_wb["country_name"].iloc[0])
                year_col = t(STRINGS, "col_year")
                df_s = (df_wb[df_wb["country_name"] == sample]
                        .sort_values("date")[["date", "value"]]
                        .rename(columns={"date": year_col, "value": ind_code}))
                st.markdown(f"**{sample}** — {ind_code}")
                st.line_chart(df_s.set_index(year_col), height=180)
                with st.expander(t(STRINGS, "preview_first_10")):
                    st.dataframe(df_wb[["country_name", "date", "value"]].head(10),
                                 width='stretch')
            else:
                st.warning(t(STRINGS, "no_data_indicator"))
        except Exception as e:
            st.error(t(STRINGS, "api_error").format(e=e))


# ════════════════════════════════════════════════════════════════
# RUN PIPELINE — local dev (.env present, CSV-only, unchanged), or
# a cloud deployment with a valid backend in Streamlit secrets
# (bridged into os.environ in Home.py) — in that case we also pass
# --db so the run actually lands in that backend, not just CSV.
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label(t(STRINGS, "run_pipeline_local"))

has_local_env = (ROOT / ".env").exists()

run_backend = None
if not has_local_env:
    from src.config.db_config import get_db_config
    try:
        run_backend = get_db_config()["backend"]
    except EnvironmentError:
        run_backend = None

is_prod = not (has_local_env or run_backend)

if is_prod:
    st.markdown(f"""
    <div style="background:#141E30; border:1px solid #243044; border-radius:10px;
                padding:.9rem 1.1rem; font-size:14px; color:#5A7A9A;">
      {t(STRINGS, "prod_note")}
    </div>
    """, unsafe_allow_html=True)
else:
    note_key = "dev_note" if has_local_env else "cloud_note"
    st.markdown(f"""
    <div style="background:#1C1407; border:1px solid #78350F44;
                border-left:3px solid #F59E0B; border-radius:0 10px 10px 0;
                padding:.9rem 1.1rem; font-size:14px; color:#D97706; margin-bottom:.75rem;">
      {t(STRINGS, note_key)}
    </div>
    """, unsafe_allow_html=True)

    import subprocess

    for key, default in [
        ("bronze_proc",   None),
        ("bronze_result", None),
        ("silver_result", None),
        ("gold_result",   None),
        ("silver_stderr", None),
        ("gold_stderr",   None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    bronze_running = (
        st.session_state.bronze_proc is not None
        and st.session_state.bronze_proc.poll() is None
    )
    bronze_just_done = (
        st.session_state.bronze_proc is not None and not bronze_running
    )

    if bronze_just_done:
        rc = st.session_state.bronze_proc.returncode
        st.session_state.bronze_proc   = None
        st.session_state.bronze_result = "ok" if rc == 0 else f"fail:{rc}"
        etl = load_etl_status()

    def _stage_row(pipeline_key, color):
        p = etl.pipelines.get(pipeline_key)
        if not p:
            return
        cols_s = st.columns(len(p.stages))
        for col, stage in zip(cols_s, p.stages):
            lbl = STAGE_LABELS.get(stage.name, stage.name)
            col.markdown(
                f"<div style='font-size:12px;color:{color};font-weight:600;'>{lbl}</div>"
                f"<div style='font-size:12px;color:#A8C0D8;'>{stage.rows_fmt}{t(STRINGS, 'rows_suffix')}"
                f" · {stage.duration_fmt}</div>",
                unsafe_allow_html=True,
            )

    # ── Result messages — stack in pipeline order ────────────────
    if st.session_state.bronze_result is not None:
        res = st.session_state.bronze_result
        if res == "ok":
            st.success(t(STRINGS, "bronze_completed"))
            _stage_row("bronze", "#34D399")
        else:
            st.error(t(STRINGS, "bronze_failed").format(code=res.split(':', 1)[1]))

    if st.session_state.silver_result is not None:
        res = st.session_state.silver_result
        if res == "ok":
            st.success(t(STRINGS, "silver_completed"))
            _stage_row("silver", "#60A5FA")
        elif st.session_state.silver_stderr:
            st.error(t(STRINGS, "silver_failed").format(err=st.session_state.silver_stderr[-500:]))
        else:
            st.error(t(STRINGS, "silver_failed_exit").format(code=res.split(':', 1)[1]))

    if st.session_state.gold_result is not None:
        res = st.session_state.gold_result
        if res == "ok":
            st.success(t(STRINGS, "gold_completed"))
            _stage_row("gold", "#FBBF24")
        elif st.session_state.gold_stderr:
            st.error(t(STRINGS, "gold_failed").format(err=st.session_state.gold_stderr[-500:]))
        else:
            st.error(t(STRINGS, "gold_failed_exit").format(code=res.split(':', 1)[1]))

    def _cmd(module: str) -> list[str]:
        cmd = [sys.executable, "-m", module]
        if run_backend:
            cmd += ["--db", run_backend]
        return cmd

    # ── Buttons ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    if c1.button(t(STRINGS, "btn_run_bronze"), key="run_bronze", disabled=bronze_running):
        for k in ("bronze_result", "silver_result", "gold_result", "silver_stderr", "gold_stderr"):
            st.session_state[k] = None
        st.session_state.bronze_proc = subprocess.Popen(
            _cmd("src.pipelines.run_bronze"), cwd=str(ROOT), env=os.environ.copy(),
        )
        st.rerun()

    if bronze_running:
        c1.markdown(
            f"<div style='font-size:13px;color:#F59E0B;margin-top:6px;'>"
            f"{t(STRINGS, 'running')}</div>",
            unsafe_allow_html=True,
        )

    if c2.button(t(STRINGS, "btn_run_silver"), key="run_silver", disabled=bronze_running):
        with st.spinner(t(STRINGS, "spinner_silver")):
            r = subprocess.run(
                _cmd("src.pipelines.run_silver"),
                cwd=str(ROOT), capture_output=True, text=True, env=os.environ.copy(),
            )
        etl = load_etl_status()
        st.session_state.silver_result = "ok" if r.returncode == 0 else f"fail:{r.returncode}"
        st.session_state.silver_stderr = r.stderr if r.returncode != 0 else None
        st.rerun()

    if c3.button(t(STRINGS, "btn_run_gold"), key="run_gold", disabled=bronze_running):
        with st.spinner(t(STRINGS, "spinner_gold")):
            r = subprocess.run(
                _cmd("src.pipelines.run_gold"),
                cwd=str(ROOT), capture_output=True, text=True, env=os.environ.copy(),
            )
        etl = load_etl_status()
        st.session_state.gold_result = "ok" if r.returncode == 0 else f"fail:{r.returncode}"
        st.session_state.gold_stderr = r.stderr if r.returncode != 0 else None
        st.rerun()

    # Auto-refresh every 3s while Bronze is running
    if bronze_running:
        time.sleep(3)
        st.rerun()
