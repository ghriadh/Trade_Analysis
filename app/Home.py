import os
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from utils.styles import inject_styles, kpi, thin_divider, section_label, finding
from utils.i18n import t, render_lang_toggle

st.set_page_config(
    page_title="Trade Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud "Secrets" populate st.secrets, not os.environ — bridge them
# once here so the rest of the app (os.getenv()-based config) works the same
# way locally (via .env) and when deployed.
try:
    os.environ.update({k: str(v) for k, v in st.secrets.items()})
except Exception:
    pass

render_lang_toggle()

_P = Path(__file__).parent / "pages"

# ════════════════════════════════════════════════════════════════
# STRINGS
# ════════════════════════════════════════════════════════════════

STRINGS = {
    "en": {
        "eyebrow": "Overview",
        "title": "Trade Intelligence",
        "caption": "World Bank API · Medallion architecture · 187 countries · 10 years",

        "intro_1": (
            "Built on a Medallion ETL pipeline extracting <strong style='color:#EEF4FB;'>World Bank open data</strong> — "
            "async extraction with checkpointing, multi-layer cleaning and normalisation, "
            "through to a query-ready star schema."
        ),
        "intro_2": (
            "The first analytical layer maps <strong style='color:#EEF4FB;'>international trade</strong>: "
            "global and country-level flows, bilateral analysis, structural typologies "
            "and composite scoring across multiple dimensions."
        ),
        "intro_3": (
            "The second layer examines <strong style='color:#EEF4FB;'>16 macro-economic indicators</strong> at multiple scales — "
            "from global concentration and stability to country-level growth dynamics, "
            "cross-indicator correlations, convergence analysis across country groups, "
            "multi-approach clustering, and composite scoring separating structural position "
            "from growth momentum."
        ),
        "intro_4": (
            "The two layers then <strong style='color:#EEF4FB;'>converge</strong>: looking for a macro fingerprint "
            "behind each trade profile, and testing whether macro indicators can predict trade variables — "
            "exploring the structure, direction, and limits of that link."
        ),

        "kpi_countries": "Countries",
        "kpi_years": "Years",
        "kpi_gold_rows": "Gold rows",
        "kpi_indicators": "Indicators",

        "pipeline_flow": "Data Pipeline → Analytical Flow",
        "etl_pipeline": "ETL Pipeline",
        "analytical_layers": "▼ &nbsp; Analytical Layers",

        "bronze_label": "Bronze",
        "bronze_tasks": "World Bank API extraction<br>Async · thread pool<br>Rate-limit handling<br>Checkpointing · resumable",
        "silver_label": "Silver",
        "silver_tasks": "Multi-layer cleaning<br>Type enforcement<br>Country validation<br>Schema normalisation",
        "gold_label": "Gold",
        "gold_tasks": "Star schema · 6 tables<br>316 named entities<br>Query-optimised joins<br>544K rows",

        "trade_analysis_label": "Trade Analysis",
        "trade_list": [
            "Global trade dynamics — volumes, balance, concentration, volatility",
            "Country trade profile — trends, world share, growth rates, structural scoring",
            "Cross-country comparison — quadrant views, growth heatmap",
            "Bilateral analysis — HHI, BDIS dependency score, dependency map",
            "Typology & composite scoring — trade profiles, strategic quadrant, country scorecard",
        ],
        "indicators_label": "Indicators",
        "indicators_list": [
            "Global snapshot — concentration, stability, volatility per indicator",
            "Country indicator profile — time series, z-score, CAGR, shock detection",
            "Cross-indicator correlations & PCA — structure in the macro space",
            "Convergence analysis — sigma & beta by country group",
            "Country clustering — per indicator, multi-indicator, PCA-based",
            "Composite scoring — Econ Health, Performance, Competitiveness",
        ],
        "link_analysis_label": "Link Analysis",
        "link_list": [
            "Correlation atlas — 44 trade variables × 16 macro indicators",
            "Trade intensity & derived metrics — elasticity, trade/GDP ratios",
            "Level vs dynamics — structural macro link vs idiosyncratic behavior",
            "Macro fingerprint — macro profile per trade typology",
            "Predictive analysis — macro → trade and trade → macro",
            "Country diagnostic — trade × macro positioning scorecard",
        ],

        "key_findings": "Key Findings",

        "findings_trade": [
            "World trade is structurally concentrated — top economies dominate global flows; "
            "the world aggregate masks deep country-level heterogeneity in volumes and dynamics",
            "4 structural trade profiles emerge clearly — Global Champions, Competitive Exporters, "
            "Import-Dependent, and Fragile-Isolated — each with a distinct bilateral pattern",
            "Bilateral dependencies are asymmetric: periphery countries show high dependency "
            "toward core economies; the reverse does not hold",
            "Country trade profiles are structurally stable over 10 years — "
            "typology is structural, not cyclical",
            "Composite trade scoring stratifies countries across 5 independent dimensions — "
            "volume, dynamism, bilateral exposure, resilience, and profile",
        ],
        "findings_indicators": [
            "Economic indicators are highly concentrated globally — a small number of countries "
            "dominates absolute levels; the world aggregate masks country-level heterogeneity",
            "Growth dynamics diverge sharply across countries: same indicator, very different "
            "CAGR, volatility and shock frequency depending on development stage",
            "Convergence is group-dependent — no universal sigma or beta convergence holds "
            "across all country sets; the result changes with the group defined",
            "Multi-approach clustering (per indicator, multi-indicator, PCA-based) produces "
            "overlapping but distinct groupings — no single universal macro typology emerges",
            "Composite scoring reveals a structural-momentum duality: a country can rank high "
            "on structural value position but low on growth momentum — two distinct dimensions",
        ],
        "findings_link": [
            "Trade profiles do not map cleanly onto a single macro signature — "
            "profiles overlap in the macro space; no clean fingerprint emerges",
            "Some trade variables show consistent macro correlates across methods; "
            "others resist all indicators tested — the link is selective, not global",
            "Results are bounded by the dataset scope: 16 general WB indicators, "
            "10-year median — a narrower or more granular indicator set may yield different conclusions",
        ],
        "findings_footer": (
            "⚠ Findings valid on this dataset scope: ~170 countries · 16 general WB macro indicators "
            "· 10-year median 2015–2024."
        ),

        "nav_home": "Home",
        "nav_pipeline": "Pipeline",
        "nav_architecture": "Architecture",
        "nav_overview_scope": "Overview & Scope",
        "nav_global_dynamics": "Global Dynamics",
        "nav_country_profile": "Country Profile",
        "nav_cross_country": "Cross-Country",
        "nav_bilateral_analysis": "Bilateral Analysis",
        "nav_typology_scoring": "Typology & Scoring",
        "nav_deep_dive": "Deep Dive",
        "nav_advanced_analysis": "Advanced Analysis",
        "nav_signals": "Signals",
        "nav_strategy": "Strategy",

        "section_etl": "ETL Pipeline",
        "section_trade": "Trade Analysis",
        "section_indicators": "Indicators",
        "section_link": "Link Analysis",
    },
    "fr": {
        "eyebrow": "Aperçu",
        "title": "Trade Intelligence",
        "caption": "API World Bank · Architecture médaillon · 187 pays · 10 ans",

        "intro_1": (
            "Construit sur un pipeline ETL médaillon qui extrait les <strong style='color:#EEF4FB;'>données ouvertes de la World Bank</strong> — "
            "extraction asynchrone avec checkpoints, nettoyage et normalisation multi-couches, "
            "jusqu'à un schéma en étoile prêt pour l'interrogation."
        ),
        "intro_2": (
            "La première couche analytique cartographie le <strong style='color:#EEF4FB;'>commerce international</strong> : "
            "flux mondiaux et par pays, analyse bilatérale, typologies structurelles "
            "et scoring composite sur plusieurs dimensions."
        ),
        "intro_3": (
            "La deuxième couche examine <strong style='color:#EEF4FB;'>16 indicateurs macro-économiques</strong> à plusieurs échelles — "
            "de la concentration et de la stabilité mondiales à la dynamique de croissance par pays, "
            "aux corrélations inter-indicateurs, à l'analyse de convergence par groupe de pays, "
            "au clustering multi-approches, et au scoring composite séparant la position structurelle "
            "de la dynamique de croissance."
        ),
        "intro_4": (
            "Les deux couches <strong style='color:#EEF4FB;'>convergent</strong> ensuite : recherche d'une signature macro "
            "derrière chaque profil commercial, et test de la capacité des indicateurs macro à prédire les variables commerciales — "
            "exploration de la structure, de la direction et des limites de ce lien."
        ),

        "kpi_countries": "Pays",
        "kpi_years": "Années",
        "kpi_gold_rows": "Lignes gold",
        "kpi_indicators": "Indicateurs",

        "pipeline_flow": "Pipeline de données → Flux analytique",
        "etl_pipeline": "Pipeline ETL",
        "analytical_layers": "▼ &nbsp; Couches analytiques",

        "bronze_label": "Bronze",
        "bronze_tasks": "Extraction API World Bank<br>Async · pool de threads<br>Gestion du rate-limit<br>Checkpoints · reprenable",
        "silver_label": "Silver",
        "silver_tasks": "Nettoyage multi-couches<br>Validation des types<br>Validation des pays<br>Normalisation du schéma",
        "gold_label": "Gold",
        "gold_tasks": "Schéma en étoile · 6 tables<br>316 entités nommées<br>Jointures optimisées<br>544K lignes",

        "trade_analysis_label": "Analyse commerciale",
        "trade_list": [
            "Dynamique commerciale mondiale — volumes, balance, concentration, volatilité",
            "Profil commercial par pays — tendances, part mondiale, taux de croissance, scoring structurel",
            "Comparaison entre pays — vues en quadrants, heatmap de croissance",
            "Analyse bilatérale — HHI, score de dépendance BDIS, carte de dépendance",
            "Typologie & scoring composite — profils commerciaux, quadrant stratégique, fiche pays",
        ],
        "indicators_label": "Indicateurs",
        "indicators_list": [
            "Aperçu mondial — concentration, stabilité, volatilité par indicateur",
            "Profil d'indicateurs par pays — séries temporelles, z-score, CAGR, détection de chocs",
            "Corrélations inter-indicateurs & ACP — structure dans l'espace macro",
            "Analyse de convergence — sigma & beta par groupe de pays",
            "Clustering de pays — par indicateur, multi-indicateurs, basé ACP",
            "Scoring composite — Santé économique, Performance, Compétitivité",
        ],
        "link_analysis_label": "Analyse de lien",
        "link_list": [
            "Atlas de corrélation — 44 variables commerciales × 16 indicateurs macro",
            "Intensité commerciale & métriques dérivées — élasticité, ratios commerce/PIB",
            "Niveau vs dynamique — lien macro structurel vs comportement idiosyncratique",
            "Signature macro — profil macro par typologie commerciale",
            "Analyse prédictive — macro → commerce et commerce → macro",
            "Diagnostic pays — fiche de positionnement commerce × macro",
        ],

        "key_findings": "Principaux constats",

        "findings_trade": [
            "Le commerce mondial est structurellement concentré — les plus grandes économies dominent les flux mondiaux ; "
            "l'agrégat « World » masque une forte hétérogénéité des volumes et des dynamiques au niveau pays",
            "4 profils commerciaux structurels se dégagent clairement — Champions mondiaux, Exportateurs compétitifs, "
            "Dépendants des importations, et Fragiles-isolés — chacun avec un profil bilatéral distinct",
            "Les dépendances bilatérales sont asymétriques : les pays périphériques montrent une forte dépendance "
            "envers les économies centrales ; l'inverse ne se vérifie pas",
            "Les profils commerciaux des pays sont structurellement stables sur 10 ans — "
            "la typologie est structurelle, pas cyclique",
            "Le scoring commercial composite stratifie les pays sur 5 dimensions indépendantes — "
            "volume, dynamisme, exposition bilatérale, résilience et profil",
        ],
        "findings_indicators": [
            "Les indicateurs économiques sont très concentrés au niveau mondial — un petit nombre de pays "
            "domine les niveaux absolus ; l'agrégat « World » masque l'hétérogénéité entre pays",
            "Les dynamiques de croissance divergent fortement entre pays : même indicateur, CAGR, volatilité "
            "et fréquence de chocs très différents selon le niveau de développement",
            "La convergence dépend du groupe — aucune convergence sigma ou beta universelle ne tient "
            "pour tous les ensembles de pays ; le résultat change avec le groupe défini",
            "Le clustering multi-approches (par indicateur, multi-indicateurs, basé ACP) produit "
            "des groupements qui se chevauchent mais restent distincts — aucune typologie macro universelle unique",
            "Le scoring composite révèle une dualité structure-dynamique : un pays peut être bien classé "
            "sur la position structurelle mais faible sur la dynamique de croissance — deux dimensions distinctes",
        ],
        "findings_link": [
            "Les profils commerciaux ne se projettent pas clairement sur une signature macro unique — "
            "les profils se chevauchent dans l'espace macro ; aucune signature nette n'émerge",
            "Certaines variables commerciales montrent des corrélats macro cohérents selon les méthodes ; "
            "d'autres résistent à tous les indicateurs testés — le lien est sélectif, pas global",
            "Les résultats sont bornés par le périmètre du jeu de données : 16 indicateurs WB généraux, "
            "médiane sur 10 ans — un ensemble d'indicateurs plus ciblé pourrait donner des conclusions différentes",
        ],
        "findings_footer": (
            "⚠ Constats valides sur ce périmètre de données : ~170 pays · 16 indicateurs macro WB généraux "
            "· médiane sur 10 ans 2015–2024."
        ),

        "nav_home": "Accueil",
        "nav_pipeline": "Pipeline",
        "nav_architecture": "Architecture",
        "nav_overview_scope": "Aperçu & périmètre",
        "nav_global_dynamics": "Dynamique mondiale",
        "nav_country_profile": "Profil pays",
        "nav_cross_country": "Comparaison pays",
        "nav_bilateral_analysis": "Analyse bilatérale",
        "nav_typology_scoring": "Typologie & scoring",
        "nav_deep_dive": "Analyse approfondie",
        "nav_advanced_analysis": "Analyse avancée",
        "nav_signals": "Signaux",
        "nav_strategy": "Stratégie",

        "section_etl": "Pipeline ETL",
        "section_trade": "Analyse commerciale",
        "section_indicators": "Indicateurs",
        "section_link": "Analyse de lien",
    },
}


def _s(key: str) -> str:
    return t(STRINGS, key)


# ════════════════════════════════════════════════════════════════
# HOME PAGE CONTENT
# ════════════════════════════════════════════════════════════════

def _home_page():
    inject_styles()

    # ── Header ──────────────────────────────────────────────────
    st.markdown(f'<div class="page-eyebrow">{_s("eyebrow")}</div>', unsafe_allow_html=True)
    st.title(_s("title"))
    st.caption(_s("caption"))
    thin_divider()

    # ── Description ─────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom:1.4rem;'>"
        f"<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>{_s('intro_1')}</p>"
        f"<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>{_s('intro_2')}</p>"
        f"<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>{_s('intro_3')}</p>"
        f"<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0;'>{_s('intro_4')}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    thin_divider()

    # ── KPIs ────────────────────────────────────────────────────
    cols = st.columns(4)
    for col, (label, value, color) in zip(cols, [
        (_s("kpi_countries"),  "187",  "teal"),
        (_s("kpi_years"),      "10",   "blue"),
        (_s("kpi_gold_rows"),  "544K", "amber"),
        (_s("kpi_indicators"), "16",   ""),
    ]):
        col.markdown(kpi(label, value, color), unsafe_allow_html=True)
    thin_divider()

    # ── Pipeline visual ─────────────────────────────────────────
    section_label(_s("pipeline_flow"))

    def _li(items: list[str]) -> str:
        return "".join(f"<li>{item}</li>" for item in items)

    st.markdown("""
<style>
.etl-wrap {
    display: flex; align-items: stretch;
    width: 100%; margin: 1rem 0 0 0; gap: 0;
}
.etl-node {
    flex: 1;
    background: linear-gradient(160deg, #1A2A42 0%, #162035 100%);
    border-radius: 10px; padding: 1rem 0.8rem 0.9rem; text-align: center;
}
.etl-icon  { font-size: 22px; display: block; margin-bottom: 0.35rem; }
.etl-label {
    font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; display: block; margin-bottom: 0.5rem;
}
.etl-tasks {
    font-size: 14px; color: #5A7898; line-height: 1.75;
    display: block; text-align: left; padding: 0 0.2rem;
}
.pip-arrow {
    display: flex; align-items: center;
    padding: 0 4px; color: #253550; font-size: 18px; flex-shrink: 0;
}
.layer-sep {
    display: flex; align-items: center; gap: 1rem;
    margin: 1.3rem 0 1.1rem 0;
    color: #304870; font-size: 14px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.layer-sep::before, .layer-sep::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, transparent, #304870, transparent);
}
.analytics-wrap {
    display: flex; gap: 12px;
    width: 100%; margin: 0 0 2rem 0; align-items: stretch;
}
.analytics-card {
    flex: 1;
    background: linear-gradient(160deg, #1A2A42 0%, #162035 100%);
    border-radius: 10px; padding: 1.1rem 1rem 1rem;
}
.an-icon  { font-size: 22px; display: block; margin-bottom: 0.35rem; }
.an-label {
    font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; display: block; margin-bottom: 0.7rem;
}
.an-list { list-style: none; padding: 0; margin: 0; }
.an-list li {
    font-size: 14.5px; color: #6888A8; line-height: 1.6;
    padding: 0.28rem 0;
    border-bottom: 1px solid rgba(48,72,112,0.3);
    display: flex; align-items: flex-start; gap: 0.45rem;
}
.an-list li:last-child { border-bottom: none; }
.an-list li::before { content: '›'; flex-shrink: 0; margin-top: 1px; }

.n-bronze { border-top: 3px solid #B87333; }
.n-bronze .etl-label { color: #D4924A; }
.n-silver { border-top: 3px solid #8899BB; }
.n-silver .etl-label { color: #9AAAC4; }
.n-gold   { border-top: 3px solid #D4A820; }
.n-gold   .etl-label { color: #D4A820; }

.n-trade  { border-top: 3px solid #10B981; }
.n-trade  .an-label { color: #10B981; }
.n-trade  .an-list li::before { color: #10B981; }
.n-indic  { border-top: 3px solid #60A5FA; }
.n-indic  .an-label { color: #60A5FA; }
.n-indic  .an-list li::before { color: #60A5FA; }
.n-link   { border-top: 3px solid #A78BFA; }
.n-link   .an-label { color: #A78BFA; }
.n-link   .an-list li::before { color: #A78BFA; }
</style>
""", unsafe_allow_html=True)

    st.markdown(f'<div class="layer-sep">{_s("etl_pipeline")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="etl-wrap">
  <div class="etl-node n-bronze">
    <span class="etl-icon">🗄️</span>
    <span class="etl-label">{_s("bronze_label")}</span>
    <span class="etl-tasks">{_s("bronze_tasks")}</span>
  </div>
  <div class="pip-arrow">▶</div>
  <div class="etl-node n-silver">
    <span class="etl-icon">⚙️</span>
    <span class="etl-label">{_s("silver_label")}</span>
    <span class="etl-tasks">{_s("silver_tasks")}</span>
  </div>
  <div class="pip-arrow">▶</div>
  <div class="etl-node n-gold">
    <span class="etl-icon">⭐</span>
    <span class="etl-label">{_s("gold_label")}</span>
    <span class="etl-tasks">{_s("gold_tasks")}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<div class="layer-sep">{_s("analytical_layers")}</div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="analytics-wrap">
  <div class="analytics-card n-trade">
    <span class="an-icon">📊</span>
    <span class="an-label">{_s("trade_analysis_label")}</span>
    <ul class="an-list">{_li(t(STRINGS, "trade_list"))}</ul>
  </div>
  <div class="analytics-card n-indic">
    <span class="an-icon">🔬</span>
    <span class="an-label">{_s("indicators_label")}</span>
    <ul class="an-list">{_li(t(STRINGS, "indicators_list"))}</ul>
  </div>
  <div class="analytics-card n-link">
    <span class="an-icon">🔗</span>
    <span class="an-label">{_s("link_analysis_label")}</span>
    <ul class="an-list">{_li(t(STRINGS, "link_list"))}</ul>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key Findings ────────────────────────────────────────────
    thin_divider()
    section_label(_s("key_findings"))

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#10B981;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        f"margin:0.8rem 0 0.5rem 0;'>{_s('trade_analysis_label')}</div>",
        unsafe_allow_html=True,
    )
    for text in t(STRINGS, "findings_trade"):
        finding(text)

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#60A5FA;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        f"margin:1.3rem 0 0.5rem 0;'>{_s('indicators_label')}</div>",
        unsafe_allow_html=True,
    )
    for text in t(STRINGS, "findings_indicators"):
        finding(text)

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#A78BFA;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        f"margin:1.3rem 0 0.5rem 0;'>{_s('link_analysis_label')}</div>",
        unsafe_allow_html=True,
    )
    for text in t(STRINGS, "findings_link"):
        finding(text)

    st.markdown(
        "<p style='font-size:11.5px;color:#4A6080;margin-top:1.2rem;font-style:italic;'>"
        f"{_s('findings_footer')}"
        "</p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
# NAVIGATION
# ════════════════════════════════════════════════════════════════

pg = st.navigation({
    "": [
        st.Page(_home_page, title=_s("nav_home"), icon="🏠", default=True),
    ],
    _s("section_etl"): [
        st.Page(str(_P / "_01_Pipeline.py"),      title=_s("nav_pipeline"),      icon="🔩"),
        st.Page(str(_P / "_02_Architecture.py"),  title=_s("nav_architecture"),  icon="🏗️"),
    ],
    _s("section_trade"): [
        st.Page(str(_P / "_03_Trade_Overview.py"),          title=_s("nav_overview_scope"),     icon="📋"),
        st.Page(str(_P / "_04_Global_Trade_Dynamic.py"),    title=_s("nav_global_dynamics"),    icon="📈"),
        st.Page(str(_P / "_05_Country_Trade_Profile.py"),   title=_s("nav_country_profile"),    icon="🏳️"),
        st.Page(str(_P / "_06_Cross_Country_Comparison.py"),title=_s("nav_cross_country"),      icon="⚖️"),
        st.Page(str(_P / "_07_Bilateral_Risck_Analysis.py"),title=_s("nav_bilateral_analysis"), icon="🤝"),
        st.Page(str(_P / "_08_Typology_Scoring.py"),        title=_s("nav_typology_scoring"),   icon="🎯"),
    ],
    _s("section_indicators"): [
        st.Page(str(_P / "_09_Indicators_Overview.py"),         title=_s("nav_overview_scope"),     icon="🔬"),
        st.Page(str(_P / "_10_Indicator_Deep_Dive.py"),         title=_s("nav_deep_dive"),          icon="🔍"),
        st.Page(str(_P / "_11_Country_Indicators_Profile.py"),  title=_s("nav_country_profile"),    icon="🌐"),
        st.Page(str(_P / "_12_Advanced_Analysis.py"),           title=_s("nav_advanced_analysis"),  icon="⚡"),
    ],
    _s("section_link"): [
        st.Page(str(_P / "_13_Link_Signals.py"),   title=_s("nav_signals"),  icon="📡"),
        st.Page(str(_P / "_14_Link_Strategy.py"),  title=_s("nav_strategy"), icon="🗺️"),
    ],
})

pg.run()
