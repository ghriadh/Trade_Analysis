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

st.set_page_config(
    page_title="Trade Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_P = Path(__file__).parent / "pages"

# ════════════════════════════════════════════════════════════════
# HOME PAGE CONTENT
# ════════════════════════════════════════════════════════════════

def _home_page():
    inject_styles()

    # ── Header ──────────────────────────────────────────────────
    st.markdown('<div class="page-eyebrow">Overview</div>', unsafe_allow_html=True)
    st.title("Trade Intelligence")
    st.caption("World Bank API · Medallion architecture · 187 countries · 10 years")
    thin_divider()

    # ── Description ─────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom:1.4rem;'>"

        "<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>"
        "Built on a Medallion ETL pipeline extracting <strong style='color:#EEF4FB;'>World Bank open data</strong> — "
        "async extraction with checkpointing, multi-layer cleaning and normalisation, "
        "through to a query-ready star schema."
        "</p>"

        "<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>"
        "The first analytical layer maps <strong style='color:#EEF4FB;'>international trade</strong>: "
        "global and country-level flows, bilateral analysis, structural typologies "
        "and composite scoring across multiple dimensions."
        "</p>"

        "<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0 0 0.7rem 0;'>"
        "The second layer examines <strong style='color:#EEF4FB;'>16 macro-economic indicators</strong> at multiple scales — "
        "from global concentration and stability to country-level growth dynamics, "
        "cross-indicator correlations, convergence analysis across country groups, "
        "multi-approach clustering, and composite scoring separating structural position "
        "from growth momentum."
        "</p>"

        "<p style='font-size:15px;color:#C4D5E8;line-height:1.75;margin:0;'>"
        "The two layers then <strong style='color:#EEF4FB;'>converge</strong>: looking for a macro fingerprint "
        "behind each trade profile, and testing whether macro indicators can predict trade variables — "
        "exploring the structure, direction, and limits of that link."
        "</p>"

        "</div>",
        unsafe_allow_html=True,
    )
    thin_divider()

    # ── KPIs ────────────────────────────────────────────────────
    cols = st.columns(4)
    for col, (label, value, color) in zip(cols, [
        ("Countries",  "187",  "teal"),
        ("Years",      "10",   "blue"),
        ("Gold rows",  "544K", "amber"),
        ("Indicators", "16",   ""),
    ]):
        col.markdown(kpi(label, value, color), unsafe_allow_html=True)
    thin_divider()

    # ── Pipeline visual ─────────────────────────────────────────
    section_label("Data Pipeline → Analytical Flow")

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

<div class="layer-sep">ETL Pipeline</div>
<div class="etl-wrap">
  <div class="etl-node n-bronze">
    <span class="etl-icon">🗄️</span>
    <span class="etl-label">Bronze</span>
    <span class="etl-tasks">
      World Bank API extraction<br>Async · thread pool<br>
      Rate-limit handling<br>Checkpointing · resumable
    </span>
  </div>
  <div class="pip-arrow">▶</div>
  <div class="etl-node n-silver">
    <span class="etl-icon">⚙️</span>
    <span class="etl-label">Silver</span>
    <span class="etl-tasks">
      Multi-layer cleaning<br>Type enforcement<br>
      Country validation<br>Schema normalisation
    </span>
  </div>
  <div class="pip-arrow">▶</div>
  <div class="etl-node n-gold">
    <span class="etl-icon">⭐</span>
    <span class="etl-label">Gold</span>
    <span class="etl-tasks">
      Star schema · 6 tables<br>316 named entities<br>
      Query-optimised joins<br>544K rows
    </span>
  </div>
</div>

<div class="layer-sep">▼ &nbsp; Analytical Layers</div>

<div class="analytics-wrap">
  <div class="analytics-card n-trade">
    <span class="an-icon">📊</span>
    <span class="an-label">Trade Analysis</span>
    <ul class="an-list">
      <li>Global trade dynamics — volumes, balance, concentration, volatility</li>
      <li>Country trade profile — trends, world share, growth rates, structural scoring</li>
      <li>Cross-country comparison — quadrant views, growth heatmap</li>
      <li>Bilateral analysis — HHI, BDIS dependency score, dependency map</li>
      <li>Typology & composite scoring — trade profiles, strategic quadrant, country scorecard</li>
    </ul>
  </div>
  <div class="analytics-card n-indic">
    <span class="an-icon">🔬</span>
    <span class="an-label">Indicators</span>
    <ul class="an-list">
      <li>Global snapshot — concentration, stability, volatility per indicator</li>
      <li>Country indicator profile — time series, z-score, CAGR, shock detection</li>
      <li>Cross-indicator correlations & PCA — structure in the macro space</li>
      <li>Convergence analysis — sigma & beta by country group</li>
      <li>Country clustering — per indicator, multi-indicator, PCA-based</li>
      <li>Composite scoring — Econ Health, Performance, Competitiveness</li>
    </ul>
  </div>
  <div class="analytics-card n-link">
    <span class="an-icon">🔗</span>
    <span class="an-label">Link Analysis</span>
    <ul class="an-list">
      <li>Correlation atlas — 44 trade variables × 16 macro indicators</li>
      <li>Trade intensity & derived metrics — elasticity, trade/GDP ratios</li>
      <li>Level vs dynamics — structural macro link vs idiosyncratic behavior</li>
      <li>Macro fingerprint — macro profile per trade typology</li>
      <li>Predictive analysis — macro → trade and trade → macro</li>
      <li>Country diagnostic — trade × macro positioning scorecard</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key Findings ────────────────────────────────────────────
    thin_divider()
    section_label("Key Findings")

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#10B981;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "margin:0.8rem 0 0.5rem 0;'>Trade Analysis</div>",
        unsafe_allow_html=True,
    )
    finding("World trade is structurally concentrated — top economies dominate global flows; "
            "the world aggregate masks deep country-level heterogeneity in volumes and dynamics")
    finding("4 structural trade profiles emerge clearly — Global Champions, Competitive Exporters, "
            "Import-Dependent, and Fragile-Isolated — each with a distinct bilateral pattern")
    finding("Bilateral dependencies are asymmetric: periphery countries show high dependency "
            "toward core economies; the reverse does not hold")
    finding("Country trade profiles are structurally stable over 10 years — "
            "typology is structural, not cyclical")
    finding("Composite trade scoring stratifies countries across 5 independent dimensions — "
            "volume, dynamism, bilateral exposure, resilience, and profile")

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#60A5FA;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "margin:1.3rem 0 0.5rem 0;'>Indicators</div>",
        unsafe_allow_html=True,
    )
    finding("Economic indicators are highly concentrated globally — a small number of countries "
            "dominates absolute levels; the world aggregate masks country-level heterogeneity")
    finding("Growth dynamics diverge sharply across countries: same indicator, very different "
            "CAGR, volatility and shock frequency depending on development stage")
    finding("Convergence is group-dependent — no universal sigma or beta convergence holds "
            "across all country sets; the result changes with the group defined")
    finding("Multi-approach clustering (per indicator, multi-indicator, PCA-based) produces "
            "overlapping but distinct groupings — no single universal macro typology emerges")
    finding("Composite scoring reveals a structural-momentum duality: a country can rank high "
            "on structural value position but low on growth momentum — two distinct dimensions")

    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#A78BFA;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "margin:1.3rem 0 0.5rem 0;'>Link Analysis</div>",
        unsafe_allow_html=True,
    )
    finding("Trade profiles do not map cleanly onto a single macro signature — "
            "profiles overlap in the macro space; no clean fingerprint emerges")
    finding("Some trade variables show consistent macro correlates across methods; "
            "others resist all indicators tested — the link is selective, not global")
    finding("Results are bounded by the dataset scope: 16 general WB indicators, "
            "10-year median — a narrower or more granular indicator set may yield different conclusions")

    st.markdown(
        "<p style='font-size:11.5px;color:#4A6080;margin-top:1.2rem;font-style:italic;'>"
        "⚠ Findings valid on this dataset scope: ~170 countries · 16 general WB macro indicators "
        "· 10-year median 2015–2024."
        "</p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
# NAVIGATION
# ════════════════════════════════════════════════════════════════

pg = st.navigation({
    "": [
        st.Page(_home_page, title="Home", icon="🏠", default=True),
    ],
    "ETL Pipeline": [
        st.Page(str(_P / "_01_Pipeline.py"),      title="Pipeline",      icon="🔩"),
        st.Page(str(_P / "_02_Architecture.py"),  title="Architecture",  icon="🏗️"),
    ],
    "Trade Analysis": [
        st.Page(str(_P / "_03_Trade_Overview.py"),          title="Overview & Scope",    icon="📋"),
        st.Page(str(_P / "_04_Global_Trade_Dynamic.py"),    title="Global Dynamics",     icon="📈"),
        st.Page(str(_P / "_05_Country_Trade_Profile.py"),   title="Country Profile",     icon="🏳️"),
        st.Page(str(_P / "_06_Cross_Country_Comparison.py"),title="Cross-Country",       icon="⚖️"),
        st.Page(str(_P / "_07_Bilateral_Risck_Analysis.py"),title="Bilateral Analysis",  icon="🤝"),
        st.Page(str(_P / "_08_Typology_Scoring.py"),        title="Typology & Scoring",  icon="🎯"),
    ],
    "Indicators": [
        st.Page(str(_P / "_09_Indicators_Overview.py"),         title="Overview & Scope",  icon="🔬"),
        st.Page(str(_P / "_10_Indicator_Deep_Dive.py"),         title="Deep Dive",         icon="🔍"),
        st.Page(str(_P / "_11_Country_Indicators_Profile.py"),  title="Country Profile",   icon="🌐"),
        st.Page(str(_P / "_12_Advanced_Analysis.py"),           title="Advanced Analysis", icon="⚡"),
    ],
    "Link Analysis": [
        st.Page(str(_P / "_13_Link_Signals.py"),   title="Signals",  icon="📡"),
        st.Page(str(_P / "_14_Link_Strategy.py"),  title="Strategy", icon="🗺️"),
    ],
})

pg.run()
