"""
Trade Analysis — T1 · Overview & Methodology
Covers: B. Executive Summary + C. Data & Methodology
Static page — no selectors needed
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "App"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project paths ─────────────────────────────────────────────
FIGURES_DIR  = ROOT / "Analysis" / "Figures"
TRADE_FIGS   = FIGURES_DIR / "Trade"
SCORE_GUIDE  = FIGURES_DIR / "score_reference_guide.png"
GOLD_DIR     = ROOT / "data" / "gold"

import pandas as pd
import streamlit as st
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding

inject_styles()

TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"

@st.cache_data
def load_overview_stats():
    df_c  = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_fp = pd.read_csv(TRADE_FILES / "df_full_periode.csv")
    df_st = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_cn = pd.read_csv(TRADE_FILES / "df_concentration.csv")
    df_fin         = pd.read_csv(TRADE_FILES / "df_final.csv")
    n_total        = len(df_c)
    n_champion     = int((df_c["combined_profile"] == "Global Champion").sum())
    pct_champion   = n_champion / n_total * 100
    n_critical     = int((df_fp["profile_full_periode"] == "Critical Dependency").sum())
    pct_critical   = n_critical / n_total * 100
    n_gl_leader    = int((df_st["meta_profile"] == "Global Leader").sum())
    top10_min      = df_cn["top10_share"].min()
    top10_max      = df_cn["top10_share"].max()
    min_corr_year  = int(df_cn.dropna(subset=["spearman_correlation"])
                           .loc[df_cn["spearman_correlation"].idxmin(), "year"])
    n_bilateral_2024 = int(df_fin[df_fin["year"] == 2024]["country_name"].nunique())
    return dict(
        n_total=n_total, n_champion=n_champion, pct_champion=pct_champion,
        n_critical=n_critical, pct_critical=pct_critical,
        n_gl_leader=n_gl_leader,
        top10_min=top10_min, top10_max=top10_max,
        min_corr_year=min_corr_year,
        n_bilateral_2024=n_bilateral_2024,
    )

_s = load_overview_stats()


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Trade Analysis · Page 1 of 6</div>', unsafe_allow_html=True)
st.title("Overview & Methodology")
st.caption(
    f"{_s['n_total']} countries · 10 years · 3 analytical dimensions · "
    "UN Comtrade · 2015–2024"
)
thin_divider()


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

cols = st.columns(4)
kpis = [
    ("Countries",   str(_s["n_total"]), "teal"),
    ("Years",       "10",               "blue"),
    ("Dimensions",  "3",                "amber"),
    ("Gold rows",   "544K",             ""),
]
for col, (label, value, color) in zip(cols, kpis):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# B. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

section_label("Executive Summary")

# Key findings — native Streamlit layout
results = [
    ("Global Trade",   f"Top 10 countries control **{_s['top10_min']:.0f}–{_s['top10_max']:.0f}% of global exports** — highly concentrated and structurally stable over the decade"),
    ("COVID Impact",   f"2020 did not reshape bilateral architectures — rank correlations of country positions **remained above 0.995** throughout 2015–2024, with maximum structural shift in **{_s['min_corr_year']}**"),
    ("Trade Power",    f"**{_s['n_gl_leader']} countries** hold a Global Leader profile in trade structure — led by China, USA, Germany, Japan"),
    ("Bilateral Risk", f"**{_s['pct_critical']:.0f}% of countries** ({_s['n_critical']}) face Critical bilateral dependency — predominantly small commodity-dependent economies"),
    ("Combined Score", f"**{_s['n_champion']} Global Champions** combine top trade power with strong bilateral resilience"),
]

for label, text in results:
    col1, col2 = st.columns([1, 3])
    col1.markdown(f"<div style='font-size:13px;font-weight:600;color:#FCD34D;padding:.4rem 0;'>{label}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='font-size:14px;color:#C4D5E8;padding:.4rem 0;'>{text}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #304870;margin:.1rem 0;'>", unsafe_allow_html=True)

# Main conclusion
st.markdown(f"""
<div style="background:#1A2C48;border:1px solid #304870;
            border-left:4px solid #10B981;border-radius:0 12px 12px 0;
            padding:1rem 1.25rem;margin-bottom:.5rem;">
  <div style="font-size:13px;font-weight:600;color:#10B981;
              text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;">
    Main Conclusion
  </div>
  <div style="font-size:15px;color:#C4D5E8;line-height:1.8;font-style:italic;">
    Trade strength and trade resilience are complementary but independent.
    The most strategically valuable countries are those that combine both —
    a rare profile held by <strong style="color:#34D399;">fewer than {_s['pct_champion']:.0f}%</strong>
    of the world's trading nations.
  </div>
</div>
""", unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# C. DATA & METHODOLOGY
# ════════════════════════════════════════════════════════════════

section_label("Data Scope")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">Dataset</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:2;">
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Source</span>
             UN Comtrade — annual trade flows</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Period</span>
             2015–2024 (10 years)</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Countries</span>
             {_s['n_total']} (aggregate) / {_s['n_bilateral_2024']} (bilateral 2024)</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Flows</span>
             Exports (X), Imports (M), Total Trade</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Granularity</span>
             Annual · aggregate (all products)</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">Pipeline</span>
             Comtrade API → ETL Bronze/Silver/Gold</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">Key limitation</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:1.8;margin-bottom:.8rem;">
        Bilateral data for <strong style="color:#EEF4FB;">2024 covers 133 countries</strong>
        due to UN Comtrade reporting lags.
        Full-period bilateral analysis uses all available years per country (2015–2023/2024).
      </div>
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin:.8rem 0 .5rem;">Gold layer tables</div>
      <div style="font-size:13px;color:#A8C0D8;line-height:1.9;
                  font-family:'IBM Plex Mono',monospace;">
        <span style="color:#34D399;">trade_data</span>
        &nbsp;·&nbsp;544K rows · 188 reporters<br>
        <span style="color:#34D399;">entities</span>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp;312 entities (262 countries + 50 aggregates)<br>
        <span style="color:#34D399;">trade_flow</span>
        &nbsp;&nbsp;&nbsp;·&nbsp;M / X / RM / RX
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# ANALYTICAL FRAMEWORK — 3 sections
# ════════════════════════════════════════════════════════════════

section_label("Analytical Framework")

st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;
            gap:1rem;align-items:stretch;">

  <div style="background:#231800;border:1.5px solid #D97706;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#FCD34D;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      F · Global Trade Structure</div>
    <div style="font-size:13px;color:#FBBf24;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">① Global system</span><br>
        <span style="opacity:.8;">Concentration · evolution · volatility</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">② Country profile</span><br>
        <span style="opacity:.8;">Export share · coverage · growth</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">③ Scoring</span><br>
        <span style="opacity:.8;">Strong · Good · Moderate · Weak</span>
      </div>
      <div>
        <span style="color:#FCD34D;font-weight:600;">④ Quadrants</span><br>
        <span style="opacity:.8;">Static · Dynamic · Strategic</span>
      </div>
    </div>
  </div>

  <div style="background:#0F2240;border:1.5px solid #3B82F6;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#93C5FD;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      G · Bilateral Trade Analysis</div>
    <div style="font-size:13px;color:#BAD7F8;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#93C5FD;font-weight:600;">① Partner map</span><br>
        <span style="opacity:.8;">Sankey · top partners · HHI</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#93C5FD;font-weight:600;">② Dependency</span><br>
        <span style="opacity:.8;">BDI · critical share · asymmetry</span>
      </div>
      <div>
        <span style="color:#93C5FD;font-weight:600;">③ Scoring</span><br>
        <span style="opacity:.8;">Robust · Stable but Concentrated<br>Vulnerable · Critical Dependency</span>
      </div>
    </div>
  </div>

  <div style="background:#082E18;border:1.5px solid #10B981;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#34D399;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      H · Combined Scoring</div>
    <div style="font-size:13px;color:#6EE7B7;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#34D399;font-weight:600;">Formula</span><br>
        <span style="opacity:.8;">0.70 × F rank + 0.30 × G rank</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#34D399;font-weight:600;">4 profiles</span><br>
        <span style="opacity:.8;">Global Champion · Competitive<br>Vulnerable · Fragile</span>
      </div>
      <div>
        <span style="color:#34D399;font-weight:600;">4 quadrants</span><br>
        <span style="opacity:.8;">Strong & Resilient · Strong but Exposed<br>Diversified Minor · Weak & Vulnerable</span>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# DESIGN CHOICES
# ════════════════════════════════════════════════════════════════

section_label("Design Choices")
finding(
    "<strong>Median over mean</strong> for full-period aggregation — "
    "robust to the 2020 COVID outlier across all indicators"
)
finding(
    "<strong>Rank normalization</strong> in H — ensures equal contribution "
    "from F and G despite their different score distributions"
)
finding(
    "<strong>Export-centric scoring</strong> in F — trade power is measured by "
    "competitive export capacity, not import volume"
)
finding(
    f"<strong>Aggregate-level granularity</strong> — product disaggregation requires one API call "
    f"per reporter × year combination (~{_s['n_total']} countries × 10 years per product code). "
    f"Aggregate flows provide complete, consistent {_s['n_total']}-country coverage "
    f"that product-level data cannot match at this scale"
)

thin_divider()


# ════════════════════════════════════════════════════════════════
# SCORE READING GUIDE
# ════════════════════════════════════════════════════════════════

section_label("Score Reading Guide")
for candidate in [
    SCORE_GUIDE,
    ROOT / "Analysis" / "Figures" / "score_reference_guide.png",
]:
    if candidate.exists():
        st.image(str(candidate), width='stretch')
        break
else:
    st.markdown("""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem;text-align:center;color:#6890B0;font-size:13px;">
      Score reference guide image not found<br>
      <span style="font-size:11px;font-family:'IBM Plex Mono',monospace;">
        Expected: Analysis/Figures/score_reference_guide.png
      </span>
    </div>
    """, unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# NAVIGATION HINT
# ════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='font-size:13px;color:#6890B0;margin-bottom:.4rem;'>Continue to →</div>",
    unsafe_allow_html=True,
)
_nav_cols = st.columns(5)
_nav_pages = [
    ("pages/_04_Global_Trade_Dynamic.py",     "Global Trade Dynamics"),
    ("pages/_05_Country_Trade_Profile.py",     "Country Trade Profile"),
    ("pages/_06_Cross_Country_Comparison.py",  "Cross-Country Comparison"),
    ("pages/_07_Bilateral_Risck_Analysis.py",  "Bilateral Risk"),
    ("pages/_08_Typology_Scoring.py",          "Typology & Scoring"),
]
for _col, (_path, _label) in zip(_nav_cols, _nav_pages):
    _col.page_link(_path, label=_label)