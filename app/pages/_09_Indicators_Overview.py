"""
Indicators Analysis — I1 · Overview & Methodology
Covers: B. Executive Summary + C. Data & Methodology + E. Global Snapshot
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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding

inject_styles()

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_conc  = pd.read_csv(IND_FILES / "concentration.csv")
    df_stab  = pd.read_csv(IND_FILES / "stability.csv")
    df_ind   = pd.read_csv(IND_FILES / "df_indicators.csv")
    df_cv    = pd.read_csv(IND_FILES / "df_cv.csv")
    return df_conc, df_stab, df_ind, df_cv

df_conc, df_stab, df_ind, df_cv = load_data()

INDICATORS = sorted(df_ind["indicator_name"].dropna().unique().tolist())
N_COUNTRIES = df_ind[df_ind["entity_type"] == "COUNTRY"]["country_name"].nunique()
N_YEARS     = df_ind["year"].nunique()

# ════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1F3050",
    plot_bgcolor="#162035",
    font=dict(family="IBM Plex Sans", color="#C4D5E8", size=13),
    title_font=dict(color="#EEF4FB", size=15),
    legend=dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="#304870", linecolor="#304870"),
    yaxis=dict(gridcolor="#304870", linecolor="#304870"),
)

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Indicators Analysis · Page 1 of 4</div>',
            unsafe_allow_html=True)
st.title("Overview & Methodology")
st.caption(
    "16 macroeconomic indicators · ~180 countries · 2015–2024 · World Bank"
)
thin_divider()

# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

_yr_min  = int(df_ind["year"].min())
_yr_max  = int(df_ind["year"].max())
_n_obs   = int(df_ind[df_ind["entity_type"] == "COUNTRY"]["value"].notna().sum())
_cov_pct = _n_obs / (N_COUNTRIES * len(INDICATORS) * N_YEARS) * 100

cols = st.columns(4)
for col, (label, value, color) in zip(cols, [
    ("Countries",  str(N_COUNTRIES),            "teal"),
    ("Indicators", str(len(INDICATORS)),         "blue"),
    ("Period",     f"{_yr_min}–{_yr_max}",       "amber"),
    ("Coverage",   f"{_cov_pct:.0f}%",           ""),
]):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# B. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

section_label("Executive Summary")

results = [
    ("Global",
     "Economic indicators are **highly concentrated** — a small number of countries "
     "dominates absolute levels. Cross-indicator stability varies widely between "
     "level and flow indicators."),
    ("World Profile",
     "The World aggregate **masks heterogeneity** — global trends are driven by a "
     "handful of large economies whose cycles dominate the average."),
    ("Country Profile",
     "Country-level profiles reveal **structural divergences invisible** at the global "
     "level — same indicator, very different trajectories depending on size and "
     "development stage."),
    ("Comparisons",
     "Cross-country comparisons on **index-100 and z-score** reveal convergence and "
     "divergence patterns that raw values cannot capture."),
]

for label, text in results:
    col1, col2 = st.columns([1, 3])
    col1.markdown(
        f"<div style='font-size:13px;font-weight:600;color:#FCD34D;"
        f"padding:.4rem 0;'>{label}</div>",
        unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size:14px;color:#C4D5E8;padding:.4rem 0;'>{text}</div>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #304870;margin:.1rem 0;'>",
                unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# C. ANALYTICAL FRAMEWORK
# ════════════════════════════════════════════════════════════════

section_label("Analytical Framework")

col_e, col_fg, col_hln = st.columns(3)

with col_e:
    st.markdown("""
    <div style="background:#231800;border:1.5px solid #D97706;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#FCD34D;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">E · Global Vision</div>
      <div style="font-size:13px;color:#FBBf24;line-height:1.9;opacity:.9;">
        <div>① Trend — global average</div>
        <div>② Distribution cross-country</div>
        <div>③ Stability (CV + rolling)</div>
        <div>④ Concentration (Gini)</div>
        <div>⑤ Growth dynamics</div>
        <div>⑥ Top & worst countries</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_fg:
    st.markdown("""
    <div style="background:#0F2240;border:1.5px solid #3B82F6;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">F–I · Deep Dive</div>
      <div style="font-size:13px;color:#BAD7F8;line-height:1.9;opacity:.9;">
        <div>F · World aggregate profile</div>
        <div>G · Country-level profile</div>
        <div>H · Cross-country comparison</div>
        <div>I · Growth dynamics</div>
        <div style="margin-top:.5rem;font-size:12px;color:#1E3A8A;">
          CAGR · volatility · shocks<br>
          index-100 · z-score
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_hln:
    st.markdown("""
    <div style="background:#082E18;border:1.5px solid #10B981;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#34D399;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">J–N · Advanced</div>
      <div style="font-size:13px;color:#6EE7B7;line-height:1.9;opacity:.9;">
        <div>J · Correlation & PCA</div>
        <div>K · Convergence (σ & β)</div>
        <div>L · Clustering (KMeans)</div>
        <div>M · Forecasting (H-W)</div>
        <div>N · Multi-indicator scoring</div>
        <div style="margin-top:.5rem;font-size:12px;color:#065F46;">
          Econ Health · Performance<br>
          Competitiveness
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# DATA SCOPE
# ════════════════════════════════════════════════════════════════

section_label("Data Scope")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;min-height:260px;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">Dataset</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:2.1;">
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Source</span>
             World Bank — Gold layer</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Period</span>
             2015–2024 (10 years)</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Countries</span>
             ~180 (COUNTRY entities only)</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Indicators</span>
             16 macroeconomic</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Coverage threshold</span>
             70% min non-null per indicator</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">Pipeline</span>
             WB API → ETL Bronze/Silver/Gold</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Indicator families
    st.markdown("""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;min-height:260px;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">Indicator Families</div>
      <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <div><span style="color:#10B981;font-weight:600;">Activity & Growth</span>
             &nbsp;— GDP · GDP per capita</div>
        <div><span style="color:#60A5FA;font-weight:600;">Prices & Monetary</span>
             &nbsp;— Inflation · Real interest rate</div>
        <div><span style="color:#F59E0B;font-weight:600;">Labor market</span>
             &nbsp;— Unemployment</div>
        <div><span style="color:#A78BFA;font-weight:600;">Public finances</span>
             &nbsp;— Expense (% GDP)</div>
        <div><span style="color:#F472B6;font-weight:600;">External sector</span>
             &nbsp;— Current account · Trade · Exports · Imports</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# E. GLOBAL SNAPSHOT — Concentration & Stability ranking
# ════════════════════════════════════════════════════════════════

section_label("E · Global Snapshot — Indicator Ranking")

col1, col2 = st.columns(2)

with col1:
    # Concentration ranking (Gini)
    df_c_sorted = df_conc.sort_values("mean_gini", ascending=True).copy()
    # Short names
    df_c_sorted["short_name"] = df_c_sorted["indicator_name"].apply(
        lambda x: x[:40] + "…" if len(x) > 40 else x)

    colors_c = ["#EF4444" if v > 0.8
                else "#F59E0B" if v > 0.6
                else "#60A5FA" if v > 0.4
                else "#10B981"
                for v in df_c_sorted["mean_gini"]]

    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        x=df_c_sorted["mean_gini"],
        y=df_c_sorted["short_name"],
        orientation="h",
        marker_color=colors_c,
        text=df_c_sorted["mean_gini"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        customdata=df_c_sorted[["trend_gini","last_gini"]],
        hovertemplate="<b>%{y}</b><br>Mean Gini: %{x:.3f}<br>"
                      "Last Gini: %{customdata[1]:.3f}<br>"
                      "Trend: %{customdata[0]:+.3f}<extra></extra>",
        opacity=0.85,
    ))
    fig_conc.add_vline(x=0.8, line_dash="dash", line_color="#EF4444",
                       annotation_text="High (>0.8)",
                       annotation_font_color="#EF4444")
    fig_conc.update_layout(**PLOTLY_LAYOUT,
        title="Concentration Ranking — Mean Gini (higher = more concentrated)",
        xaxis_title="Mean Gini",
        height=520,
    )
    st.plotly_chart(fig_conc, width='stretch')

with col2:
    # Stability ranking (CV — lower = more stable)
    df_s_sorted = df_stab.sort_values("global_cv", ascending=True).copy()
    df_s_sorted["short_name"] = df_s_sorted["indicator_name"].apply(
        lambda x: x[:40] + "…" if len(x) > 40 else x)

    colors_s = ["#10B981" if v < 10
                else "#60A5FA" if v < 20
                else "#F59E0B" if v < 40
                else "#EF4444"
                for v in df_s_sorted["global_cv"]]

    fig_stab = go.Figure()
    fig_stab.add_trace(go.Bar(
        x=df_s_sorted["global_cv"],
        y=df_s_sorted["short_name"],
        orientation="h",
        marker_color=colors_s,
        text=df_s_sorted["global_cv"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Global CV: %{x:.1f}%<extra></extra>",
    ))
    fig_stab.update_layout(**PLOTLY_LAYOUT,
        title="Stability Ranking — Global CV% (lower = more stable)",
        xaxis_title="Coefficient of Variation (%)",
        height=520,
    )
    st.plotly_chart(fig_stab, width='stretch')

with st.expander("Key findings"):
    most_conc  = df_conc.nlargest(1,"mean_gini")["indicator_name"].iloc[0]
    most_stab  = df_stab.nsmallest(1,"global_cv")["indicator_name"].iloc[0]
    most_volat = df_stab.nlargest(1,"global_cv")["indicator_name"].iloc[0]
    finding(f"<strong>Most concentrated indicator</strong>: {most_conc} "
            f"— a handful of countries dominates absolute levels")
    finding(f"<strong>Most stable indicator</strong>: {most_stab} "
            f"— low cross-country variation over the decade")
    finding(f"<strong>Most volatile indicator</strong>: {most_volat} "
            f"— high sensitivity to macro shocks (COVID, commodity cycles)")

thin_divider()

# ════════════════════════════════════════════════════════════════
# DESIGN CHOICES
# ════════════════════════════════════════════════════════════════

section_label("Design Choices")

finding(
    "<strong>Parametric design</strong>: all analyses run on a single "
    "<code>indicator</code> parameter — changing it re-runs the full section "
    "on any indicator without code changes"
)
finding(
    "<strong>COUNTRY entities only</strong>: World and regional aggregates "
    "are excluded from cross-country analyses (correlation, PCA, clustering, "
    "convergence) to preserve statistical validity"
)
finding(
    "<strong>Double reading in scoring (Section N)</strong>: each score is "
    "computed twice — on indicator <em>values</em> (current position) and on "
    "<em>growth scores</em> (momentum). Same indicators, same weights, different data — "
    "this distinguishes structural position from directional trajectory"
)
finding(
    "<strong>Per-country median for full-period aggregation</strong>: "
    "robust to the 2020 COVID outlier — consistent with the Trade notebook strategy"
)