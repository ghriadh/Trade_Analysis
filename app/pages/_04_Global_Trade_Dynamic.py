"""
Trade Analysis — T2 · Global Trade Dynamics
Covers: F.2 — Global Trade System
- Evolution (total, exports, imports)
- Top trading countries
- Trade balance (butterfly chart)
- Concentration (Top N shares + HHI)
- Hierarchy stability (Spearman)
- Volatility map (HTML) + volatility matrix
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, format_value

inject_styles()

# ── Paths ─────────────────────────────────────────────────────
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
TRADE_FIGS  = ROOT / "Analysis" / "Figures" / "Trade"


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_trade   = pd.read_csv(TRADE_FILES / "df_trade.csv")
    df_dynamic = pd.read_csv(TRADE_FILES / "df_trade_dynamic.csv")
    df_conc    = pd.read_csv(TRADE_FILES / "df_concentration.csv")
    return df_trade, df_dynamic, df_conc

df_trade, df_dynamic, df_conc = load_data()

YEARS = sorted(df_trade["year"].unique())
YEAR_MAX = int(YEARS[-1])


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
PLOTLY_LAYOUT_COMPACT = {k: v for k, v in PLOTLY_LAYOUT.items()
                         if k not in ("margin","yaxis","xaxis","legend",
                                      "paper_bgcolor","plot_bgcolor")}

def _fmt(x):
    return format_value(x)


COLORS = {
    "export":  "#F59E0B",
    "import":  "#60A5FA",
    "total":   "#10B981",
    "balance": "#A78BFA",
}


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Trade Analysis · Page 2 of 6</div>', unsafe_allow_html=True)
st.title("Global Trade Dynamics")
st.caption("F.2 — Global trade system · evolution · concentration · stability · volatility")
thin_divider()


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

total_exports = df_trade[df_trade["year"] == YEAR_MAX]["X"].sum()
total_imports = df_trade[df_trade["year"] == YEAR_MAX]["M"].sum()
n_countries   = df_trade["country_name"].nunique()
top10_share   = df_conc[df_conc["year"] == YEAR_MAX]["top10_share"].values[0] \
                if "top10_share" in df_conc.columns else 0

cols = st.columns(4)
cols[0].markdown(kpi("World Exports",  f"${total_exports/1e12:.1f}T", "amber"), unsafe_allow_html=True)
cols[1].markdown(kpi("World Imports",  f"${total_imports/1e12:.1f}T", "blue"),  unsafe_allow_html=True)
cols[2].markdown(kpi("Countries",      str(n_countries),              "teal"),  unsafe_allow_html=True)
cols[3].markdown(kpi("Top 10 share",   f"{top10_share:.1f}%",         ""),      unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.1 — GLOBAL TRADE EVOLUTION
# ════════════════════════════════════════════════════════════════

section_label("F.2.1 · Global Trade Evolution")

metric_sel = st.radio(
    "Metric",
    options=["Total Trade", "Imports/Exports", "Growth YoY %"],
    horizontal=True,
    key="evo_metric",
)

trends = df_trade.groupby("year")[["X", "M", "total_trade"]].sum().reset_index()
trends["yoy_X"] = trends["X"].pct_change() * 100
trends["yoy_M"] = trends["M"].pct_change() * 100

if metric_sel == "Total Trade":
    fig_evo = go.Figure()
    trends["_fmt_total"] = trends["total_trade"].apply(lambda v: f"${_fmt(v)}")
    fig_evo.add_trace(go.Scatter(
        x=trends["year"], y=trends["total_trade"] / 1e12,
        mode="lines+markers", name="Total Trade",
        line=dict(color=COLORS["total"], width=2.5),
        marker=dict(size=7, color=COLORS["total"]),
        customdata=trends[["_fmt_total"]],
        hovertemplate="<b>%{x}</b><br>Total Trade: %{customdata[0]}<extra></extra>",
    ))
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title="Global Total Trade Volume (USD Trillion)",
        yaxis_title="USD Trillion")

elif metric_sel == "Exports":
    fig_evo = go.Figure()
    trends["_fmt_X"] = trends["X"].apply(lambda v: f"${_fmt(v)}")
    trends["_fmt_M"] = trends["M"].apply(lambda v: f"${_fmt(v)}")
    fig_evo.add_trace(go.Scatter(
        x=trends["year"], y=trends["X"] / 1e12,
        mode="lines+markers", name="Exports",
        line=dict(color=COLORS["export"], width=2.5),
        marker=dict(size=7),
        customdata=trends[["_fmt_X"]],
        hovertemplate="<b>%{x}</b><br>Exports: %{customdata[0]}<extra></extra>",
    ))
    fig_evo.add_trace(go.Scatter(
        x=trends["year"], y=trends["M"] / 1e12,
        mode="lines+markers", name="Imports",
        line=dict(color=COLORS["import"], width=2.5, dash="dash"),
        marker=dict(size=7),
        customdata=trends[["_fmt_M"]],
        hovertemplate="<b>%{x}</b><br>Imports: %{customdata[0]}<extra></extra>",
    ))
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title="Global Exports & Imports Volume (USD Trillion)",
        yaxis_title="USD Trillion")

else:  # YoY %
    fig_evo = go.Figure()
    fig_evo.add_trace(go.Bar(
        x=trends["year"], y=trends["yoy_X"],
        name="Export YoY %", marker_color=COLORS["export"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Export YoY: %{y:.1f}%<extra></extra>",
    ))
    fig_evo.add_trace(go.Bar(
        x=trends["year"], y=trends["yoy_M"],
        name="Import YoY %", marker_color=COLORS["import"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Import YoY: %{y:.1f}%<extra></extra>",
    ))
    fig_evo.add_hline(y=0, line_color="#304870", line_width=1)
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title="Year-over-Year Growth Rate (%)",
        yaxis_title="%", barmode="group")

st.plotly_chart(fig_evo, width='stretch')

with st.expander("Key findings"):
    finding("World trade contracted sharply in <strong>2020 (COVID)</strong> — exports fell ~7–10% — then recovered strongly in 2021–2022")
    finding("Growth stabilized from 2023 onward, suggesting a post-pandemic normalization rather than structural expansion")

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.2 — TOP TRADING COUNTRIES
# ════════════════════════════════════════════════════════════════

section_label("F.2.2 · Top Trading Countries")

col1, col2 = st.columns([1, 3])
with col1:
    year_top = st.selectbox("Year", YEARS[::-1], index=0, key="top_year")
    metric_top = st.selectbox("Metric", ["Exports", "Imports", "Total Trade"], key="top_metric")
    n_top = st.selectbox("Top N", [10, 15, 20], index=0, key="top_n")

metric_map = {"Exports": "X", "Imports": "M", "Total Trade": "total_trade"}
col_use = metric_map[metric_top]

df_year = df_trade[df_trade["year"] == year_top].copy()
df_agg  = df_year.groupby("country_name")[col_use].sum().nlargest(n_top).sort_values().reset_index()
df_agg["share"] = df_agg[col_use] / df_year[col_use].sum() * 100
df_agg["cum_share"] = df_agg["share"][::-1].cumsum()[::-1]

with col2:
    fig_top = go.Figure()
    fig_top.add_trace(go.Bar(
        x=df_agg[col_use] / 1e9,
        y=df_agg["country_name"],
        orientation="h",
        marker=dict(
            color=df_agg[col_use],
            colorscale=[[0, "#1F3050"], [1, COLORS["export"]]],
        ),
        text=df_agg["share"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        name=metric_top,
    ))
    fig_top.update_layout(**PLOTLY_LAYOUT,
        title=f"Top {n_top} Countries by {metric_top} — {year_top}",
        xaxis_title="USD Billion",
        height=max(400, n_top * 38),
    )
    st.plotly_chart(fig_top, width='stretch')

with st.expander("Key findings"):
    finding("The <strong>top 3 exporters</strong> (China, Germany, USA) consistently account for 25–30% of world exports")
    finding("The ranking is remarkably stable across years — trade hierarchy changes very slowly")

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.3 — TRADE BALANCE
# ════════════════════════════════════════════════════════════════

section_label("F.2.3 · Trade Balance — Surplus vs Deficit")

col1, col2 = st.columns([1, 3])
with col1:
    year_bal = st.selectbox(
        "Year (or Full Period Median)",
        ["Full Period"] + list(YEARS[::-1]),
        index=0, key="bal_year"
    )
    n_bal = st.selectbox("Top N each side", [5, 10], index=0, key="bal_n")

if year_bal == "Full Period":
    df_bal = df_trade.groupby("country_name")["trade_balance"].median()
    title_suffix = "Full Period (Median)"
else:
    df_bal = df_trade[df_trade["year"] == int(year_bal)].groupby("country_name")["trade_balance"].sum()
    title_suffix = str(year_bal)

surplus = df_bal.nlargest(n_bal).reset_index().sort_values("trade_balance")
deficit = df_bal.nsmallest(n_bal).reset_index().sort_values("trade_balance", ascending=False)

with col2:
    fig_bal = go.Figure()
    surplus["_fmt"] = surplus["trade_balance"].apply(lambda v: f"${_fmt(v)}")
    deficit["_fmt"] = deficit["trade_balance"].apply(lambda v: f"${_fmt(v)}")
    fig_bal.add_trace(go.Bar(
        x=surplus["trade_balance"] / 1e9,
        y=surplus["country_name"],
        orientation="h",
        marker_color=COLORS["export"],
        name="Surplus",
        opacity=0.85,
        customdata=surplus[["_fmt"]],
        hovertemplate="<b>%{y}</b><br>Surplus: %{customdata[0]}<extra></extra>",
    ))
    fig_bal.add_trace(go.Bar(
        x=deficit["trade_balance"] / 1e9,
        y=deficit["country_name"],
        orientation="h",
        marker_color=COLORS["import"],
        name="Deficit",
        opacity=0.85,
        customdata=deficit[["_fmt"]],
        hovertemplate="<b>%{y}</b><br>Deficit: %{customdata[0]}<extra></extra>",
    ))
    fig_bal.add_vline(x=0, line_color="#EEF4FB", line_width=1)
    fig_bal.update_layout(**PLOTLY_LAYOUT,
        title=f"Trade Balance — Top Surplus & Deficit — {title_suffix}",
        xaxis_title="USD Billion",
        height=450,
    )
    st.plotly_chart(fig_bal, width='stretch')

with st.expander("Key findings"):
    finding("China and Germany are the <strong>dominant surplus nations</strong> — persistent structural exporters")
    finding("The USA maintains the world's largest trade deficit — structural importer financing through capital flows")

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.4 — CONCENTRATION
# ════════════════════════════════════════════════════════════════

section_label("F.2.4 · Trade Concentration & Hierarchy Stability")

col1, col2 = st.columns(2)

with col1:
    fig_conc = go.Figure()
    for col, color, label in [
        ("top3_share",  "#EF4444", "Top 3"),
        ("top5_share",  "#F59E0B", "Top 5"),
        ("top10_share", "#10B981", "Top 10"),
    ]:
        if col in df_conc.columns:
            fig_conc.add_trace(go.Scatter(
                x=df_conc["year"],
                y=df_conc[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>%{{x}}</b><br>{label} share: %{{y:.1f}}%<extra></extra>",
            ))
    fig_conc.update_layout(**PLOTLY_LAYOUT,
        title="Concentration Ratios — Export Share (%)",
        yaxis_title="Share (%)",
    )
    st.plotly_chart(fig_conc, width='stretch')

with col2:
    if "hhi_global" in df_conc.columns:
        hhi_col = "hhi_global"
        fig_hhi = go.Figure()
        fig_hhi.add_trace(go.Scatter(
            x=df_conc["year"], y=df_conc[hhi_col],
            mode="lines+markers", name="HHI",
            line=dict(color="#A78BFA", width=2.5),
            marker=dict(size=7, color="#A78BFA"),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.1)",
            hovertemplate="<b>%{x}</b><br>HHI: %{y:.4f}<extra></extra>",
        ))
        fig_hhi.update_layout(**PLOTLY_LAYOUT,
            title="Herfindahl-Hirschman Index (HHI)",
            yaxis_title="HHI",
        )
        st.plotly_chart(fig_hhi, width='stretch')

    if "spearman_correlation" in df_conc.columns:
        df_spear = df_conc.dropna(subset=["spearman_correlation"])
        fig_stab = go.Figure()
        fig_stab.add_trace(go.Scatter(
            x=df_spear["year"], y=df_spear["spearman_correlation"],
            mode="lines+markers", name="Spearman ρ",
            line=dict(color="#60A5FA", width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Spearman ρ: %{y:.4f}<extra></extra>",
        ))
        fig_stab.add_hline(y=0.95, line_dash="dash", line_color="#304870",
                           annotation_text="ρ = 0.95")
        fig_stab.update_layout(**PLOTLY_LAYOUT,
            title="Hierarchy Stability (Spearman Rank Correlation)",
            yaxis_title="Spearman ρ",
        )
        st.plotly_chart(fig_stab, width='stretch')

with st.expander("Key findings"):
    finding("Top 10 countries control <strong>55–60% of world exports</strong> — the global trade system is highly concentrated")
    finding("Concentration ratios are <strong>structurally frozen</strong> — no significant redistribution over the decade despite COVID")
    finding("Spearman rank correlation stays above <strong>0.95</strong> — trade hierarchy is extremely stable year to year")

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.5 — VOLATILITY
# ════════════════════════════════════════════════════════════════

section_label("F.2.5 · Trade Volatility")

tab1, tab2 = st.tabs(["🗺️  World Map", "📊  Volatility Matrix"])

with tab1:
    map_path = TRADE_FIGS / "F_2_6_1_world_map_volatility.html"
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=550, scrolling=False)
    else:
        if "volatility_export" in df_dynamic.columns:
            # Cap color scale at 95th percentile to avoid outlier distortion
            cap_val = df_dynamic["volatility_export"].quantile(0.95)
            fig_map = px.choropleth(
                df_dynamic,
                locations="country_name",
                locationmode="country names",
                color="volatility_export",
                hover_name="country_name",
                hover_data={"volatility_export": ":.2f", "volatility_import": ":.2f"},
                color_continuous_scale="Reds",
                range_color=[0, cap_val],
                title="World Trade Volatility — Export Volatility",
            )
            fig_map.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig_map, width='stretch')
            st.caption(
                f"⚠️ Color scale capped at 95th percentile ({cap_val:.1f}) for readability. "
                f"Extreme outliers (Yemen: 835, Sierra Leone: 241) are shown in max color."
            )

with tab2:
    if "volatility_export" in df_dynamic.columns and "volatility_import" in df_dynamic.columns:
        df_vol = df_dynamic[[
            "country_name", "volatility_export", "volatility_import", "total_export"
        ]].dropna()

        # Cap at 150 — only ~4 extreme outliers excluded (visible on world map)
        cap_x = 150
        cap_y = 150
        df_vol_plot = df_vol[
            (df_vol["volatility_export"] <= cap_x) &
            (df_vol["volatility_import"] <= cap_y)
        ].copy()
        n_excluded = len(df_vol) - len(df_vol_plot)

        median_exp = df_vol_plot["volatility_export"].median()
        median_imp = df_vol_plot["volatility_import"].median()

        df_vol_plot["quadrant"] = df_vol_plot.apply(lambda r:
            "High-High" if r["volatility_export"] >= median_exp and r["volatility_import"] >= median_imp
            else "High-Low" if r["volatility_export"] >= median_exp
            else "Low-High" if r["volatility_import"] >= median_imp
            else "Low-Low", axis=1
        )

        color_map = {
            "High-High": "#EF4444",
            "High-Low":  "#F59E0B",
            "Low-High":  "#60A5FA",
            "Low-Low":   "#10B981",
        }

        # Default highlights — top 10 exporters
        _vol_defaults = [cc for cc in
                         ["USA","China","Japan","Canada","Brazil","Qatar","Nigeria","Egypt"]
                         if cc in df_vol_plot["country_name"].values]
        highlights = st.multiselect(
            "Highlight countries",
            options=sorted(df_vol_plot["country_name"].tolist()),
            default=_vol_defaults,
            key="vol_highlights",
        )

        # Size proportional to trade volume
        size_max = df_vol_plot["total_export"].max()
        # Uniform size: highlights larger, others smaller
        df_vol_plot["marker_size"] = df_vol_plot["country_name"].apply(
            lambda x: 12 if x in highlights else 6
        )

        # Labels only for highlighted countries
        df_vol_plot["label"] = df_vol_plot["country_name"].apply(
            lambda x: x if x in highlights else ""
        )

        fig_vmat = go.Figure()

        for quadrant, color in color_map.items():
            df_q = df_vol_plot[df_vol_plot["quadrant"] == quadrant]
            if df_q.empty:
                continue
            fig_vmat.add_trace(go.Scatter(
                x=df_q["volatility_export"],
                y=df_q["volatility_import"],
                mode="markers+text",
                name=quadrant,
                marker=dict(
                    color=color,
                    size=df_q["marker_size"],
                    opacity=0.75,
                    line=dict(width=0.5, color="#304870"),
                ),
                text=df_q["label"],
                textposition="top center",
                textfont=dict(size=11, color="#EEF4FB"),
                customdata=df_q[["country_name", "total_export"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Export volatility: %{x:.1f}<br>"
                    "Import volatility: %{y:.1f}<br>"
                    "Total exports: $%{customdata[1]:.2e}<extra></extra>"
                    "<extra></extra>"
                ),
            ))

        fig_vmat.add_vline(x=median_exp, line_dash="dash", line_color="#304870",
                           annotation_text=f"Median {median_exp:.1f}")
        fig_vmat.add_hline(y=median_imp, line_dash="dash", line_color="#304870",
                           annotation_text=f"Median {median_imp:.1f}")
        # Remove size_max line (no longer needed)
        fig_vmat.update_layout(**PLOTLY_LAYOUT_COMPACT,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title="Volatility Matrix — Export vs Import Volatility",
            xaxis=dict(title="Export Volatility", range=[-10, 60],
                       gridcolor="#304870", linecolor="#304870"),
            yaxis=dict(title="Import Volatility", range=[0, 40],
                       gridcolor="#304870", linecolor="#304870"),
            legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=560,
        )
        st.plotly_chart(fig_vmat, width='stretch')

        if n_excluded > 0:
            st.caption(
                f"⚠️ {n_excluded} extreme outlier(s) excluded from scatter "
                f"(volatility > {cap_x}). Visible on the world map above."
            )

with st.expander("Key findings"):
    finding("Most countries cluster in the <strong>Low-Low quadrant</strong> — trade flows are generally stable")
    finding("Small commodity-dependent economies show the highest volatility — both export and import")
    finding("COVID created a volatility spike in 2020 but did not permanently elevate baseline volatility")