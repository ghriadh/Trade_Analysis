"""
Trade Analysis — T4 · Cross-Country Comparison
Covers: F.4 — Multi-country comparison
- F.4.1 Export/Import comparison (line + bar)
- F.4.2 Balance comparison
- F.4.3 Share comparison
- F.4.4 Growth comparison + CAGR heatmap
- F.4.6 Global quadrant views (all countries)
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
import numpy as np
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, format_value

inject_styles()

def _fmt(x):
    return format_value(x)


TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_trade      = pd.read_csv(TRADE_FILES / "df_trade.csv")
    df_structural = pd.read_csv(TRADE_FILES / "df_structural.csv")
    return df_trade, df_structural

df_trade, df_structural = load_data()
COUNTRIES = sorted(df_trade["country_name"].dropna().unique().tolist())
YEARS     = sorted(df_trade["year"].unique())
YEAR_MAX  = int(YEARS[-1])

# Colour palette — up to 10 countries
PALETTE = [
    "#F59E0B","#60A5FA","#10B981","#EF4444",
    "#A78BFA","#F472B6","#34D399","#FB923C",
    "#38BDF8","#FBBF24",
]

def country_color(countries):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(countries)}

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
PLOTLY_LAYOUT_COMPACT = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}

QUAD_COLORS = {
    "Dominant":       "#10B981",
    "Balanced":       "#60A5FA",
    "Vulnerable":     "#F59E0B",
    "Marginal":       "#6B7280",
    "Stars":          "#10B981",
    "Cash Cows":      "#60A5FA",
    "Question Marks": "#F59E0B",
    "Dogs":           "#EF4444",
    "Rising Star":    "#10B981",
    "Established":    "#60A5FA",
    "Declining":      "#F59E0B",
    "Struggling":     "#EF4444",
}

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Trade Analysis · Page 4 of 6</div>', unsafe_allow_html=True)
st.title("Cross-Country Comparison")
st.caption("F.4 — Multi-country analysis · volume · balance · share · growth · quadrants")
thin_divider()

# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR
# ════════════════════════════════════════════════════════════════

DEFAULT = ["USA","China","France","Canada","Brazil","Qatar","India","Nigeria","Germany","Japan"]
default_valid = [c for c in DEFAULT if c in COUNTRIES]

countries = st.multiselect(
    "Select countries to compare (2–10)",
    options=COUNTRIES,
    default=default_valid,
    max_selections=10,
    key="t4_countries",
)

if len(countries) < 2:
    st.warning("Please select at least 2 countries.")
    st.stop()

cmap = country_color(countries)
df_sel = df_trade[df_trade["country_name"].isin(countries)].copy()
df_str_sel = df_structural[df_structural["country_name"].isin(countries)].copy()

thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Volume",
    "⚖️ Balance",
    "📊 Share",
    "📈 Growth",
    "🎯 Quadrants",
])


# ── TAB 1 : VOLUME ───────────────────────────────────────────
with tab1:
    section_label("F.4.1 · Export & Import Comparison")

    metric = st.radio("Flow", ["Exports (X)", "Imports (M)", "Total Trade"],
                      horizontal=True, key="vol_metric")
    col_map = {"Exports (X)": "X", "Imports (M)": "M", "Total Trade": "total_trade"}
    col_use = col_map[metric]

    col1, col2 = st.columns(2)

    with col1:
        # Line chart — evolution
        fig_line = go.Figure()
        for c in countries:
            df_c = df_sel[df_sel["country_name"] == c].sort_values("year").copy()
            df_c["_fmt_v"] = df_c[col_use].apply(lambda v: f"${_fmt(v)}")
            fig_line.add_trace(go.Scatter(
                x=df_c["year"], y=df_c[col_use] / 1e9,
                mode="lines+markers", name=c,
                line=dict(color=cmap[c], width=2),
                marker=dict(size=6),
                customdata=df_c[["_fmt_v"]],
                hovertemplate="<b>" + c + "</b><br>%{x}<br>Value: %{customdata[0]}<extra></extra>",
            ))
        fig_line.update_layout(**PLOTLY_LAYOUT,
            title=f"{metric} Evolution (USD Billion)",
            yaxis_title="USD Billion",
        )
        st.plotly_chart(fig_line, width='stretch')

    with col2:
        # Bar chart — latest year snapshot
        df_snap = (df_sel[df_sel["year"] == YEAR_MAX]
                   .groupby("country_name")[col_use].sum()
                   .reindex(countries).reset_index()
                   .sort_values(col_use))

        fig_bar = go.Figure()
        df_snap["_fmt_val"] = df_snap[col_use].apply(lambda v: f"${_fmt(v)}")
        fig_bar.add_trace(go.Bar(
            x=df_snap[col_use] / 1e9,
            y=df_snap["country_name"],
            orientation="h",
            marker_color=[cmap[c] for c in df_snap["country_name"]],
            text=df_snap["_fmt_val"],
            textposition="auto",
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Value: %{text}<extra></extra>",
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT,
            title=f"{metric} Snapshot — {YEAR_MAX} (USD Billion)",
            xaxis_title="USD Billion",
        )
        st.plotly_chart(fig_bar, width='stretch')


# ── TAB 2 : BALANCE ──────────────────────────────────────────
with tab2:
    section_label("F.4.2 · Trade Balance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig_bline = go.Figure()
        for c in countries:
            df_c = df_sel[df_sel["country_name"] == c].sort_values("year")
            df_c = df_c.copy()
            df_c["_fmt_b"] = df_c["trade_balance_B"].apply(lambda v: f"${_fmt(abs(v) * 1e9)}")
            fig_bline.add_trace(go.Scatter(
                x=df_c["year"], y=df_c["trade_balance_B"],
                mode="lines+markers", name=c,
                line=dict(color=cmap[c], width=2),
                marker=dict(size=6),
                customdata=df_c[["_fmt_b"]],
                hovertemplate="<b>" + c + "</b><br>%{x}<br>Balance: %{customdata[0]}<extra></extra>",
            ))
        fig_bline.add_hline(y=0, line_color="#EEF4FB", line_width=1, line_dash="dash")
        fig_bline.update_layout(**PLOTLY_LAYOUT,
            title="Trade Balance Evolution (USD Billion)",
            yaxis_title="USD Billion",
        )
        st.plotly_chart(fig_bline, width='stretch')

    with col2:
        df_bsnap = (df_sel[df_sel["year"] == YEAR_MAX]
                    .groupby("country_name")["trade_balance_B"].sum()
                    .reindex(countries).reset_index()
                    .sort_values("trade_balance_B"))

        fig_bbar = go.Figure()
        df_bsnap["_fmt_b"] = df_bsnap["trade_balance_B"].apply(
            lambda v: f"${_fmt(abs(v) * 1e9)}")
        fig_bbar.add_trace(go.Bar(
            x=df_bsnap["trade_balance_B"],
            y=df_bsnap["country_name"],
            orientation="h",
            marker_color=[
                cmap[c] if df_bsnap.loc[
                    df_bsnap["country_name"] == c, "trade_balance_B"
                ].values[0] >= 0 else "#EF4444"
                for c in df_bsnap["country_name"]
            ],
            opacity=0.85,
            text=df_bsnap["_fmt_b"],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
        ))
        fig_bbar.add_vline(x=0, line_color="#EEF4FB", line_width=1)
        fig_bbar.update_layout(**PLOTLY_LAYOUT,
            title=f"Trade Balance Snapshot — {YEAR_MAX}",
            xaxis_title="USD Billion",
        )
        st.plotly_chart(fig_bbar, width='stretch')

    # Coverage ratio comparison
    section_label("Coverage Ratio")
    fig_cov = go.Figure()
    for c in countries:
        df_c = df_sel[df_sel["country_name"] == c].sort_values("year")
        fig_cov.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["coverage_ratio"] * 100,
            mode="lines+markers", name=c,
            line=dict(color=cmap[c], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{c}</b><br>%{{x}}<br>Coverage: %{{y:.1f}}%<extra></extra>",
        ))
    fig_cov.add_hline(y=100, line_dash="dash", line_color="#EEF4FB",
                      annotation_text="Balanced (100%)")
    fig_cov.update_layout(**PLOTLY_LAYOUT,
        title="Coverage Ratio Evolution (Exports / Imports × 100%)",
        yaxis_title="Coverage (%)",
    )
    st.plotly_chart(fig_cov, width='stretch')


# ── TAB 3 : SHARE ────────────────────────────────────────────
with tab3:
    section_label("F.4.3 · World Share Comparison")

    share_metric = st.radio(
        "Share metric",
        ["Export Share", "Import Share"],
        horizontal=True, key="share_metric"
    )
    scol = "export_share" if share_metric == "Export Share" else "import_share"

    col1, col2 = st.columns(2)

    with col1:
        fig_sline = go.Figure()
        for c in countries:
            df_c = df_sel[df_sel["country_name"] == c].sort_values("year")
            fig_sline.add_trace(go.Scatter(
                x=df_c["year"], y=df_c[scol],
                mode="lines+markers", name=c,
                line=dict(color=cmap[c], width=2),
                marker=dict(size=6),
                hovertemplate="<b>" + c + "</b><br>%{x}<br>" + share_metric + ": %{y:.3f}%<extra></extra>",
            ))
        fig_sline.update_layout(**PLOTLY_LAYOUT,
            title=f"{share_metric} Evolution",
            yaxis_title="Share (%)",
        )
        st.plotly_chart(fig_sline, width='stretch')

    with col2:
        df_ssnap = (df_sel[df_sel["year"] == YEAR_MAX]
                    .groupby("country_name")[scol].mean()
                    .reindex(countries).reset_index()
                    .sort_values(scol))

        fig_sbar = go.Figure()
        fig_sbar.add_trace(go.Bar(
            x=df_ssnap[scol],
            y=df_ssnap["country_name"],
            orientation="h",
            marker_color=[cmap[c] for c in df_ssnap["country_name"]],
            text=df_ssnap[scol].apply(lambda v: f"{v:.3f}%"),
            textposition="auto",
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Share: %{text}<extra></extra>",
        ))
        fig_sbar.update_layout(**PLOTLY_LAYOUT,
            title=f"{share_metric} Snapshot — {YEAR_MAX}",
            xaxis_title="Share (%)",
        )
        st.plotly_chart(fig_sbar, width='stretch')


# ── TAB 4 : GROWTH ───────────────────────────────────────────
with tab4:
    section_label("F.4.4 · Growth Comparison")

    g_metric = st.radio(
        "Growth metric",
        ["Export Growth", "Import Growth"],
        horizontal=True, key="growth_metric"
    )
    gcol = "export_growth" if g_metric == "Export Growth" else "import_growth"
    cagr_col = "cagr_export" if g_metric == "Export Growth" else "cagr_import"

    col1, col2 = st.columns(2)

    with col1:
        # YoY line chart
        fig_gline = go.Figure()
        fig_gline.add_hline(y=0, line_color="#EEF4FB", line_width=1, line_dash="dash")
        for c in countries:
            df_c = df_sel[df_sel["country_name"] == c].dropna(subset=[gcol]).sort_values("year")
            fig_gline.add_trace(go.Scatter(
                x=df_c["year"], y=df_c[gcol],
                mode="lines+markers", name=c,
                line=dict(color=cmap[c], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{c}</b><br>%{{x}}<br>Growth: %{{y:.1f}}%<extra></extra>",
            ))
        fig_gline.update_layout(**PLOTLY_LAYOUT,
            title=f"{g_metric} YoY (%)",
            yaxis_title="%",
        )
        st.plotly_chart(fig_gline, width='stretch')

    with col2:
        # CAGR bar
        if cagr_col in df_structural.columns:
            df_cagr = (df_structural[df_structural["country_name"].isin(countries)]
                       [["country_name", cagr_col]]
                       .sort_values(cagr_col))

            fig_cagr = go.Figure()
            fig_cagr.add_trace(go.Bar(
                x=df_cagr[cagr_col] * 100,
                y=df_cagr["country_name"],
                orientation="h",
                marker_color=[
                    cmap[c] if df_cagr.loc[df_cagr["country_name"]==c, cagr_col].values[0] >= 0
                    else "#EF4444"
                    for c in df_cagr["country_name"]
                ],
                text=(df_cagr[cagr_col] * 100).apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>CAGR: %{text}<extra></extra>",
                opacity=0.85,
            ))
            fig_cagr.add_vline(x=0, line_color="#EEF4FB", line_width=1)
            fig_cagr.update_layout(**PLOTLY_LAYOUT,
                title=f"CAGR {g_metric} — Full Period (%)",
                xaxis_title="%",
            )
            st.plotly_chart(fig_cagr, width='stretch')

    # Growth heatmap
    section_label("Growth Heatmap — YoY by Country × Year")
    df_heat = df_sel.pivot_table(
        index="country_name", columns="year", values=gcol
    ).reindex(countries)

    fig_heat = go.Figure(go.Heatmap(
        z=df_heat.values,
        x=[str(y) for y in df_heat.columns],
        y=df_heat.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(df_heat.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=11),
        hoverongaps=False,
        colorbar=dict(title="%"),
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT,
        title=f"{g_metric} (%) — Year × Country",
        xaxis_title="Year",
        height=max(300, len(countries) * 45 + 100),
    )
    st.plotly_chart(fig_heat, width='stretch')


# ── TAB 5 : QUADRANTS ────────────────────────────────────────
with tab5:
    section_label("F.4.6 · Global Quadrant Views — All Countries")

    quad_tabs = st.tabs(["📐 Static", "⚡ Dynamic", "⭐ Strategic"])

    def _global_scatter(x_col, y_col, x_thr, y_thr,
                        x_label, y_label, title, q_col,
                        highlights, quad_labels=None,
                        x_range=None, y_range=None):
        df_all = df_structural.copy().dropna(subset=[x_col, y_col])
        df_all["_size"] = (
            df_all["structural_score"].rank() / len(df_all) * 25 + 6
        ).clip(6, 31)

        fig = go.Figure()

        df_bg = df_all[~df_all["country_name"].isin(highlights)]
        fig.add_trace(go.Scatter(
            x=df_bg[x_col], y=df_bg[y_col],
            mode="markers", name="Other countries",
            marker=dict(
                color=[QUAD_COLORS.get(q, "#6B7280") for q in df_bg[q_col]],
                size=8, opacity=0.35, line=dict(width=0),
            ),
            customdata=df_bg[["country_name", q_col]],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          f"{x_label}: %{{x:.3f}}<br>"
                          f"{y_label}: %{{y:.3f}}<br>"
                          "Quadrant: %{customdata[1]}<extra></extra>",
            showlegend=False,
        ))

        for c in highlights:
            row_h = df_all[df_all["country_name"] == c]
            if row_h.empty:
                continue
            r = row_h.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r[x_col]], y=[r[y_col]],
                mode="markers+text", name=c,
                marker=dict(color=cmap.get(c, "#EEF4FB"), size=16, symbol="star",
                            line=dict(width=1.5, color="#EEF4FB")),
                text=[c], textposition="top center",
                textfont=dict(size=11, color="#EEF4FB"),
                customdata=[[r.get(q_col, "—"), f"{r[x_col]:.3f}", f"{r[y_col]:.3f}"]],
                hovertemplate=f"<b>{c}</b><br>"
                              f"{x_label}: %{{customdata[1]}}<br>"
                              f"{y_label}: %{{customdata[2]}}<br>"
                              "Quadrant: %{customdata[0]}<extra></extra>",
            ))

        fig.add_vline(x=x_thr, line_dash="dash", line_color="#6890B0", line_width=2)
        fig.add_hline(y=y_thr, line_dash="dash", line_color="#6890B0", line_width=2)

        # Quadrant labels in data coordinates
        if quad_labels:
            _xlo = x_range[0] if x_range else df_all[x_col].min()
            _xhi = x_range[1] if x_range else df_all[x_col].max()
            _ylo = y_range[0] if y_range else df_all[y_col].min()
            _yhi = y_range[1] if y_range else df_all[y_col].max()
            _xr = (x_thr + _xhi) / 2
            _xl = (_xlo + x_thr) / 2
            _yt = (y_thr + _yhi) / 2
            _yb = (_ylo + y_thr) / 2
            positions = {"TR": (_xr, _yt), "TL": (_xl, _yt),
                         "BR": (_xr, _yb), "BL": (_xl, _yb)}
            for quad, txt, clr in quad_labels:
                xp, yp = positions.get(quad, (x_thr, y_thr))
                fig.add_annotation(x=xp, y=yp, text=txt, showarrow=False,
                    font=dict(size=10, color=clr),
                    bgcolor="rgba(22,32,53,0.75)", borderpad=4, align="center")

        _layout = dict(**PLOTLY_LAYOUT)
        if x_range:
            _layout["xaxis"] = dict(title=x_label, range=x_range,
                                    gridcolor="#304870", linecolor="#304870")
        else:
            _layout["xaxis"] = dict(title=x_label, autorange=True,
                                    gridcolor="#304870", linecolor="#304870")
        if y_range:
            _layout["yaxis"] = dict(title=y_label, range=y_range,
                                    gridcolor="#304870", linecolor="#304870")
        else:
            _layout["yaxis"] = dict(title=y_label, autorange=True,
                                    gridcolor="#304870", linecolor="#304870")
        _layout.pop("xaxis", None) if x_range is None else None
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis","yaxis")},
            title=title, height=560,
            xaxis=_layout.get("xaxis", dict(title=x_label, gridcolor="#304870")),
            yaxis=_layout.get("yaxis", dict(title=y_label, gridcolor="#304870")),
        )
        return fig

    with quad_tabs[0]:
        _sq_xhi = df_structural["export_share_structural"].max() * 1.05
        _sq_yhi = df_structural["coverage_ratio_structural"].max() * 1.05
        _sq_ylo = df_structural["coverage_ratio_structural"].min() * 0.95
        fig_sq = _global_scatter(
            "export_share_structural", "coverage_ratio_structural",
            1.0, 1.0,
            "Export Share (%)", "Coverage Ratio",
            "Static Quadrant — All Countries",
            "static_quadrant", countries,
            quad_labels=[
                ("TR", "🟢 Dominant<br><sub>Large & Surplus</sub>",   "#10B981"),
                ("TL", "🔵 Balanced<br><sub>Small & Surplus</sub>",   "#60A5FA"),
                ("BR", "🟡 Vulnerable<br><sub>Large & Deficit</sub>", "#F59E0B"),
                ("BL", "🔴 Marginal<br><sub>Small & Deficit</sub>",   "#F472B6"),
            ],
            x_range=[0, _sq_xhi],
            y_range=[_sq_ylo, _sq_yhi],
        )
        st.plotly_chart(fig_sq, width='stretch')

    with quad_tabs[1]:
        _vol_median = df_structural["volatility_export"].median()
        _vol_cap    = df_structural["volatility_export"].quantile(0.95)
        _cagr_xlo   = df_structural["cagr_export"].quantile(0.03) * 100
        _cagr_xhi   = df_structural["cagr_export"].quantile(0.97) * 100
        _cagr_ylo   = df_structural["volatility_export"].min()
        # Compute label midpoints
        _dq_xr = (0 + _cagr_xhi) / 2
        _dq_xl = (_cagr_xlo + 0) / 2
        _dq_ylo_v = df_structural["volatility_export"].min()
        _dq_yt = (_vol_median + _vol_cap) / 2
        _dq_yb = (_dq_ylo_v + _vol_median) / 2

        df_dq_plot = df_structural.copy().dropna(
            subset=["cagr_export","volatility_export","dynamic_quadrant"])
        DYN_COLORS = {"Star":"#10B981","Risky Grower":"#F59E0B",
                      "Declining":"#EF4444","Stable Stagnant":"#60A5FA"}
        fig_dq = go.Figure()
        df_dq_bg = df_dq_plot[~df_dq_plot["country_name"].isin(countries)]
        fig_dq.add_trace(go.Scatter(
            x=df_dq_bg["cagr_export"]*100, y=df_dq_bg["volatility_export"],
            mode="markers", name="Others",
            marker=dict(color=[DYN_COLORS.get(q,"#6B7280")
                               for q in df_dq_bg["dynamic_quadrant"]],
                        size=7, opacity=0.35, line=dict(width=0)),
            customdata=df_dq_bg[["country_name","dynamic_quadrant"]],
            hovertemplate="<b>%{customdata[0]}</b><br>CAGR: %{x:.1f}%<br>"
                          "Volatility: %{y:.2f}<br>Quadrant: %{customdata[1]}<extra></extra>",
            showlegend=False,
        ))
        for cc in countries:
            rr = df_dq_plot[df_dq_plot["country_name"] == cc]
            if rr.empty: continue
            rr = rr.iloc[0]
            fig_dq.add_trace(go.Scatter(
                x=[rr["cagr_export"]*100], y=[rr["volatility_export"]],
                mode="markers+text", name=cc,
                marker=dict(color=cmap.get(cc,"#EEF4FB"), size=16, symbol="star",
                            line=dict(width=1.5, color="#EEF4FB")),
                text=[cc], textposition="top center",
                textfont=dict(size=11, color="#EEF4FB"),
                customdata=[[rr.get("dynamic_quadrant","—"),
                             f"{rr['cagr_export']:.2%}", f"{rr['volatility_export']:.2f}"]],
                hovertemplate=f"<b>{cc}</b><br>CAGR: %{{customdata[1]}}<br>"
                              "Volatility: %{customdata[2]}<br>"
                              "Quadrant: %{customdata[0]}<extra></extra>",
            ))
        fig_dq.add_vline(x=0, line_dash="dash", line_color="#6890B0", line_width=2)
        fig_dq.add_hline(y=_vol_median, line_dash="dash", line_color="#6890B0", line_width=2)
        for txt, xp, yp, clr in [
            ("⭐ Star<br>(High CAGR · Low Vol.)",           _dq_xr, _dq_yb, "#10B981"),
            ("😴 Stable Stagnant<br>(Low CAGR · Low Vol.)", _dq_xl, _dq_yb, "#60A5FA"),
            ("📈 Risky Grower<br>(High CAGR · High Vol.)",  _dq_xr, _dq_yt, "#F59E0B"),
            ("📉 Declining<br>(Low CAGR · High Vol.)",      _dq_xl, _dq_yt, "#EF4444"),
        ]:
            fig_dq.add_annotation(x=xp, y=yp, text=txt, showarrow=False,
                font=dict(size=10, color=clr),
                bgcolor="rgba(22,32,53,0.75)", borderpad=4, align="center")
        fig_dq.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                                if k not in ("xaxis","yaxis")},
            title="Dynamic Quadrant — All Countries",
            xaxis=dict(title="CAGR Export (%)",
                       range=[_cagr_xlo, _cagr_xhi], gridcolor="#304870"),
            yaxis=dict(title="Export Volatility",
                       range=[0, _vol_cap], gridcolor="#304870"),
            height=560,
        )
        st.plotly_chart(fig_dq, width='stretch')

    with quad_tabs[2]:
        _strat_xhi = df_structural["export_share_structural"].max() * 1.05
        _strat_yhi = df_structural["cagr_export"].max()
        _strat_ylo = df_structural["cagr_export"].min()
        fig_strat = _global_scatter(
            "export_share_structural", "cagr_export",
            1.0, 0,
            "Export Share (%)", "CAGR Export (%)",
            "Strategic Quadrant (BCG-style) — All Countries",
            "strategic_quadrant", countries,
            quad_labels=[
                ("TR", "⭐ Stars<br><sub>Large & Growing</sub>",      "#10B981"),
                ("TL", "🌱 Emerging<br><sub>Small & Growing</sub>",   "#60A5FA"),
                ("BR", "🐄 Cash Cows<br><sub>Large & Stagnant</sub>", "#A78BFA"),
                ("BL", "📉 Marginal<br><sub>Small & Shrinking</sub>", "#6B7280"),
            ],
            x_range=[0, _strat_xhi],
            y_range=[_strat_ylo, _strat_yhi],
        )
        st.plotly_chart(fig_strat, width='stretch')

    with st.expander("Key findings — Quadrant distribution"):
        finding(
            "<strong>Static quadrant</strong>: Most countries are in Marginal or Balanced — "
            "only ~15% qualify as Dominant (large share + surplus). Threshold: 1% export share."
        )
        finding(
            "<strong>Dynamic quadrant</strong>: Stars (high CAGR + low volatility) dominate "
            "large economies — Declining concentrated in commodity-dependent small traders"
        )
        finding(
            "<strong>Strategic quadrant</strong>: Emerging = small share but positive CAGR — "
            "the most interesting group for future growth potential"
        )