"""
Trade × Indicators — Link · Country Profile
Trade positioning × macro context per country
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
import numpy as np
from scipy.stats import spearmanr
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding

inject_styles()

LINK_FILES  = ROOT / "Analysis" / "Analysis_files" / "Link"
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
IND_FILES   = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_comb = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_ind  = pd.read_csv(IND_FILES / "df_indicators.csv",
                          usecols=["country_name", "entity_type",
                                   "indicator_name", "year", "value"])
    df_sp   = pd.read_csv(LINK_FILES / "df_spearman.csv")

    # Latest macro value per country per indicator
    df_latest = (df_ind[df_ind["entity_type"] == "COUNTRY"]
                 .sort_values("year")
                 .groupby(["country_name", "indicator_name"])["value"]
                 .last().reset_index()
                 .pivot(index="country_name", columns="indicator_name", values="value")
                 .reset_index())
    df_latest.columns.name = None

    df_link = df_comb.merge(df_latest, on="country_name", how="inner")
    return df_comb, df_link, df_sp

df_comb, df_link, df_sp = load_data()

# ── Macro column definitions ──
IND_LABELS = {
    "GDP per capita (current US$)":                                          "GDP/capita (US$)",
    "GDP (current US$)":                                                     "GDP (US$)",
    "Current account balance (% of GDP)":                                   "Current Acct % GDP",
    "Inflation, consumer prices (annual %)":                                 "Inflation %",
    "Real interest rate (%)":                                                "Real Interest Rate %",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)":  "Unemployment %",
    "Trade (% of GDP)":                                                      "Trade % GDP",
    "Expense (% of GDP)":                                                    "Expense % GDP",
    "Exports of goods and services (% of GDP)":                             "Exports % GDP",
    "Imports of goods and services (% of GDP)":                             "Imports % GDP",
    "Foreign direct investment, net inflows (BoP, current US$)":           "FDI Inflows (US$)",
    "Population, total":                                                     "Population",
}

TRADE_LABELS = {
    "combined_score":              "Combined Score",
    "structural_score":            "Structural Score",
    "f_norm_rank":                 "F Norm Rank",
    "g_norm_rank":                 "G Norm Rank",
    "export_share_structural":     "Export Share",
    "import_share_structural":     "Import Share",
    "cagr_export":                 "CAGR Export",
    "cagr_import":                 "CAGR Import",
    "volatility_export":           "Volatility Export",
    "resilience_export":           "Resilience Export",
    "coverage_ratio_structural":   "Coverage Ratio",
}

MACRO_COLS = [c for c in IND_LABELS if c in df_link.columns]
TRADE_COLS = [c for c in TRADE_LABELS if c in df_link.columns]

# ════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1F3050",
    plot_bgcolor="#162035",
    font=dict(family="IBM Plex Sans", color="#C4D5E8", size=13),
    title_font=dict(color="#EEF4FB", size=15),
    legend=dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
    margin=dict(l=60, r=40, t=50, b=60),
    xaxis=dict(gridcolor="#304870", linecolor="#304870"),
    yaxis=dict(gridcolor="#304870", linecolor="#304870"),
)

def _pl(fig, **kw):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kw})
    return fig

# ════════════════════════════════════════════════════════════════
# HEADER + COUNTRY SELECTOR
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Link · Trade × Indicators · Page 2 of 2</div>',
            unsafe_allow_html=True)
st.title("Country Profile")
st.caption(
    "Trade positioning in macroeconomic context — "
    "how each country's trade scores relate to its macro indicators"
)
thin_divider()

countries = sorted(df_link["country_name"].tolist())
_default  = "France" if "France" in countries else countries[0]
country   = st.selectbox("Country", countries, index=countries.index(_default))

thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_a, tab_b, tab_c = st.tabs([
    "A · Trade & Macro Profile",
    "B · Scatter Explorer",
    "C · Score Positioning",
])

# ────────────────────────────────────────────────────────────────
# TAB A — TRADE & MACRO PROFILE SUMMARY
# ────────────────────────────────────────────────────────────────
with tab_a:
    section_label("A · Trade & Macro Profile — " + country)

    row = df_link[df_link["country_name"] == country].iloc[0]

    # ── Trade KPIs ──
    st.markdown("**Trade Position**")
    kpi_cols_t = st.columns(5)
    _trade_kpis = [
        ("Combined Score",   f"{row['combined_score']:.2f}",                    "#10B981"),
        ("Structural Score", f"{row['structural_score']:.2f}",                  "#60A5FA"),
        ("Combined Profile", str(row.get("combined_profile", "—")),             "#A78BFA"),
        ("Meta Profile",     str(row.get("meta_profile", "—")),                 "#F59E0B"),
        ("Strat. Quadrant",  str(row.get("strategic_quadrant_combined", "—")), "#34D399"),
    ]
    for col, (label, val, color) in zip(kpi_cols_t, _trade_kpis):
        col.markdown(
            f'<div class="kpi-card" style="border-top-color:{color};">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color};font-size:16px;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:.6rem;'></div>", unsafe_allow_html=True)

    # ── Trade metrics bar vs global ──
    col_bar_a, col_macro_a = st.columns([3, 2])

    with col_bar_a:
        _metrics = ["export_share_structural", "import_share_structural",
                    "coverage_ratio_structural", "cagr_export", "cagr_import",
                    "volatility_export"]
        _m_labels = [TRADE_LABELS.get(m, m) for m in _metrics if m in df_link.columns]
        _m_cols   = [m for m in _metrics if m in df_link.columns]
        _country_vals = [row[m] for m in _m_cols]
        _global_means = [df_link[m].median() for m in _m_cols]

        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            name="Global Median",
            x=_m_labels,
            y=_global_means,
            marker_color="#304870",
            opacity=0.7,
        ))
        fig_bars.add_trace(go.Bar(
            name=country,
            x=_m_labels,
            y=_country_vals,
            marker_color="#10B981",
        ))
        _pl(fig_bars,
            title="Trade Metrics vs Global Median",
            barmode="group",
            height=340,
            margin=dict(l=50, r=20, t=50, b=80),
            xaxis=dict(tickangle=-35, gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    with col_macro_a:
        st.markdown("**Macro Indicators (latest available)**")
        _KEY_MACRO = [
            "GDP per capita (current US$)",
            "Current account balance (% of GDP)",
            "Inflation, consumer prices (annual %)",
            "Unemployment, total (% of total labor force) (modeled ILO estimate)",
            "Real interest rate (%)",
            "Trade (% of GDP)",
        ]
        for ind in _KEY_MACRO:
            if ind not in df_link.columns:
                continue
            val = row.get(ind, None)
            lbl = IND_LABELS.get(ind, ind)
            global_med = df_link[ind].median()
            if pd.isna(val):
                disp = "—"
                delta_html = ""
            else:
                if ind == "GDP per capita (current US$)":
                    disp = f"${val:,.0f}"
                    med_disp = f"${global_med:,.0f}"
                elif ind in ["GDP (current US$)", "Foreign direct investment, net inflows (BoP, current US$)",
                             "Population, total"]:
                    disp = f"{val:.2e}"
                    med_disp = f"{global_med:.2e}"
                else:
                    disp = f"{val:+.1f}%" if val < 0 else f"{val:.1f}%"
                    med_disp = f"{global_med:.1f}%"
                above = val > global_med
                arrow = "▲" if above else "▼"
                arrow_color = "#10B981" if above else "#EF4444"
                delta_html = (
                    f'<span style="color:{arrow_color};font-size:11px;">'
                    f'{arrow} vs median {med_disp}</span>'
                )

            st.markdown(
                f'<div style="background:#162035;border:1px solid #304870;border-radius:8px;'
                f'padding:.45rem .75rem;margin-bottom:4px;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span style="font-size:12px;color:#8AABB0;">{lbl}</span>'
                f'<span style="font-size:14px;font-weight:600;color:#EEF4FB;">'
                f'{disp}&nbsp;{delta_html}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ────────────────────────────────────────────────────────────────
# TAB B — SCATTER EXPLORER
# ────────────────────────────────────────────────────────────────
with tab_b:
    section_label("B · Scatter Explorer — All Countries")

    col_b1, col_b2 = st.columns(2)
    sel_trade_b = col_b1.selectbox(
        "Trade variable (X axis)",
        TRADE_COLS,
        format_func=lambda x: TRADE_LABELS.get(x, x),
        index=TRADE_COLS.index("combined_score") if "combined_score" in TRADE_COLS else 0,
        key="scatter_trade",
    )
    sel_macro_b = col_b2.selectbox(
        "Macro indicator (Y axis)",
        MACRO_COLS,
        format_func=lambda x: IND_LABELS.get(x, x),
        index=MACRO_COLS.index("GDP per capita (current US$)")
              if "GDP per capita (current US$)" in MACRO_COLS else 0,
        key="scatter_macro",
    )

    df_b = df_link[[
        "country_name", sel_trade_b, sel_macro_b
    ]].dropna()

    if len(df_b) < 10:
        st.info("Not enough data for this combination.")
    else:
        # Spearman correlation
        rho_b, pval_b = spearmanr(df_b[sel_trade_b], df_b[sel_macro_b])
        sig_b = pval_b < 0.05

        # Classify as highlighted or not
        is_country = df_b["country_name"] == country
        df_others  = df_b[~is_country]
        df_sel     = df_b[is_country]

        # ── [T]/[S]/[E] flag for macro indicator ──
        _ind_flag_b = df_sp[df_sp["indicator"] == sel_macro_b]["ind_flag"].iloc[0] \
            if sel_macro_b in df_sp["indicator"].values else "—"
        _flag_color_b = {"[E]": "#10B981", "[T]": "#EF4444", "[S]": "#F59E0B"}.get(_ind_flag_b, "#60A5FA")

        fig_b = go.Figure()

        # All countries
        fig_b.add_trace(go.Scatter(
            x=df_others[sel_trade_b],
            y=df_others[sel_macro_b],
            mode="markers",
            marker=dict(color="#304870", size=6, opacity=0.7),
            text=df_others["country_name"],
            hovertemplate="<b>%{text}</b><br>X = %{x:.3f}<br>Y = %{y:.3f}<extra></extra>",
            name="Other countries",
        ))

        # Selected country
        if not df_sel.empty:
            fig_b.add_trace(go.Scatter(
                x=df_sel[sel_trade_b],
                y=df_sel[sel_macro_b],
                mode="markers+text",
                marker=dict(color="#10B981", size=14, symbol="diamond",
                            line=dict(color="#FFFFFF", width=2)),
                text=[country],
                textposition="top center",
                textfont=dict(size=13, color="#FFFFFF"),
                hovertemplate=f"<b>{country}</b><br>X = %{{x:.3f}}<br>Y = %{{y:.3f}}<extra></extra>",
                name=country,
            ))

        # Trend line (OLS)
        _x_arr = df_b[sel_trade_b].values
        _y_arr = df_b[sel_macro_b].values
        _poly  = np.polyfit(_x_arr, _y_arr, 1)
        _x_line = np.linspace(_x_arr.min(), _x_arr.max(), 100)
        _y_line = np.polyval(_poly, _x_line)
        fig_b.add_trace(go.Scatter(
            x=_x_line, y=_y_line,
            mode="lines",
            line=dict(color="#60A5FA", dash="dash", width=1.5),
            name="OLS trend",
            hoverinfo="skip",
        ))

        _sig_txt = "✓ significant" if sig_b else "✗ not significant"
        _sig_color = "#10B981" if sig_b else "#6890B0"
        fig_b.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=(f"ρ = {rho_b:+.3f}  <span style='color:{_sig_color}'>{_sig_txt}</span><br>"
                  f"<span style='color:{_flag_color_b}'>{_ind_flag_b}</span> indicator  "
                  f"n = {len(df_b)}"),
            showarrow=False,
            align="right",
            xanchor="right", yanchor="top",
            bgcolor="#162035",
            bordercolor="#304870",
            font=dict(size=12, color="#C4D5E8"),
        )

        _pl(fig_b,
            title=(f"{TRADE_LABELS.get(sel_trade_b, sel_trade_b)} × "
                   f"{IND_LABELS.get(sel_macro_b, sel_macro_b)}"),
            xaxis_title=TRADE_LABELS.get(sel_trade_b, sel_trade_b),
            yaxis_title=IND_LABELS.get(sel_macro_b, sel_macro_b),
            height=500,
        )
        st.plotly_chart(fig_b, use_container_width=True)

        if not sig_b:
            st.info(
                f"No significant correlation (ρ = {rho_b:+.3f}, p = {pval_b:.3f}). "
                f"This indicator-trade pair does not show a systematic relationship "
                f"across countries."
            )

# ────────────────────────────────────────────────────────────────
# TAB C — SCORE POSITIONING
# ────────────────────────────────────────────────────────────────
with tab_c:
    section_label("C · Score Positioning — " + country)

    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>"
        "Where does this country sit in the trade score vs macro space? "
        "Compare the selected country against all countries on four key "
        "trade × macro dimensions.</p>",
        unsafe_allow_html=True,
    )

    _ROW_COUNTRY = df_link[df_link["country_name"] == country]
    if _ROW_COUNTRY.empty:
        st.warning(f"No data for {country}.")
    else:
        _r = _ROW_COUNTRY.iloc[0]

        # 4 scatter panels: combined_score vs 4 macro indicators
        _PANELS = [
            ("GDP per capita (current US$)",              "combined_score"),
            ("Current account balance (% of GDP)",        "structural_score"),
            ("Inflation, consumer prices (annual %)",      "combined_score"),
            ("Unemployment, total (% of total labor force) (modeled ILO estimate)", "structural_score"),
        ]

        cols_c = st.columns(2)

        for idx, (macro_col, trade_col) in enumerate(_PANELS):
            if macro_col not in df_link.columns or trade_col not in df_link.columns:
                continue
            df_p = df_link[["country_name", macro_col, trade_col]].dropna()
            if len(df_p) < 10:
                continue

            _rho_p, _pval_p = spearmanr(df_p[macro_col], df_p[trade_col])
            _sig_p = _pval_p < 0.05

            _is_sel = df_p["country_name"] == country
            _others_p = df_p[~_is_sel]
            _sel_p    = df_p[_is_sel]

            # Color by combined_profile if available
            _color_map = {
                "Global Champion": "#10B981",
                "Strong":          "#60A5FA",
                "Moderate":        "#F59E0B",
                "Weak":            "#EF4444",
                "Marginal":        "#6890B0",
            }
            _comb_profiles = df_link.set_index("country_name").get("combined_profile", pd.Series())
            _pt_colors = [
                _color_map.get(str(_comb_profiles.get(n, "")).split()[0], "#304870")
                for n in _others_p["country_name"]
            ]

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=_others_p[macro_col],
                y=_others_p[trade_col],
                mode="markers",
                marker=dict(color=_pt_colors, size=5, opacity=0.6),
                text=_others_p["country_name"],
                hovertemplate="<b>%{text}</b><br>%{x:.2f} | %{y:.2f}<extra></extra>",
                showlegend=False,
            ))
            if not _sel_p.empty:
                fig_p.add_trace(go.Scatter(
                    x=_sel_p[macro_col],
                    y=_sel_p[trade_col],
                    mode="markers+text",
                    marker=dict(color="#FFD700", size=14, symbol="star",
                                line=dict(color="#FFFFFF", width=2)),
                    text=[country],
                    textposition="top center",
                    textfont=dict(size=12, color="#FFFFFF"),
                    hovertemplate=f"<b>{country}</b><br>%{{x:.2f}} | %{{y:.2f}}<extra></extra>",
                    showlegend=False,
                ))

            _sig_ann = f"ρ={_rho_p:+.3f}" + (" ✓" if _sig_p else " ✗")
            _pl(fig_p,
                title=f"{IND_LABELS.get(macro_col, macro_col)} × {TRADE_LABELS.get(trade_col, trade_col)}",
                xaxis_title=IND_LABELS.get(macro_col, macro_col),
                yaxis_title=TRADE_LABELS.get(trade_col, trade_col),
                height=320,
                margin=dict(l=50, r=20, t=50, b=50),
                annotations=[dict(
                    text=_sig_ann, x=0.98, y=0.98,
                    xref="paper", yref="paper",
                    showarrow=False, xanchor="right", yanchor="top",
                    bgcolor="#162035", bordercolor="#304870",
                    font=dict(size=11, color="#10B981" if _sig_p else "#6890B0"),
                )],
            )
            with cols_c[idx % 2]:
                st.plotly_chart(fig_p, use_container_width=True)

        # Percentile summary for selected country
        thin_divider()
        st.markdown(f"**{country} — Global Percentile Ranking**")
        _pct_cols = st.columns(4)
        _PCT_METRICS = [
            ("combined_score",          "Combined Score",     "#10B981"),
            ("structural_score",        "Structural Score",   "#60A5FA"),
            ("f_norm_rank",             "F Norm Rank",        "#A78BFA"),
            ("g_norm_rank",             "G Norm Rank",        "#F59E0B"),
        ]
        for col, (mc, lbl, color) in zip(_pct_cols, _PCT_METRICS):
            if mc not in df_link.columns:
                continue
            val = _r.get(mc)
            if pd.isna(val):
                pct_str = "—"
            else:
                pct = (df_link[mc] <= val).mean() * 100
                pct_str = f"Top {100 - pct:.0f}%"
            col.markdown(
                f'<div class="kpi-card" style="border-top-color:{color};">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value" style="color:{color};font-size:20px;">{pct_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

thin_divider()
