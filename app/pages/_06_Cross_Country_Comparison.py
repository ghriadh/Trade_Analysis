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
from utils.i18n import t

inject_styles()

# ════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ════════════════════════════════════════════════════════════════
# NOTE: quadrant/profile category values coming from df_structural
# columns (e.g. "Dominant", "Stars", "Cash Cows", "Declining", ...)
# are DATA and are never translated, even when shown as plain text.

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 4 of 6",
        "title": "Cross-Country Comparison",
        "caption": "F.4 — Multi-country analysis · volume · balance · share · growth · quadrants",
        "lbl_select_countries": "Select countries to compare (2–10)",
        "warn_min_countries": "Please select at least 2 countries.",

        "tab_volume": "📦 Volume",
        "tab_balance": "⚖️ Balance",
        "tab_share": "📊 Share",
        "tab_growth": "📈 Growth",
        "tab_quadrants": "🎯 Quadrants",

        "sec_volume": "F.4.1 · Export & Import Comparison",
        "lbl_flow": "Flow",
        "opt_exports_x": "Exports (X)",
        "opt_imports_m": "Imports (M)",
        "opt_total_trade": "Total Trade",
        "title_evolution": "{metric} Evolution (USD Billion)",
        "unit_usd_billion": "USD Billion",
        "title_snapshot": "{metric} Snapshot — {year} (USD Billion)",

        "sec_balance": "F.4.2 · Trade Balance Comparison",
        "title_balance_evolution": "Trade Balance Evolution (USD Billion)",
        "title_balance_snapshot": "Trade Balance Snapshot — {year}",
        "sec_coverage_ratio": "Coverage Ratio",
        "balanced_100": "Balanced (100%)",
        "title_coverage_evolution": "Coverage Ratio Evolution (Exports / Imports × 100%)",
        "unit_coverage_pct": "Coverage (%)",

        "sec_share": "F.4.3 · World Share Comparison",
        "lbl_share_metric": "Share metric",
        "opt_export_share": "Export Share",
        "opt_import_share": "Import Share",
        "title_share_evolution": "{metric} Evolution",
        "unit_share_pct": "Share (%)",
        "title_share_snapshot": "{metric} Snapshot — {year}",

        "sec_growth": "F.4.4 · Growth Comparison",
        "lbl_growth_metric": "Growth metric",
        "opt_export_growth": "Export Growth",
        "opt_import_growth": "Import Growth",
        "title_growth_yoy": "{metric} YoY (%)",
        "unit_pct": "%",
        "title_cagr": "CAGR {metric} — Full Period (%)",
        "sec_growth_heatmap": "Growth Heatmap — YoY by Country × Year",
        "title_heatmap": "{metric} (%) — Year × Country",
        "axis_year": "Year",

        "sec_quadrants": "F.4.6 · Global Quadrant Views — All Countries",
        "quad_tab_static": "📐 Static",
        "quad_tab_dynamic": "⚡ Dynamic",
        "quad_tab_strategic": "⭐ Strategic",
        "other_countries": "Other countries",
        "others": "Others",
        "quad_dominant_desc": "🟢 Dominant<br><sub>Large & Surplus</sub>",
        "quad_balanced_desc": "🔵 Balanced<br><sub>Small & Surplus</sub>",
        "quad_vulnerable_desc": "🟡 Vulnerable<br><sub>Large & Deficit</sub>",
        "quad_marginal_desc": "🔴 Marginal<br><sub>Small & Deficit</sub>",
        "title_static_quadrant": "Static Quadrant — All Countries",
        "axis_export_share_pct": "Export Share (%)",
        "axis_coverage_ratio": "Coverage Ratio",
        "hover_quadrant": "Quadrant",
        "hover_cagr": "CAGR",
        "hover_volatility": "Volatility",
        "quad_star_desc": "⭐ Star<br>(High CAGR · Low Vol.)",
        "quad_stable_desc": "😴 Stable Stagnant<br>(Low CAGR · Low Vol.)",
        "quad_risky_desc": "📈 Risky Grower<br>(High CAGR · High Vol.)",
        "quad_declining_desc": "📉 Declining<br>(Low CAGR · High Vol.)",
        "title_dynamic_quadrant": "Dynamic Quadrant — All Countries",
        "axis_cagr_export": "CAGR Export (%)",
        "axis_export_volatility": "Export Volatility",
        "quad_stars_desc": "⭐ Stars<br><sub>Large & Growing</sub>",
        "quad_emerging_desc": "🌱 Emerging<br><sub>Small & Growing</sub>",
        "quad_cashcows_desc": "🐄 Cash Cows<br><sub>Large & Stagnant</sub>",
        "quad_marginal2_desc": "📉 Marginal<br><sub>Small & Shrinking</sub>",
        "title_strategic_quadrant": "Strategic Quadrant (BCG-style) — All Countries",

        "expander_quad_findings": "Key findings — Quadrant distribution",
        "find_quad_1": "<strong>Static quadrant</strong>: Most countries are in Marginal or Balanced — "
                       "only ~15% qualify as Dominant (large share + surplus). Threshold: 1% export share.",
        "find_quad_2": "<strong>Dynamic quadrant</strong>: Stars (high CAGR + low volatility) dominate "
                       "large economies — Declining concentrated in commodity-dependent small traders",
        "find_quad_3": "<strong>Strategic quadrant</strong>: Emerging = small share but positive CAGR — "
                       "the most interesting group for future growth potential",
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 4 sur 6",
        "title": "Comparaison pays",
        "caption": "F.4 — Analyse multi-pays · volume · balance · part · croissance · quadrants",
        "lbl_select_countries": "Sélectionner les pays à comparer (2 à 10)",
        "warn_min_countries": "Veuillez sélectionner au moins 2 pays.",

        "tab_volume": "📦 Volume",
        "tab_balance": "⚖️ Balance",
        "tab_share": "📊 Part",
        "tab_growth": "📈 Croissance",
        "tab_quadrants": "🎯 Quadrants",

        "sec_volume": "F.4.1 · Comparaison des exportations & importations",
        "lbl_flow": "Flux",
        "opt_exports_x": "Exportations (X)",
        "opt_imports_m": "Importations (M)",
        "opt_total_trade": "Commerce total",
        "title_evolution": "Évolution — {metric} (milliards USD)",
        "unit_usd_billion": "Milliards USD",
        "title_snapshot": "{metric} — Instantané {year} (milliards USD)",

        "sec_balance": "F.4.2 · Comparaison de la balance commerciale",
        "title_balance_evolution": "Évolution de la balance commerciale (milliards USD)",
        "title_balance_snapshot": "Balance commerciale — Instantané {year}",
        "sec_coverage_ratio": "Taux de couverture",
        "balanced_100": "Équilibré (100 %)",
        "title_coverage_evolution": "Évolution du taux de couverture (Exportations / Importations × 100 %)",
        "unit_coverage_pct": "Couverture (%)",

        "sec_share": "F.4.3 · Comparaison des parts mondiales",
        "lbl_share_metric": "Indicateur de part",
        "opt_export_share": "Part des exportations",
        "opt_import_share": "Part des importations",
        "title_share_evolution": "Évolution — {metric}",
        "unit_share_pct": "Part (%)",
        "title_share_snapshot": "{metric} — Instantané {year}",

        "sec_growth": "F.4.4 · Comparaison de la croissance",
        "lbl_growth_metric": "Indicateur de croissance",
        "opt_export_growth": "Croissance des exportations",
        "opt_import_growth": "Croissance des importations",
        "title_growth_yoy": "{metric} annuelle (%)",
        "unit_pct": "%",
        "title_cagr": "CAGR {metric} — Période complète (%)",
        "sec_growth_heatmap": "Carte de chaleur de la croissance — annuelle par pays × année",
        "title_heatmap": "{metric} (%) — Année × Pays",
        "axis_year": "Année",

        "sec_quadrants": "F.4.6 · Vues globales des quadrants — Tous les pays",
        "quad_tab_static": "📐 Statique",
        "quad_tab_dynamic": "⚡ Dynamique",
        "quad_tab_strategic": "⭐ Stratégique",
        "other_countries": "Autres pays",
        "others": "Autres",
        "quad_dominant_desc": "🟢 Dominant<br><sub>Grande taille & excédent</sub>",
        "quad_balanced_desc": "🔵 Balanced<br><sub>Petite taille & excédent</sub>",
        "quad_vulnerable_desc": "🟡 Vulnerable<br><sub>Grande taille & déficit</sub>",
        "quad_marginal_desc": "🔴 Marginal<br><sub>Petite taille & déficit</sub>",
        "title_static_quadrant": "Quadrant statique — Tous les pays",
        "axis_export_share_pct": "Part des exportations (%)",
        "axis_coverage_ratio": "Taux de couverture",
        "hover_quadrant": "Quadrant",
        "hover_cagr": "CAGR",
        "hover_volatility": "Volatilité",
        "quad_star_desc": "⭐ Star<br>(CAGR élevé · faible volatilité)",
        "quad_stable_desc": "😴 Stable Stagnant<br>(CAGR faible · faible volatilité)",
        "quad_risky_desc": "📈 Risky Grower<br>(CAGR élevé · forte volatilité)",
        "quad_declining_desc": "📉 Declining<br>(CAGR faible · forte volatilité)",
        "title_dynamic_quadrant": "Quadrant dynamique — Tous les pays",
        "axis_cagr_export": "CAGR des exportations (%)",
        "axis_export_volatility": "Volatilité des exportations",
        "quad_stars_desc": "⭐ Stars<br><sub>Grande taille & croissance</sub>",
        "quad_emerging_desc": "🌱 Emerging<br><sub>Petite taille & croissance</sub>",
        "quad_cashcows_desc": "🐄 Cash Cows<br><sub>Grande taille & stagnation</sub>",
        "quad_marginal2_desc": "📉 Marginal<br><sub>Petite taille & déclin</sub>",
        "title_strategic_quadrant": "Quadrant stratégique (style BCG) — Tous les pays",

        "expander_quad_findings": "Principaux constats — Répartition des quadrants",
        "find_quad_1": "<strong>Quadrant statique</strong> : la plupart des pays sont dans Marginal ou Balanced — "
                       "seuls ~15 % se qualifient comme Dominant (grande part + excédent). Seuil : 1 % de part des exportations.",
        "find_quad_2": "<strong>Quadrant dynamique</strong> : les Stars (CAGR élevé + faible volatilité) dominent "
                       "les grandes économies — Declining se concentre chez les petits pays dépendants des matières premières",
        "find_quad_3": "<strong>Quadrant stratégique</strong> : Emerging = part faible mais CAGR positif — "
                       "le groupe le plus intéressant pour le potentiel de croissance future",
    },
}

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

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR
# ════════════════════════════════════════════════════════════════

DEFAULT = ["USA","China","France","Canada","Brazil","Qatar","India","Nigeria","Germany","Japan"]
default_valid = [c for c in DEFAULT if c in COUNTRIES]

countries = st.multiselect(
    t(STRINGS, "lbl_select_countries"),
    options=COUNTRIES,
    default=default_valid,
    max_selections=10,
    key="t4_countries",
)

if len(countries) < 2:
    st.warning(t(STRINGS, "warn_min_countries"))
    st.stop()

cmap = country_color(countries)
df_sel = df_trade[df_trade["country_name"].isin(countries)].copy()
df_str_sel = df_structural[df_structural["country_name"].isin(countries)].copy()

thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t(STRINGS, "tab_volume"),
    t(STRINGS, "tab_balance"),
    t(STRINGS, "tab_share"),
    t(STRINGS, "tab_growth"),
    t(STRINGS, "tab_quadrants"),
])


# ── TAB 1 : VOLUME ───────────────────────────────────────────
with tab1:
    section_label(t(STRINGS, "sec_volume"))

    FLOW_OPTS = ["Exports (X)", "Imports (M)", "Total Trade"]
    FLOW_LABELS = {
        "Exports (X)": t(STRINGS, "opt_exports_x"),
        "Imports (M)": t(STRINGS, "opt_imports_m"),
        "Total Trade": t(STRINGS, "opt_total_trade"),
    }
    metric = st.radio(t(STRINGS, "lbl_flow"), FLOW_OPTS,
                      format_func=lambda x: FLOW_LABELS[x],
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
            title=t(STRINGS, "title_evolution").format(metric=FLOW_LABELS[metric]),
            yaxis_title=t(STRINGS, "unit_usd_billion"),
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
            title=t(STRINGS, "title_snapshot").format(metric=FLOW_LABELS[metric], year=YEAR_MAX),
            xaxis_title=t(STRINGS, "unit_usd_billion"),
        )
        st.plotly_chart(fig_bar, width='stretch')


# ── TAB 2 : BALANCE ──────────────────────────────────────────
with tab2:
    section_label(t(STRINGS, "sec_balance"))

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
            title=t(STRINGS, "title_balance_evolution"),
            yaxis_title=t(STRINGS, "unit_usd_billion"),
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
            title=t(STRINGS, "title_balance_snapshot").format(year=YEAR_MAX),
            xaxis_title=t(STRINGS, "unit_usd_billion"),
        )
        st.plotly_chart(fig_bbar, width='stretch')

    # Coverage ratio comparison
    section_label(t(STRINGS, "sec_coverage_ratio"))
    fig_cov = go.Figure()
    for c in countries:
        df_c = df_sel[df_sel["country_name"] == c].sort_values("year")
        fig_cov.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["coverage_ratio"] * 100,
            mode="lines+markers", name=c,
            line=dict(color=cmap[c], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{c}</b><br>%{{x}}<br>" + t(STRINGS, "unit_coverage_pct") + ": %{y:.1f}%<extra></extra>",
        ))
    fig_cov.add_hline(y=100, line_dash="dash", line_color="#EEF4FB",
                      annotation_text=t(STRINGS, "balanced_100"))
    fig_cov.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_coverage_evolution"),
        yaxis_title=t(STRINGS, "unit_coverage_pct"),
    )
    st.plotly_chart(fig_cov, width='stretch')


# ── TAB 3 : SHARE ────────────────────────────────────────────
with tab3:
    section_label(t(STRINGS, "sec_share"))

    SHARE_METRIC_OPTS = ["Export Share", "Import Share"]
    SHARE_METRIC_LABELS = {
        "Export Share": t(STRINGS, "opt_export_share"),
        "Import Share": t(STRINGS, "opt_import_share"),
    }
    share_metric = st.radio(
        t(STRINGS, "lbl_share_metric"),
        SHARE_METRIC_OPTS,
        format_func=lambda x: SHARE_METRIC_LABELS[x],
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
                hovertemplate="<b>" + c + "</b><br>%{x}<br>" + SHARE_METRIC_LABELS[share_metric] + ": %{y:.3f}%<extra></extra>",
            ))
        fig_sline.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_share_evolution").format(metric=SHARE_METRIC_LABELS[share_metric]),
            yaxis_title=t(STRINGS, "unit_share_pct"),
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
            title=t(STRINGS, "title_share_snapshot").format(metric=SHARE_METRIC_LABELS[share_metric], year=YEAR_MAX),
            xaxis_title=t(STRINGS, "unit_share_pct"),
        )
        st.plotly_chart(fig_sbar, width='stretch')


# ── TAB 4 : GROWTH ───────────────────────────────────────────
with tab4:
    section_label(t(STRINGS, "sec_growth"))

    GROWTH_METRIC_OPTS = ["Export Growth", "Import Growth"]
    GROWTH_METRIC_LABELS = {
        "Export Growth": t(STRINGS, "opt_export_growth"),
        "Import Growth": t(STRINGS, "opt_import_growth"),
    }
    g_metric = st.radio(
        t(STRINGS, "lbl_growth_metric"),
        GROWTH_METRIC_OPTS,
        format_func=lambda x: GROWTH_METRIC_LABELS[x],
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
            title=t(STRINGS, "title_growth_yoy").format(metric=GROWTH_METRIC_LABELS[g_metric]),
            yaxis_title=t(STRINGS, "unit_pct"),
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
                title=t(STRINGS, "title_cagr").format(metric=GROWTH_METRIC_LABELS[g_metric]),
                xaxis_title=t(STRINGS, "unit_pct"),
            )
            st.plotly_chart(fig_cagr, width='stretch')

    # Growth heatmap
    section_label(t(STRINGS, "sec_growth_heatmap"))
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
        title=t(STRINGS, "title_heatmap").format(metric=GROWTH_METRIC_LABELS[g_metric]),
        xaxis_title=t(STRINGS, "axis_year"),
        height=max(300, len(countries) * 45 + 100),
    )
    st.plotly_chart(fig_heat, width='stretch')


# ── TAB 5 : QUADRANTS ────────────────────────────────────────
with tab5:
    section_label(t(STRINGS, "sec_quadrants"))

    quad_tabs = st.tabs([t(STRINGS, "quad_tab_static"), t(STRINGS, "quad_tab_dynamic"), t(STRINGS, "quad_tab_strategic")])

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
            mode="markers", name=t(STRINGS, "other_countries"),
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
            t(STRINGS, "axis_export_share_pct"), t(STRINGS, "axis_coverage_ratio"),
            t(STRINGS, "title_static_quadrant"),
            "static_quadrant", countries,
            quad_labels=[
                ("TR", t(STRINGS, "quad_dominant_desc"),   "#10B981"),
                ("TL", t(STRINGS, "quad_balanced_desc"),   "#60A5FA"),
                ("BR", t(STRINGS, "quad_vulnerable_desc"), "#F59E0B"),
                ("BL", t(STRINGS, "quad_marginal_desc"),   "#F472B6"),
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
            mode="markers", name=t(STRINGS, "others"),
            marker=dict(color=[DYN_COLORS.get(q,"#6B7280")
                               for q in df_dq_bg["dynamic_quadrant"]],
                        size=7, opacity=0.35, line=dict(width=0)),
            customdata=df_dq_bg[["country_name","dynamic_quadrant"]],
            hovertemplate="<b>%{customdata[0]}</b><br>" + t(STRINGS, "hover_cagr") + ": %{x:.1f}%<br>"
                          + t(STRINGS, "hover_volatility") + ": %{y:.2f}<br>" + t(STRINGS, "hover_quadrant") + ": %{customdata[1]}<extra></extra>",
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
                hovertemplate=f"<b>{cc}</b><br>" + t(STRINGS, "hover_cagr") + ": %{customdata[1]}<br>"
                              + t(STRINGS, "hover_volatility") + ": %{customdata[2]}<br>"
                              + t(STRINGS, "hover_quadrant") + ": %{customdata[0]}<extra></extra>",
            ))
        fig_dq.add_vline(x=0, line_dash="dash", line_color="#6890B0", line_width=2)
        fig_dq.add_hline(y=_vol_median, line_dash="dash", line_color="#6890B0", line_width=2)
        for txt, xp, yp, clr in [
            (t(STRINGS, "quad_star_desc"),     _dq_xr, _dq_yb, "#10B981"),
            (t(STRINGS, "quad_stable_desc"),   _dq_xl, _dq_yb, "#60A5FA"),
            (t(STRINGS, "quad_risky_desc"),    _dq_xr, _dq_yt, "#F59E0B"),
            (t(STRINGS, "quad_declining_desc"),_dq_xl, _dq_yt, "#EF4444"),
        ]:
            fig_dq.add_annotation(x=xp, y=yp, text=txt, showarrow=False,
                font=dict(size=10, color=clr),
                bgcolor="rgba(22,32,53,0.75)", borderpad=4, align="center")
        fig_dq.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                                if k not in ("xaxis","yaxis")},
            title=t(STRINGS, "title_dynamic_quadrant"),
            xaxis=dict(title=t(STRINGS, "axis_cagr_export"),
                       range=[_cagr_xlo, _cagr_xhi], gridcolor="#304870"),
            yaxis=dict(title=t(STRINGS, "axis_export_volatility"),
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
            t(STRINGS, "axis_export_share_pct"), t(STRINGS, "axis_cagr_export"),
            t(STRINGS, "title_strategic_quadrant"),
            "strategic_quadrant", countries,
            quad_labels=[
                ("TR", t(STRINGS, "quad_stars_desc"),    "#10B981"),
                ("TL", t(STRINGS, "quad_emerging_desc"), "#60A5FA"),
                ("BR", t(STRINGS, "quad_cashcows_desc"), "#A78BFA"),
                ("BL", t(STRINGS, "quad_marginal2_desc"),"#6B7280"),
            ],
            x_range=[0, _strat_xhi],
            y_range=[_strat_ylo, _strat_yhi],
        )
        st.plotly_chart(fig_strat, width='stretch')

    with st.expander(t(STRINGS, "expander_quad_findings")):
        finding(t(STRINGS, "find_quad_1"))
        finding(t(STRINGS, "find_quad_2"))
        finding(t(STRINGS, "find_quad_3"))