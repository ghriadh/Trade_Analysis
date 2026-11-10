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
from utils.i18n import t

inject_styles()

# ════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ════════════════════════════════════════════════════════════════

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 2 of 6",
        "title": "Global Trade Dynamics",
        "caption": "F.2 — Global trade system · evolution · concentration · stability · volatility",
        "kpi_world_exports": "World Exports",
        "kpi_world_imports": "World Imports",
        "kpi_countries": "Countries",
        "kpi_top10_share": "Top 10 share",

        "sec_evolution": "F.2.1 · Global Trade Evolution",
        "lbl_metric": "Metric",
        "opt_total_trade": "Total Trade",
        "opt_imports_exports": "Imports/Exports",
        "opt_growth_yoy": "Growth YoY %",
        "opt_exports": "Exports",
        "opt_imports": "Imports",
        "unit_usd_trillion": "USD Trillion",
        "unit_usd_billion": "USD Billion",
        "unit_pct": "%",
        "title_total_trade": "Global Total Trade Volume (USD Trillion)",
        "title_exports_imports": "Global Exports & Imports Volume (USD Trillion)",
        "trace_export_yoy": "Export YoY %",
        "trace_import_yoy": "Import YoY %",
        "title_yoy": "Year-over-Year Growth Rate (%)",
        "key_findings": "Key findings",
        "find_evo_1": "World trade contracted sharply in <strong>2020 (COVID)</strong> — exports fell ~7–10% — then recovered strongly in 2021–2022",
        "find_evo_2": "Growth stabilized from 2023 onward, suggesting a post-pandemic normalization rather than structural expansion",

        "sec_top_countries": "F.2.2 · Top Trading Countries",
        "lbl_year": "Year",
        "lbl_top_n": "Top N",
        "title_top_n": "Top {n} Countries by {metric} — {year}",
        "find_top_1": "The <strong>top 3 exporters</strong> (China, Germany, USA) consistently account for 25–30% of world exports",
        "find_top_2": "The ranking is remarkably stable across years — trade hierarchy changes very slowly",

        "sec_balance": "F.2.3 · Trade Balance — Surplus vs Deficit",
        "lbl_year_or_full": "Year (or Full Period Median)",
        "opt_full_period": "Full Period",
        "lbl_top_n_each_side": "Top N each side",
        "full_period_median": "Full Period (Median)",
        "trace_surplus": "Surplus",
        "trace_deficit": "Deficit",
        "title_balance": "Trade Balance — Top Surplus & Deficit — {suffix}",
        "find_bal_1": "China and Germany are the <strong>dominant surplus nations</strong> — persistent structural exporters",
        "find_bal_2": "The USA maintains the world's largest trade deficit — structural importer financing through capital flows",

        "sec_concentration": "F.2.4 · Trade Concentration & Hierarchy Stability",
        "lbl_share_hover": "share",
        "title_concentration": "Concentration Ratios — Export Share (%)",
        "lbl_share_pct": "Share (%)",
        "title_hhi": "Herfindahl-Hirschman Index (HHI)",
        "title_stability": "Hierarchy Stability (Spearman Rank Correlation)",
        "find_conc_1": "Top 10 countries control <strong>55–60% of world exports</strong> — the global trade system is highly concentrated",
        "find_conc_2": "Concentration ratios are <strong>structurally frozen</strong> — no significant redistribution over the decade despite COVID",
        "find_conc_3": "Spearman rank correlation stays above <strong>0.95</strong> — trade hierarchy is extremely stable year to year",

        "sec_volatility": "F.2.5 · Trade Volatility",
        "tab_world_map": "🗺️  World Map",
        "tab_vol_matrix": "📊  Volatility Matrix",
        "title_world_vol": "World Trade Volatility — Export Volatility",
        "cap_outliers": "⚠️ Color scale capped at 95th percentile ({cap:.1f}) for readability. "
                        "Extreme outliers (Yemen: 835, Sierra Leone: 241) are shown in max color.",
        "lbl_highlight_countries": "Highlight countries",
        "median_lbl": "Median {v:.1f}",
        "title_vol_matrix": "Volatility Matrix — Export vs Import Volatility",
        "axis_export_vol": "Export Volatility",
        "axis_import_vol": "Import Volatility",
        "cap_excluded": "⚠️ {n} extreme outlier(s) excluded from scatter "
                        "(volatility > {cap}). Visible on the world map above.",
        "find_vol_1": "Most countries cluster in the <strong>Low-Low quadrant</strong> — trade flows are generally stable",
        "find_vol_2": "Small commodity-dependent economies show the highest volatility — both export and import",
        "find_vol_3": "COVID created a volatility spike in 2020 but did not permanently elevate baseline volatility",
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 2 sur 6",
        "title": "Dynamique du commerce mondial",
        "caption": "F.2 — Système commercial mondial · évolution · concentration · stabilité · volatilité",
        "kpi_world_exports": "Exportations mondiales",
        "kpi_world_imports": "Importations mondiales",
        "kpi_countries": "Pays",
        "kpi_top10_share": "Part du top 10",

        "sec_evolution": "F.2.1 · Évolution du commerce mondial",
        "lbl_metric": "Indicateur",
        "opt_total_trade": "Commerce total",
        "opt_imports_exports": "Importations/Exportations",
        "opt_growth_yoy": "Croissance annuelle %",
        "opt_exports": "Exportations",
        "opt_imports": "Importations",
        "unit_usd_trillion": "Mille milliards USD",
        "unit_usd_billion": "Milliards USD",
        "unit_pct": "%",
        "title_total_trade": "Volume total du commerce mondial (mille milliards USD)",
        "title_exports_imports": "Volume mondial des exportations et importations (mille milliards USD)",
        "trace_export_yoy": "Croissance annuelle des exportations %",
        "trace_import_yoy": "Croissance annuelle des importations %",
        "title_yoy": "Taux de croissance annuel (%)",
        "key_findings": "Principaux constats",
        "find_evo_1": "Le commerce mondial s'est fortement contracté en <strong>2020 (COVID)</strong> — les exportations ont chuté de ~7 à 10 % — avant une forte reprise en 2021-2022",
        "find_evo_2": "La croissance s'est stabilisée à partir de 2023, suggérant une normalisation post-pandémie plutôt qu'une expansion structurelle",

        "sec_top_countries": "F.2.2 · Principaux pays commerçants",
        "lbl_year": "Année",
        "lbl_top_n": "Top N",
        "title_top_n": "Top {n} des pays par {metric} — {year}",
        "find_top_1": "Les <strong>3 premiers exportateurs</strong> (Chine, Allemagne, États-Unis) représentent systématiquement 25 à 30 % des exportations mondiales",
        "find_top_2": "Le classement est remarquablement stable d'une année à l'autre — la hiérarchie commerciale évolue très lentement",

        "sec_balance": "F.2.3 · Balance commerciale — Excédent vs déficit",
        "lbl_year_or_full": "Année (ou médiane sur la période complète)",
        "opt_full_period": "Période complète",
        "lbl_top_n_each_side": "Top N de chaque côté",
        "full_period_median": "Période complète (médiane)",
        "trace_surplus": "Excédent",
        "trace_deficit": "Déficit",
        "title_balance": "Balance commerciale — Principaux excédents et déficits — {suffix}",
        "find_bal_1": "La Chine et l'Allemagne sont les <strong>nations excédentaires dominantes</strong> — des exportateurs structurels persistants",
        "find_bal_2": "Les États-Unis conservent le plus grand déficit commercial au monde — un importateur structurel financé par les flux de capitaux",

        "sec_concentration": "F.2.4 · Concentration commerciale & stabilité de la hiérarchie",
        "lbl_share_hover": "part",
        "title_concentration": "Ratios de concentration — Part des exportations (%)",
        "lbl_share_pct": "Part (%)",
        "title_hhi": "Indice de Herfindahl-Hirschman (HHI)",
        "title_stability": "Stabilité de la hiérarchie (corrélation de rang de Spearman)",
        "find_conc_1": "Les 10 premiers pays contrôlent <strong>55 à 60 % des exportations mondiales</strong> — le système commercial mondial est très concentré",
        "find_conc_2": "Les ratios de concentration sont <strong>structurellement figés</strong> — aucune redistribution significative sur la décennie malgré la COVID",
        "find_conc_3": "La corrélation de rang de Spearman reste au-dessus de <strong>0,95</strong> — la hiérarchie commerciale est extrêmement stable d'une année à l'autre",

        "sec_volatility": "F.2.5 · Volatilité commerciale",
        "tab_world_map": "🗺️  Carte mondiale",
        "tab_vol_matrix": "📊  Matrice de volatilité",
        "title_world_vol": "Volatilité du commerce mondial — Volatilité des exportations",
        "cap_outliers": "⚠️ Échelle de couleur plafonnée au 95ᵉ centile ({cap:.1f}) pour la lisibilité. "
                        "Les valeurs extrêmes (Yémen : 835, Sierra Leone : 241) sont affichées avec la couleur maximale.",
        "lbl_highlight_countries": "Pays à mettre en évidence",
        "median_lbl": "Médiane {v:.1f}",
        "title_vol_matrix": "Matrice de volatilité — Exportations vs importations",
        "axis_export_vol": "Volatilité des exportations",
        "axis_import_vol": "Volatilité des importations",
        "cap_excluded": "⚠️ {n} valeur(s) extrême(s) exclue(s) du nuage de points "
                        "(volatilité > {cap}). Visible sur la carte mondiale ci-dessus.",
        "find_vol_1": "La plupart des pays se regroupent dans le <strong>quadrant Low-Low</strong> — les flux commerciaux sont globalement stables",
        "find_vol_2": "Les petites économies dépendantes des matières premières affichent la plus forte volatilité — à l'export comme à l'import",
        "find_vol_3": "La COVID a provoqué un pic de volatilité en 2020 mais n'a pas durablement élevé la volatilité de référence",
    },
}

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

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
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
cols[0].markdown(kpi(t(STRINGS, "kpi_world_exports"),  f"${total_exports/1e12:.1f}T", "amber"), unsafe_allow_html=True)
cols[1].markdown(kpi(t(STRINGS, "kpi_world_imports"),  f"${total_imports/1e12:.1f}T", "blue"),  unsafe_allow_html=True)
cols[2].markdown(kpi(t(STRINGS, "kpi_countries"),      str(n_countries),              "teal"),  unsafe_allow_html=True)
cols[3].markdown(kpi(t(STRINGS, "kpi_top10_share"),    f"{top10_share:.1f}%",         ""),      unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.1 — GLOBAL TRADE EVOLUTION
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "sec_evolution"))

# Canonical (English) option values are kept for internal comparisons —
# only the displayed label is translated, via format_func.
METRIC_OPTS = ["Total Trade", "Imports/Exports", "Growth YoY %"]
METRIC_OPT_LABELS = {
    "Total Trade": t(STRINGS, "opt_total_trade"),
    "Imports/Exports": t(STRINGS, "opt_imports_exports"),
    "Growth YoY %": t(STRINGS, "opt_growth_yoy"),
}
metric_sel = st.radio(
    t(STRINGS, "lbl_metric"),
    options=METRIC_OPTS,
    format_func=lambda x: METRIC_OPT_LABELS[x],
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
        mode="lines+markers", name=t(STRINGS, "opt_total_trade"),
        line=dict(color=COLORS["total"], width=2.5),
        marker=dict(size=7, color=COLORS["total"]),
        customdata=trends[["_fmt_total"]],
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_total_trade") + ": %{customdata[0]}<extra></extra>",
    ))
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_total_trade"),
        yaxis_title=t(STRINGS, "unit_usd_trillion"))

elif metric_sel == "Exports":
    fig_evo = go.Figure()
    trends["_fmt_X"] = trends["X"].apply(lambda v: f"${_fmt(v)}")
    trends["_fmt_M"] = trends["M"].apply(lambda v: f"${_fmt(v)}")
    fig_evo.add_trace(go.Scatter(
        x=trends["year"], y=trends["X"] / 1e12,
        mode="lines+markers", name=t(STRINGS, "opt_exports"),
        line=dict(color=COLORS["export"], width=2.5),
        marker=dict(size=7),
        customdata=trends[["_fmt_X"]],
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_exports") + ": %{customdata[0]}<extra></extra>",
    ))
    fig_evo.add_trace(go.Scatter(
        x=trends["year"], y=trends["M"] / 1e12,
        mode="lines+markers", name=t(STRINGS, "opt_imports"),
        line=dict(color=COLORS["import"], width=2.5, dash="dash"),
        marker=dict(size=7),
        customdata=trends[["_fmt_M"]],
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_imports") + ": %{customdata[0]}<extra></extra>",
    ))
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_exports_imports"),
        yaxis_title=t(STRINGS, "unit_usd_trillion"))

else:  # YoY %
    fig_evo = go.Figure()
    fig_evo.add_trace(go.Bar(
        x=trends["year"], y=trends["yoy_X"],
        name=t(STRINGS, "trace_export_yoy"), marker_color=COLORS["export"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "trace_export_yoy") + ": %{y:.1f}%<extra></extra>",
    ))
    fig_evo.add_trace(go.Bar(
        x=trends["year"], y=trends["yoy_M"],
        name=t(STRINGS, "trace_import_yoy"), marker_color=COLORS["import"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "trace_import_yoy") + ": %{y:.1f}%<extra></extra>",
    ))
    fig_evo.add_hline(y=0, line_color="#304870", line_width=1)
    fig_evo.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_yoy"),
        yaxis_title=t(STRINGS, "unit_pct"), barmode="group")

st.plotly_chart(fig_evo, width='stretch')

with st.expander(t(STRINGS, "key_findings")):
    finding(t(STRINGS, "find_evo_1"))
    finding(t(STRINGS, "find_evo_2"))

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.2 — TOP TRADING COUNTRIES
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "sec_top_countries"))

col1, col2 = st.columns([1, 3])
with col1:
    year_top = st.selectbox(t(STRINGS, "lbl_year"), YEARS[::-1], index=0, key="top_year")
    TOP_METRIC_OPTS = ["Exports", "Imports", "Total Trade"]
    TOP_METRIC_LABELS = {
        "Exports": t(STRINGS, "opt_exports"),
        "Imports": t(STRINGS, "opt_imports"),
        "Total Trade": t(STRINGS, "opt_total_trade"),
    }
    metric_top = st.selectbox(t(STRINGS, "lbl_metric"), TOP_METRIC_OPTS,
                               format_func=lambda x: TOP_METRIC_LABELS[x], key="top_metric")
    n_top = st.selectbox(t(STRINGS, "lbl_top_n"), [10, 15, 20], index=0, key="top_n")

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
        name=TOP_METRIC_LABELS[metric_top],
    ))
    fig_top.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_top_n").format(n=n_top, metric=TOP_METRIC_LABELS[metric_top], year=year_top),
        xaxis_title=t(STRINGS, "unit_usd_billion"),
        height=max(400, n_top * 38),
    )
    st.plotly_chart(fig_top, width='stretch')

with st.expander(t(STRINGS, "key_findings")):
    finding(t(STRINGS, "find_top_1"))
    finding(t(STRINGS, "find_top_2"))

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.3 — TRADE BALANCE
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "sec_balance"))

col1, col2 = st.columns([1, 3])
with col1:
    BAL_YEAR_OPTS = ["Full Period"] + list(YEARS[::-1])
    year_bal = st.selectbox(
        t(STRINGS, "lbl_year_or_full"),
        BAL_YEAR_OPTS,
        format_func=lambda x: t(STRINGS, "opt_full_period") if x == "Full Period" else str(x),
        index=0, key="bal_year"
    )
    n_bal = st.selectbox(t(STRINGS, "lbl_top_n_each_side"), [5, 10], index=0, key="bal_n")

if year_bal == "Full Period":
    df_bal = df_trade.groupby("country_name")["trade_balance"].median()
    title_suffix = t(STRINGS, "full_period_median")
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
        name=t(STRINGS, "trace_surplus"),
        opacity=0.85,
        customdata=surplus[["_fmt"]],
        hovertemplate="<b>%{y}</b><br>" + t(STRINGS, "trace_surplus") + ": %{customdata[0]}<extra></extra>",
    ))
    fig_bal.add_trace(go.Bar(
        x=deficit["trade_balance"] / 1e9,
        y=deficit["country_name"],
        orientation="h",
        marker_color=COLORS["import"],
        name=t(STRINGS, "trace_deficit"),
        opacity=0.85,
        customdata=deficit[["_fmt"]],
        hovertemplate="<b>%{y}</b><br>" + t(STRINGS, "trace_deficit") + ": %{customdata[0]}<extra></extra>",
    ))
    fig_bal.add_vline(x=0, line_color="#EEF4FB", line_width=1)
    fig_bal.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_balance").format(suffix=title_suffix),
        xaxis_title=t(STRINGS, "unit_usd_billion"),
        height=450,
    )
    st.plotly_chart(fig_bal, width='stretch')

with st.expander(t(STRINGS, "key_findings")):
    finding(t(STRINGS, "find_bal_1"))
    finding(t(STRINGS, "find_bal_2"))

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.4 — CONCENTRATION
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "sec_concentration"))

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
                hovertemplate=f"<b>%{{x}}</b><br>{label} " + t(STRINGS, "lbl_share_hover") + ": %{y:.1f}%<extra></extra>",
            ))
    fig_conc.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_concentration"),
        yaxis_title=t(STRINGS, "lbl_share_pct"),
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
            title=t(STRINGS, "title_hhi"),
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
            title=t(STRINGS, "title_stability"),
            yaxis_title="Spearman ρ",
        )
        st.plotly_chart(fig_stab, width='stretch')

with st.expander(t(STRINGS, "key_findings")):
    finding(t(STRINGS, "find_conc_1"))
    finding(t(STRINGS, "find_conc_2"))
    finding(t(STRINGS, "find_conc_3"))

thin_divider()


# ════════════════════════════════════════════════════════════════
# F.2.5 — VOLATILITY
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "sec_volatility"))

tab1, tab2 = st.tabs([t(STRINGS, "tab_world_map"), t(STRINGS, "tab_vol_matrix")])

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
                title=t(STRINGS, "title_world_vol"),
            )
            fig_map.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig_map, width='stretch')
            st.caption(t(STRINGS, "cap_outliers").format(cap=cap_val))

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
            t(STRINGS, "lbl_highlight_countries"),
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
                           annotation_text=t(STRINGS, "median_lbl").format(v=median_exp))
        fig_vmat.add_hline(y=median_imp, line_dash="dash", line_color="#304870",
                           annotation_text=t(STRINGS, "median_lbl").format(v=median_imp))
        # Remove size_max line (no longer needed)
        fig_vmat.update_layout(**PLOTLY_LAYOUT_COMPACT,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title=t(STRINGS, "title_vol_matrix"),
            xaxis=dict(title=t(STRINGS, "axis_export_vol"), range=[-10, 60],
                       gridcolor="#304870", linecolor="#304870"),
            yaxis=dict(title=t(STRINGS, "axis_import_vol"), range=[0, 40],
                       gridcolor="#304870", linecolor="#304870"),
            legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=560,
        )
        st.plotly_chart(fig_vmat, width='stretch')

        if n_excluded > 0:
            st.caption(t(STRINGS, "cap_excluded").format(n=n_excluded, cap=cap_x))

with st.expander(t(STRINGS, "key_findings")):
    finding(t(STRINGS, "find_vol_1"))
    finding(t(STRINGS, "find_vol_2"))
    finding(t(STRINGS, "find_vol_3"))
