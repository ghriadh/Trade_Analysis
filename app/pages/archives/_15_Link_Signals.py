"""
Trade × Indicators — Link · Signaux & Liens
Pour chaque dimension trade, quels indicateurs macro l'expliquent le mieux ?
Sections D · G · H · I · J · K du notebook Link
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, fmt_usd

inject_styles()

LINK_FILES  = ROOT / "Analysis" / "Analysis_files" / "Link"
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
IND_FILES   = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ── Indicator column names ──────────────────────────────────────
GDP    = "GDP (current US$)"
GDP_pc = "GDP per capita (current US$)"
EXP    = "Exports of goods and services (current US$)"
IMP    = "Imports of goods and services (current US$)"
POP    = "Population, total"
CA_PCT = "Current account balance (% of GDP)"
INFL   = "Inflation, consumer prices (annual %)"
UNEMP  = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
RATE    = "Real interest rate (%)"
EXPENSE = "Expense (% of GDP)"

# ── Profile colors ──────────────────────────────────────────────
_SIZE_C  = {"Leader": "#10B981", "Major": "#60A5FA", "Mid": "#F59E0B", "Small": "#6890B0"}
_COMB_C  = {"Global Champion": "#10B981", "Competitive": "#60A5FA",
            "Vulnerable": "#F59E0B", "Fragile": "#EF4444"}
_STRUC_C = {"Strong": "#10B981", "Good": "#60A5FA", "Moderate": "#F59E0B", "Weak": "#EF4444"}
_BAL_C   = {"Strong Surplus": "#10B981", "Slight Surplus": "#60A5FA",
            "Slight Deficit": "#F59E0B", "Structural Deficit": "#EF4444"}
_STATIC_C = {"Dominant": "#10B981", "Balanced": "#60A5FA",
             "Vulnerable": "#F59E0B", "Marginal": "#EF4444"}

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def build_master():
    df_str  = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_comb = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_ind  = pd.read_csv(IND_FILES / "df_indicators.csv",
                          usecols=["country_name", "entity_type",
                                   "indicator_name", "year", "value"])
    df_sp   = pd.read_csv(LINK_FILES / "df_spearman.csv")
    df_kw   = pd.read_csv(LINK_FILES / "df_kw.csv")

    ci = df_ind[df_ind["entity_type"] == "COUNTRY"].copy()

    # Latest macro value per country
    df_latest = (ci.sort_values("year")
                 .groupby(["country_name", "indicator_name"])["value"].last()
                 .unstack().reset_index())
    df_latest.columns.name = None

    # CAGR of GDP (for trade elasticity)
    df_gdp = ci[ci["indicator_name"] == GDP].sort_values(["country_name", "year"])
    g_first = df_gdp.groupby("country_name")["value"].first()
    g_last  = df_gdp.groupby("country_name")["value"].last()
    g_ny    = df_gdp.groupby("country_name")["year"].agg(lambda x: x.max() - x.min())
    cagr_gdp = (
        (g_last / g_first.replace(0, np.nan))
        .pow(1 / g_ny.replace(0, np.nan)) - 1
    ).rename("cagr_gdp").reset_index()

    # Build master — combine all sources
    # Only pull columns unique to df_comb (avoid merge conflicts with df_str overlaps)
    _comb_unique = ["country_name", "combined_score", "f_norm_rank", "g_norm_rank",
                    "combined_profile", "strategic_quadrant_combined"]
    df_m = (df_str
            .merge(df_comb[[c for c in _comb_unique if c in df_comb.columns]],
                   on="country_name", how="inner")
            .merge(df_latest, on="country_name", how="left")
            .merge(cagr_gdp,  on="country_name", how="left"))

    # Derived metrics
    df_m["trade_intensity"]    = (df_m[EXP] + df_m[IMP]) / df_m[GDP]
    df_m["export_dep"]         = df_m[EXP] / df_m[GDP]
    df_m["import_dep"]         = df_m[IMP] / df_m[GDP]
    df_m["balance_dep"]        = (df_m[EXP] - df_m[IMP]) / df_m[GDP]
    df_m["export_pc"]          = df_m["total_export"] / df_m[POP]
    df_m["import_pc"]          = df_m["total_import"] / df_m[POP]
    _safe_gdp                  = df_m["cagr_gdp"].replace(0, np.nan)
    df_m["trade_elasticity_x"] = df_m["cagr_export"] / _safe_gdp
    df_m["trade_elasticity_m"] = df_m["cagr_import"] / _safe_gdp

    return df_m, df_sp, df_kw

df_m, df_sp, df_kw = build_master()

# ════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1F3050", plot_bgcolor="#162035",
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


def _scatter(df, x, y, color_col, color_map, title, xtitle, ytitle,
             xlog=False, ylog=False, height=400, x_clip=None, y_clip=None,
             x_fmt=None, y_fmt=None):
    """Scatter plot colored by categorical profile column.
    x_fmt / y_fmt: None = 3 decimal places, "usd" = $K/B/T
    """
    df_p = df[[x, y, color_col, "country_name"]].replace([np.inf, -np.inf], np.nan).dropna()
    if x_clip:
        df_p = df_p[(df_p[x] >= x_clip[0]) & (df_p[x] <= x_clip[1])]
    if y_clip:
        df_p = df_p[(df_p[y] >= y_clip[0]) & (df_p[y] <= y_clip[1])]
    _f = lambda v, fmt: (fmt_usd(v) if fmt == "usd" else f"{v:.3f}")
    df_p = df_p.copy()
    df_p["_xf"] = df_p[x].apply(lambda v: _f(v, x_fmt))
    df_p["_yf"] = df_p[y].apply(lambda v: _f(v, y_fmt))
    _ht = (f"<b>%{{text}}</b><br>{xtitle}: %{{customdata[0]}}<br>"
           f"{ytitle}: %{{customdata[1]}}<extra></extra>")
    fig = go.Figure()
    for cat in sorted(df_p[color_col].unique()):
        sub = df_p[df_p[color_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y],
            mode="markers",
            name=cat,
            customdata=sub[["_xf", "_yf"]].values,
            marker=dict(color=color_map.get(cat, "#304870"), size=7, opacity=0.8),
            text=sub["country_name"],
            hovertemplate=_ht,
        ))
    _pl(fig, title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=height,
        xaxis=dict(type="log" if xlog else "linear", gridcolor="#304870"),
        yaxis=dict(type="log" if ylog else "linear", gridcolor="#304870"))
    return fig


def _top_correlates(trade_var, n=6, flag_filter=None):
    """Return top correlated macro indicators for a given trade variable."""
    df_t = df_sp[df_sp["trade_var"] == trade_var].copy()
    if flag_filter:
        df_t = df_t[df_t["ind_flag"] == flag_filter]
    return df_t.sort_values("abs_rho", ascending=False).head(n)


def _corr_bar(df_tc, title, height=320):
    """Horizontal bar chart of top correlates."""
    _fc = {"[E]": "#10B981", "[T]": "#EF4444", "[S]": "#F59E0B"}
    colors = [_fc.get(f, "#6890B0") for f in df_tc["ind_flag"]]
    fig = go.Figure(go.Bar(
        x=df_tc["abs_rho"].values[::-1],
        y=df_tc["ind_label"].values[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=df_tc["rho"].values[::-1].round(3),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ = %{text}<extra></extra>",
    ))
    _pl(fig, title=title, xaxis_title="|ρ|", height=height,
        margin=dict(l=160, r=60, t=50, b=40),
        xaxis=dict(gridcolor="#304870", range=[0, 1.05]))
    return fig


def _tab_heatmap(vars_dict, title, flags=None, height=None):
    """Spearman ρ heatmap: rows = trade variables, cols = macro indicators.
    vars_dict: {trade_var_key: display_label}
    flags: list of ind_flag to include — defaults to [E] + [S], never [T]
    """
    _flags = flags or ["[E]", "[S]"]
    _df = df_sp[
        df_sp["trade_var"].isin(vars_dict.keys()) &
        df_sp["ind_flag"].isin(_flags)
    ].copy()
    if _df.empty:
        return None
    _piv = (_df.pivot_table(index="trade_var", columns="ind_label",
                             values="rho", aggfunc="first")
              .reindex(list(vars_dict.keys())).dropna(how="all"))
    _piv_sig = (_df.pivot_table(index="trade_var", columns="ind_label",
                                 values="sig", aggfunc="first")
                  .reindex(list(vars_dict.keys())))
    _ylabs = [vars_dict.get(v, v) for v in _piv.index]
    _text = []
    for r in _piv.index:
        _row = []
        for c in _piv.columns:
            v = _piv.at[r, c] if c in _piv.columns else np.nan
            try:
                s = _piv_sig.at[r, c]
                sig = bool(s) if not pd.isna(s) else False
            except Exception:
                sig = False
            if pd.isna(v):
                _row.append("")
            else:
                _row.append(f"{v:+.2f}{'*' if sig else ''}")
        _text.append(_row)
    _n_rows = max(len(_piv), 1)
    _n_cols = max(len(_piv.columns), 1)
    _row_h  = 62
    _col_w  = 90
    _h = height or max(300, _row_h * _n_rows + 220)
    _w_est  = _col_w * _n_cols + 250          # approx width for colorbar placement
    fig = go.Figure(go.Heatmap(
        z=_piv.values,
        x=list(_piv.columns),
        y=_ylabs,
        text=_text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[
            [0.0,  "#1060B0"],
            [0.35, "#1A3060"],
            [0.5,  "#1A2840"],
            [0.65, "#1A3820"],
            [1.0,  "#10B981"],
        ],
        zmid=0, zmin=-0.6, zmax=0.6,
        showscale=True,
        colorbar=dict(
            title=dict(text="ρ", font=dict(color="#7090B0", size=12)),
            thickness=14,
            len=0.7,
            yanchor="middle", y=0.5,
            tickvals=[-0.5, -0.25, 0, 0.25, 0.5],
            tickfont=dict(color="#7090B0", size=10),
            outlinewidth=0,
        ),
    ))
    _pl(fig, title=title, height=_h,
        margin=dict(l=190, r=80, t=55, b=170),
        xaxis=dict(
            tickangle=-40,
            side="bottom",
            gridcolor="#304870",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#304870",
            tickfont=dict(size=11),
            autorange="reversed",
        ))
    return fig


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Link · Trade × Indicators · Page 3 of 4</div>',
            unsafe_allow_html=True)
st.title("Signaux & Liens")
st.caption(
    "Pour chaque dimension trade, quel indicateur macro l'explique le mieux ? "
    "Sections D · G · H · I · J · K — 16 indicateurs × 44 variables trade"
)
thin_divider()

# ── KPI row ────────────────────────────────────────────────────
_n_sig_e = int(df_sp[(df_sp["sig"]) & (df_sp["ind_flag"] == "[E]")].shape[0])
_top_pair = df_sp[df_sp["sig"]].nlargest(1, "abs_rho").iloc[0]
_top_e    = (df_sp[(df_sp["ind_flag"] == "[E]") & df_sp["sig"]]
             .groupby("ind_label")["abs_rho"].mean().idxmax())

for col, (label, val, color) in zip(st.columns(4), [
    ("Pays analysés",      str(len(df_m)),                      "teal"),
    ("Sig. [E] paires",    str(_n_sig_e),                       "blue"),
    ("Signal max (tous)",  f"ρ={_top_pair['rho']:+.3f}",        "amber"),
    ("Top signal [E]",     _top_e,                              ""),
]):
    col.markdown(kpi(label, val, color), unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# E+F — CLASSIFICATION DES INDICATEURS
# ════════════════════════════════════════════════════════════════

section_label("E+F · Classification des indicateurs — Base analytique")
st.markdown(
    "<p style='font-size:13px;color:#7090B0;margin-bottom:.9rem;'>"
    "Avant d'analyser chaque dimension trade (G–K), les sections E et F du notebook "
    "ont classé les 16 indicateurs macro en trois familles selon leur nature analytique. "
    "Cette classification est utilisée tout au long de la page pour distinguer "
    "les signaux causaux des artefacts statistiques.</p>",
    unsafe_allow_html=True,
)

_flag_cols = st.columns(3)
with _flag_cols[0]:
    st.markdown(
        '<div style="background:#201820;border:1px solid #38242A;'
        'border-left:4px solid #EF4444;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        '<span style="color:#EF4444;font-size:13px;font-weight:700;">'
        '[T] Tautological</span>'
        '<span style="color:#A8B8CC;font-size:13px;"> — indicator is directly derived '
        'from trade flows (e.g. Exports USD). '
        'High correlation is structural, not causal.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _flag_cols[1]:
    st.markdown(
        '<div style="background:#201E18;border:1px solid #383020;'
        'border-left:4px solid #F59E0B;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        '<span style="color:#F59E0B;font-size:13px;font-weight:700;">'
        '[S] Size-driven</span>'
        '<span style="color:#A8B8CC;font-size:13px;"> — indicator reflects country '
        'size (GDP, Population). Correlation with trade volumes is scale effect, '
        'not economic signal.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _flag_cols[2]:
    st.markdown(
        '<div style="background:#182020;border:1px solid #243830;'
        'border-left:4px solid #10B981;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        '<span style="color:#10B981;font-size:13px;font-weight:700;">'
        '[E] Economic</span>'
        '<span style="color:#A8B8CC;font-size:13px;"> — genuine macro signal '
        'independent of trade definition (e.g. GDP/capita, Inflation). '
        'Correlations here are analytically meaningful.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "<p style='font-size:12px;color:#4A6890;margin:.6rem 0 .5rem 0;'>"
    "<strong style='color:#6890B0;'>Section E</strong> · 704 paires Spearman "
    "(44 variables × 16 indicateurs). "
    "<strong style='color:#6890B0;'>Section F</strong> · RF Feature Importance + "
    "KW η² confirment le classement [E] vs [S]. "
    "Les heatmaps de corrélation sont disponibles au début de chaque onglet ci-dessous "
    "— elles montrent quels indicateurs [E] (et [S]) expliquent chaque famille de variables.</p>",
    unsafe_allow_html=True,
)
thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_d, tab_gh, tab_ij, tab_k = st.tabs([
    "D · Trade/GDP Bridge",
    "G–H · Exports & Imports",
    "I–J · Balance & Profils",
    "K · Scores",
])

# ────────────────────────────────────────────────────────────────
# TAB D — TRADE / GDP BRIDGE
# ────────────────────────────────────────────────────────────────

def _question(text):
    st.markdown(
        f'<div style="background:#1A2840;border:1px solid #304870;'
        f'border-left:3px solid #60A5FA;border-radius:0 10px 10px 0;'
        f'padding:.6rem 1rem;margin-bottom:.8rem;">'
        f'<span style="color:#93C5FD;font-size:11px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.07em;">Question</span><br>'
        f'<span style="color:#EEF4FB;font-size:14px;">{text}</span></div>',
        unsafe_allow_html=True,
    )

def _note(text):
    st.markdown(
        f'<div style="background:#182820;border:1px solid #2A4034;'
        f'border-left:3px solid #34D399;border-radius:0 10px 10px 0;'
        f'padding:.5rem .9rem;margin:.4rem 0 .7rem 0;">'
        f'<span style="color:#34D399;font-size:11px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.07em;">Note analytique</span><br>'
        f'<span style="color:#9DBFB0;font-size:13px;">{text}</span></div>',
        unsafe_allow_html=True,
    )

def _classify_elast(cx, cg):
    if cx > 0 and cg > 0:
        return "Export-led" if cx > cg else "GDP-led"
    elif cx > 0 and cg <= 0:
        return "Export résilient"
    elif cx <= 0 and cg > 0:
        return "Contraction exports"
    elif cx < 0 and cg < 0:
        return "Déclin modéré" if cx > cg else "Déclin sévère"
    return "Autre"

_ELAST_C = {
    "Export-led":         "#10B981",
    "GDP-led":            "#60A5FA",
    "Export résilient":   "#A78BFA",
    "Contraction exports":"#F59E0B",
    "Déclin modéré":      "#FB923C",
    "Déclin sévère":      "#EF4444",
}


def _kw_box(col, profile_col, order, color_map, height=390, fmt_fn=None):
    """Boxplot of `col` grouped by `profile_col`, ordered by `order`.
    fmt_fn: optional callable to format y values for hover (e.g. fmt_usd).
    """
    fig = go.Figure()
    for cat in order:
        sub = df_m[df_m[profile_col] == cat][["country_name", col]].dropna().copy()
        if sub.empty:
            continue
        if fmt_fn:
            sub["_yf"] = sub[col].apply(fmt_fn)
            fig.add_trace(go.Box(
                y=sub[col].values, name=cat,
                text=sub["country_name"].values,
                customdata=sub[["_yf"]].values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<extra></extra>",
                marker_color=color_map.get(cat, "#304870"),
                boxpoints="outliers", line=dict(width=1.5),
            ))
        else:
            fig.add_trace(go.Box(
                y=sub[col].values, name=cat,
                text=sub["country_name"].values,
                hovertemplate="<b>%{text}</b><br>%{y:.3f}<extra></extra>",
                marker_color=color_map.get(cat, "#304870"),
                boxpoints="outliers", line=dict(width=1.5),
            ))
    return fig


def _add_refs(fig, x_col, y_col, countries, df=None, x_scale=1, y_scale=1, ht=None,
              x_fmt=None, y_fmt=None):
    """Add gold star markers for reference countries on a scatter figure."""
    _src = df if df is not None else df_m
    _ref = _src[_src["country_name"].isin(countries)][[x_col, y_col, "country_name"]].dropna()
    if _ref.empty:
        return fig
    _xv = _ref[x_col].values * x_scale
    _yv = _ref[y_col].values * y_scale
    if ht:
        _cd, _ht = None, ht
    else:
        _f   = lambda v, fmt: (fmt_usd(v) if fmt == "usd" else f"{v:.3f}")
        _cd  = list(zip([_f(v, x_fmt) for v in _xv], [_f(v, y_fmt) for v in _yv]))
        _ht  = "<b>%{text}</b><br>x: %{customdata[0]}<br>y: %{customdata[1]}<extra></extra>"
    fig.add_trace(go.Scatter(
        x=_xv, y=_yv,
        mode="markers+text",
        marker=dict(color="#FFD700", size=11, symbol="star",
                    line=dict(color="#FFFFFF", width=1.5)),
        text=_ref["country_name"].values,
        customdata=_cd,
        textposition="top center",
        textfont=dict(size=9, color="#FFD700"),
        hovertemplate=_ht,
        name="Référence", showlegend=False,
    ))
    return fig


with tab_d:
    section_label("D · Trade/GDP Bridge — Métriques Dérivées")
    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>"
        "Trois métriques dérivées qui relient directement structure trade et macro : "
        "Trade Intensity = (X+M)/GDP · Élasticité = CAGR_export/CAGR_GDP · "
        "Trade per Capita = volume normalisé par habitant.</p>",
        unsafe_allow_html=True,
    )
    _D_VARS = {
        "trade_intensity": "Trade Intensity (X+M)/GDP",
        "export_dep":      "Export / GDP (X/GDP)",
        "export_pc":       "Export per Capita",
        "cagr_export":     "CAGR Export",
    }
    _fig_d_heat = _tab_heatmap(
        _D_VARS,
        "Corrélations [E] — Variables Trade/GDP Bridge (Tab D) · * = p<0.05",
        flags=["[E]"],
    )
    if _fig_d_heat:
        st.plotly_chart(_fig_d_heat, use_container_width=True)
    thin_divider()

    # ── D.1 ────────────────────────────────────────────────────
    section_label("D.1 · Trade Intensity")
    _question("La richesse d'un pays détermine-t-elle son ouverture commerciale ?")

    df_d1 = df_m[["country_name", GDP_pc, "trade_intensity", "size_profile"]].dropna()
    df_d1 = df_d1[(df_d1[GDP_pc] > 0) & (df_d1["trade_intensity"] > 0)
                  & (df_d1["trade_intensity"] < 5)]
    fig_d1 = _scatter(df_d1, GDP_pc, "trade_intensity", "size_profile", _SIZE_C,
                      "Trade Intensity (X+M)/GDP vs GDP/capita",
                      "GDP per capita (US$, log)", "Trade Intensity (X+M)/GDP",
                      xlog=True, height=460, x_fmt="usd")
    _add_refs(fig_d1, GDP_pc, "trade_intensity",
              ["USA", "China", "Singapore", "Luxembourg"], df=df_d1, x_fmt="usd")
    st.plotly_chart(fig_d1, use_container_width=True)
    finding(
        "<strong>Oui, mais non-linéaire</strong> : les pays plus riches tendent vers une "
        "intensité trade plus élevée, avec une dispersion croissante. Les grands pays "
        "(Leaders) ont des intensités variables selon leur structure domestique — "
        "USA et Chine ont une intensité modérée malgré leur taille, "
        "les petits pays riches (Singapour, Luxembourg) sont les plus ouverts."
    )

    thin_divider()

    # ── D.2 ────────────────────────────────────────────────────
    section_label("D.2 · Élasticité Trade/PIB — 6 Situations")
    _question("Le commerce extérieur croît-il plus vite que l'économie ? Dans quels profils ?")

    df_d2 = df_m[["country_name", "cagr_gdp", "cagr_export", "structural_profile"]].copy()
    df_d2 = df_d2.replace([np.inf, -np.inf], np.nan).dropna(subset=["cagr_gdp", "cagr_export"])
    df_d2 = df_d2[df_d2["cagr_gdp"].abs() < 0.3]
    df_d2 = df_d2[df_d2["cagr_export"].abs() < 0.5]
    df_d2["situation"] = df_d2.apply(
        lambda r: _classify_elast(r["cagr_export"], r["cagr_gdp"]), axis=1)

    col_d2a, col_d2b = st.columns([3, 2])

    with col_d2a:
        fig_d2 = go.Figure()
        for sit, color in _ELAST_C.items():
            sub = df_d2[df_d2["situation"] == sit]
            if sub.empty:
                continue
            fig_d2.add_trace(go.Scatter(
                x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                mode="markers", name=sit,
                marker=dict(color=color, size=7, opacity=0.85),
                text=sub["country_name"],
                hovertemplate=(
                    "<b>%{text}</b> · %{meta}<br>"
                    "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                ),
                meta=sit,
            ))
        _lim = df_d2[["cagr_gdp", "cagr_export"]].stack().abs().quantile(0.95) * 100
        fig_d2.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                         line=dict(color="#6890B0", dash="dash", width=1.5))
        fig_d2.add_vline(x=0, line_color="#304870", line_dash="dot", line_width=1)
        fig_d2.add_hline(y=0, line_color="#304870", line_dash="dot", line_width=1)
        for ann_x, ann_y, ann_txt in [
            (_lim * 0.6,  _lim * 0.95, "Export-led ↗"),
            (_lim * 0.6,  _lim * 0.3,  "GDP-led ↗"),
            (-_lim * 0.6, _lim * 0.6,  "Export résilient"),
            (_lim * 0.5,  -_lim * 0.7, "Contraction exports"),
        ]:
            fig_d2.add_annotation(x=ann_x, y=ann_y, text=ann_txt, showarrow=False,
                                  font=dict(size=10, color="#6890B0"))
        _pl(fig_d2, title="CAGR Export vs CAGR GDP — diagonale = élasticité 1",
            xaxis_title="CAGR GDP (%)", yaxis_title="CAGR Export (%)", height=420,
            legend=dict(orientation="v", x=1.01, y=1, bgcolor="#1F3050",
                        bordercolor="#304870", borderwidth=1))
        st.plotly_chart(fig_d2, use_container_width=True)

    with col_d2b:
        sit_counts = df_d2["situation"].value_counts().reindex(list(_ELAST_C.keys())).dropna()
        fig_d2b = go.Figure(go.Bar(
            x=sit_counts.values,
            y=sit_counts.index,
            orientation="h",
            marker_color=[_ELAST_C[s] for s in sit_counts.index],
            text=sit_counts.values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>n = %{x}<extra></extra>",
        ))
        _pl(fig_d2b, title="Distribution des pays<br>par situation (2015–2024)",
            xaxis_title="Nb pays", height=420,
            margin=dict(l=150, r=50, t=70, b=40),
            xaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_d2b, use_container_width=True)

    finding(
        "<strong>Majorité GDP-led</strong> : la plupart des pays ont une croissance export "
        "plus lente que leur PIB (élasticité < 1). Les pays Export-led sont souvent des "
        "économies émergentes en phase d'intégration mondiale. "
        "<strong>Export résilient</strong> : pays dont les exports croissent même quand "
        "l'économie se contracte — signe d'une compétitivité externe robuste."
    )

    with st.expander("Définition des 6 situations"):
        st.markdown("""
        <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Situation</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Condition</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Interprétation</th>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#10B981;font-weight:600;">Export-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; CAGR_GDP &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              Les exports croissent plus vite que l'économie — moteur externe dominant</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#60A5FA;font-weight:600;">GDP-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_GDP &gt; CAGR_X &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              L'économie croît plus vite que les exports — croissance tirée par la demande interne</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#A78BFA;font-weight:600;">Export résilient</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; 0, CAGR_GDP ≤ 0</td>
            <td style="padding:.4rem .8rem;">
              Les exports progressent alors que l'économie se contracte — compétitivité externe robuste</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#F59E0B;font-weight:600;">Contraction exports</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X ≤ 0, CAGR_GDP &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              L'économie croît mais les exports reculent — substitution domestique ou perte de compétitivité</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#FB923C;font-weight:600;">Déclin modéré</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_GDP &lt; 0, |X| &lt; |GDP|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, exports résistent mieux que le PIB</td>
          </tr>
          <tr>
            <td style="padding:.4rem .8rem;">
              <span style="color:#EF4444;font-weight:600;">Déclin sévère</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_GDP &lt; 0, |X| &gt; |GDP|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, exports s'effondrent plus vite que le PIB</td>
          </tr>
        </table>
        <div style="margin-top:.8rem;font-size:12px;color:#4A6890;">
          CAGR_X = taux de croissance annuel composé des exports (2015–2024) ·
          CAGR_GDP = même calcul pour le PIB nominal ·
          Élasticité = CAGR_X / CAGR_GDP
        </div>
        </div>
        """, unsafe_allow_html=True)

    thin_divider()

    # ── D.3 ────────────────────────────────────────────────────
    section_label("D.3 · Export per Capita vs GDP/capita")
    _question(
        "Un pays riche exporte-t-il plus par habitant ? "
        "Quels pays se repositionnent quand on normalise par la population ?"
    )

    col_d3a, col_d3b = st.columns(2)

    with col_d3a:
        df_d3 = df_m[["country_name", GDP_pc, "export_pc", "combined_profile"]].dropna()
        df_d3 = df_d3[(df_d3[GDP_pc] > 0) & (df_d3["export_pc"] > 0)]
        fig_d3 = _scatter(df_d3, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                          "Export per Capita vs GDP/capita",
                          "GDP per capita (US$, log)", "Export per Capita (US$, log)",
                          xlog=True, ylog=True, height=440, x_fmt="usd", y_fmt="usd")
        st.plotly_chart(fig_d3, use_container_width=True)

    with col_d3b:
        # Rank mover chart: rank by total volume vs rank by per capita
        df_rk = df_m[["country_name", "total_export", "export_pc"]].dropna()
        df_rk["rank_vol"] = df_rk["total_export"].rank(ascending=False)
        df_rk["rank_pc"]  = df_rk["export_pc"].rank(ascending=False)
        df_rk["delta"]    = df_rk["rank_vol"] - df_rk["rank_pc"]  # >0 = gains per capita
        n_show = 12
        top_gain = df_rk.nlargest(n_show, "delta")
        top_lose = df_rk.nsmallest(n_show, "delta")
        df_rk_show = pd.concat([top_gain, top_lose]).drop_duplicates("country_name")
        df_rk_show = df_rk_show.sort_values("delta")

        fig_rk = go.Figure(go.Bar(
            x=df_rk_show["delta"],
            y=df_rk_show["country_name"],
            orientation="h",
            marker_color=["#10B981" if v > 0 else "#EF4444"
                          for v in df_rk_show["delta"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Rang volume: %{customdata[0]:.0f}<br>"
                "Rang per capita: %{customdata[1]:.0f}<br>"
                "Δ = %{x:+.0f}<extra></extra>"
            ),
            customdata=df_rk_show[["rank_vol", "rank_pc"]].values,
        ))
        fig_rk.add_vline(x=0, line_color="#6890B0", line_width=1)
        _pl(fig_rk,
            title="Rank Mover — Volume vs Per Capita<br>"
                  "<span style='font-size:11px;color:#6890B0;'>"
                  "Δ > 0 : meilleur rang per capita qu'en volume</span>",
            xaxis_title="Δ rang (volume − per capita)",
            height=440,
            margin=dict(l=140, r=40, t=70, b=40),
            xaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_rk, use_container_width=True)

    finding(
        "<strong>Oui, corrélation forte</strong> (ρ≈0.82) : les pays riches exportent "
        "davantage par habitant. <strong>Rank Movers</strong> : les grands pays pauvres "
        "(Chine, Inde) perdent des rangs quand on normalise — leur volume est lié à "
        "la taille, pas à la productivité. À l'inverse, les petits pays riches "
        "(Luxembourg, Singapour, Suisse) gagnent massivement en rang per capita."
    )

# ────────────────────────────────────────────────────────────────
# TAB G-H — EXPORTS & IMPORTS
# ────────────────────────────────────────────────────────────────
with tab_gh:
    _GH_VARS = {
        "export_share_structural": "Export Share mondial",
        "export_dep":              "Export / GDP",
        "export_pc":               "Export / Capita",
        "cagr_export":             "CAGR Export",
        "volatility_export":       "Volatilité Export",
        "import_share_structural": "Import Share mondial",
        "import_dep":              "Import / GDP",
        "import_pc":               "Import / Capita",
        "cagr_import":             "CAGR Import",
        "volatility_import":       "Volatilité Import",
    }
    _fig_gh_heat = _tab_heatmap(
        _GH_VARS,
        "Corrélations [E]+[S] — Variables Export & Import (Tabs G–H) · * = p<0.05",
        flags=["[E]", "[S]"],
    )
    if _fig_gh_heat:
        st.plotly_chart(_fig_gh_heat, use_container_width=True)
    thin_divider()

    subtab_g, subtab_h, subtab_synth = st.tabs([
        "G · Exports", "H · Imports", "G+H · Synthèse"
    ])

    # ════════════════════════════════════════════════════
    # G · EXPORTS
    # ════════════════════════════════════════════════════
    with subtab_g:

        # ── G.1 Export Level ─────────────────────────
        section_label("G.1 · Export Level — Variables expliquées par les fondamentaux macro")
        st.markdown(
            "<p style='font-size:13px;color:#7090B0;'>G.1 couvre 4 variables : "
            "Export Share mondiale, Export/GDP (ouverture), Export per Capita. "
            "Toutes répondent positivement aux indicateurs [S] et [E].</p>",
            unsafe_allow_html=True,
        )

        # G.1.1 Export Share vs GDP
        _question("La taille économique absolue détermine-t-elle la part mondiale des exports ?")
        col_g11a, col_g11b = st.columns(2)
        with col_g11a:
            df_g11 = df_m[["country_name", GDP, "export_share_structural",
                            "size_profile"]].dropna()
            df_g11 = df_g11[df_g11[GDP] > 0]
            fig_g11 = _scatter(df_g11, GDP, "export_share_structural", "size_profile", _SIZE_C,
                               "Export Share vs GDP (ρ[S]=+0.94)",
                               "GDP (US$, log)", "Export Share (%)",
                               xlog=True, height=390, x_fmt="usd")
            st.plotly_chart(fig_g11, use_container_width=True)
        with col_g11b:
            df_g11b = df_m[["country_name", GDP_pc, "export_pc",
                             "combined_profile"]].dropna()
            df_g11b = df_g11b[(df_g11b[GDP_pc] > 0) & (df_g11b["export_pc"] > 0)]
            fig_g11b = _scatter(df_g11b, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                                "Export per Capita vs GDP/capita (ρ[E]≈+0.82)",
                                "GDP per capita (US$, log)", "Export per Capita (US$, log)",
                                xlog=True, ylog=True, height=390, x_fmt="usd", y_fmt="usd")
            _add_refs(fig_g11b, GDP_pc, "export_pc",
                      ["USA", "China", "Germany", "Japan", "Brazil"], df=df_g11b,
                      x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_g11b, use_container_width=True)
        finding(
            "<strong>Déterminisme taille [S]</strong> : GDP (US$) ρ=+0.94 — "
            "la taille économique absolue explique quasi-mécaniquement la part mondiale. "
            "Signal [E] secondaire : CA% GDP ρ=+0.48 — les pays excédentaires exportent "
            "davantage que leur taille ne l'implique. "
            "Seuil dur à ~$259Md : en dessous, export share proche de zéro."
        )
        thin_divider()

        # G.1.2 Export Share vs CA% + Trade Efficiency
        _question("Un excédent courant booste-t-il la part export ? "
                  "Qui sur/sous-performe par rapport à sa taille ?")
        col_g12a, col_g12b = st.columns(2)
        with col_g12a:
            df_g12 = df_m[["country_name", CA_PCT, "export_share_structural",
                            "structural_profile"]].dropna()
            df_g12 = df_g12[df_g12["export_share_structural"] < 0.15]
            fig_g12 = _scatter(df_g12, CA_PCT, "export_share_structural",
                               "structural_profile", _STRUC_C,
                               "Export Share vs Current Account % GDP (ρ[E]=+0.48)",
                               "CA% GDP", "Export Share (%)", height=390)
            st.plotly_chart(fig_g12, use_container_width=True)
        with col_g12b:
            df_eff = df_m[["country_name", GDP, "export_share_structural",
                            "size_profile"]].dropna()
            df_eff = df_eff[df_eff[GDP] > 0].copy()
            world_gdp = df_eff[GDP].sum()
            df_eff["gdp_share"] = df_eff[GDP] / world_gdp
            df_eff["trade_eff"] = (df_eff["export_share_structural"]
                                   / df_eff["gdp_share"].replace(0, np.nan))
            df_eff = df_eff[(df_eff["trade_eff"] > 0) & (df_eff["trade_eff"] < 12)]
            top_eff = df_eff.nlargest(14, "trade_eff")
            bot_eff = df_eff.nsmallest(9, "trade_eff")
            df_eff_show = (pd.concat([top_eff, bot_eff])
                           .drop_duplicates("country_name")
                           .sort_values("trade_eff"))
            fig_eff = go.Figure(go.Bar(
                x=df_eff_show["trade_eff"],
                y=df_eff_show["country_name"],
                orientation="h",
                marker_color=["#10B981" if v > 1 else "#EF4444"
                              for v in df_eff_show["trade_eff"]],
                text=df_eff_show["trade_eff"].round(1),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Efficience = %{x:.2f}×<extra></extra>",
            ))
            fig_eff.add_vline(x=1, line_color="#6890B0", line_dash="dash",
                              line_width=1, annotation_text="Référence (1×)",
                              annotation_font_color="#6890B0")
            _pl(fig_eff,
                title="Trade Efficiency = Export Share / GDP Share",
                xaxis_title="Export Share / GDP Share (×)",
                height=390, margin=dict(l=130, r=70, t=50, b=40))
            st.plotly_chart(fig_eff, use_container_width=True)
        finding(
            "<strong>Trade Efficiency Ratio</strong> = Export Share / GDP Share. "
            "> 1 = sur-performeur (exporte plus que son poids économique). "
            "<strong>Germany ~2.6×</strong> : modèle export-industriel. "
            "<strong>China ~2.4×</strong> : intégration mondiale massive. "
            "<strong>USA ~0.65×</strong> : sous-performeur — économie domestique dominante. "
            "CA% [E] (ρ=+0.48) : les pays excédentaires sur-performent structurellement."
        )
        thin_divider()

        # G.1.2.3 KW Boxplots — profils de taille × fondamentaux macro
        _question("Les profils de taille regroupent-ils des pays "
                  "aux fondamentaux macro structurellement différents ?")
        col_kw1, col_kw2 = st.columns(2)
        _kw_order = ["Leader", "Major", "Mid", "Small"]
        with col_kw1:
            fig_kw1 = go.Figure()
            for cat in _kw_order:
                sub = df_m[df_m["size_profile"] == cat][[GDP_pc, "country_name"]].dropna().copy()
                sub["_yf"] = sub[GDP_pc].apply(fmt_usd)
                fig_kw1.add_trace(go.Box(
                    y=sub[GDP_pc].values, name=cat,
                    text=sub["country_name"].values,
                    customdata=sub[["_yf"]].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>GDP/capita=%{customdata[0]}<extra></extra>",
                ))
            _pl(fig_kw1,
                title="GDP per Capita par profil de taille (KW η² large)",
                yaxis_title="GDP per Capita (US$)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_kw1, use_container_width=True)
        with col_kw2:
            fig_kw2 = go.Figure()
            df_kw2 = df_m[["country_name", "size_profile", CA_PCT]].dropna()
            df_kw2 = df_kw2[df_kw2[CA_PCT].abs() < 25]
            for cat in _kw_order:
                sub = df_kw2[df_kw2["size_profile"] == cat]
                fig_kw2.add_trace(go.Box(
                    y=sub[CA_PCT].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>CA%=%{y:.1f}%<extra></extra>",
                ))
            fig_kw2.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_kw2,
                title="Current Account % GDP par profil de taille",
                yaxis_title="CA% GDP",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_kw2, use_container_width=True)
        finding(
            "<strong>Les profils de taille discriminent fortement les fondamentaux macro</strong> : "
            "GDP_pc médian Leaders (~$45k) est 4–5× celui des Small (~$8k). "
            "CA% : les Leaders ont une médiane légèrement négative (marché domestique large → "
            "déficit structurel), les Mid sont les plus excédentaires (économies émergentes "
            "exportatrices). Le boxplot révèle la variance intra-groupe : "
            "même les Small ont des outliers à $50k+ (Luxembourg, Singapour)."
        )
        thin_divider()

        # G.1.6 Export Share × Real Interest Rate
        _question("Un taux d'intérêt réel élevé pénalise-t-il la part mondiale des exports ?")
        col_g16a, col_g16b = st.columns(2)
        with col_g16a:
            df_g16 = df_m[["country_name", RATE, "export_share_structural",
                            "structural_profile"]].dropna()
            df_g16 = df_g16[(df_g16["export_share_structural"] < 0.15)
                             & (df_g16[RATE].abs() < 30)]
            fig_g16 = _scatter(df_g16, RATE, "export_share_structural",
                               "structural_profile", _STRUC_C,
                               "Export Share vs Real Interest Rate % (ρ[E]=-0.36)",
                               "Real Interest Rate (%)", "Export Share (%)", height=390)
            st.plotly_chart(fig_g16, use_container_width=True)
        with col_g16b:
            df_g16b = df_m[["country_name", RATE, "export_dep",
                             "structural_profile"]].dropna()
            df_g16b = df_g16b[(df_g16b["export_dep"] > 0)
                               & (df_g16b["export_dep"] < 1.5)
                               & (df_g16b[RATE].abs() < 30)]
            fig_g16b = _scatter(df_g16b, RATE, "export_dep",
                                "structural_profile", _STRUC_C,
                                "Export/GDP vs Real Interest Rate % (ρ[E]=-0.18)",
                                "Real Interest Rate (%)", "Export / GDP", height=390)
            st.plotly_chart(fig_g16b, use_container_width=True)
        finding(
            "<strong>Taux réel élevé → exports comprimés</strong> : ρ=-0.36 [E] pour "
            "Export Share, ρ=-0.18 pour Export/GDP — signal économique causal. "
            "Mécanisme : des taux réels élevés renchérissent le crédit export et la "
            "production domestique, réduisant la compétitivité externe. "
            "Signal modéré car GDP size domine (ρ[S]=0.94) — le Real Rate "
            "affine l'analyse une fois la taille contrôlée."
        )
        thin_divider()

        # Notes — parties testées sans signal significatif
        with st.expander("Autres variables testées (G.1.3 External Debt · G.1.4 FDI)"):
            _note(
                "<strong>G.1.3 · External Debt stocks (% GNI) — Résultat nul</strong> : "
                "aucune relation significative entre dette externe et performance export. "
                "Question testée : «&nbsp;Les pays plus endettés externalement exportent-ils "
                "différemment ?&nbsp;» — Réponse : non. "
                "La dette externe est elle-même dominée par la taille [S] et ne discrimine "
                "pas l'efficience export une fois la taille contrôlée."
            )
            _note(
                "<strong>G.1.4 · FDI Inflows — Tautologique [S]</strong> : "
                "ρ≈+0.76 avec Export Share mais c'est un signal [S] (tautologique). "
                "Les grands pays attirent plus d'IDE <em>et</em> exportent plus — "
                "même cause (taille), pas relation causale IDE → exports. "
                "Question testée : «&nbsp;Les pays récepteurs d'IDE exportent-ils plus ?&nbsp;» "
                "— Réponse : oui, mais par la taille, pas par l'IDE lui-même."
            )

        # G.1.3 Export/GDP (Trade Openness) vs CA% + Expense%
        _question("Quels pays ont une économie orientée export indépendamment de leur taille ?")
        col_g13a, col_g13b = st.columns(2)
        with col_g13a:
            df_g13 = df_m[["country_name", CA_PCT, "export_dep",
                            "structural_profile"]].dropna()
            df_g13 = df_g13[(df_g13["export_dep"] > 0) & (df_g13["export_dep"] < 1.5)]
            fig_g13 = _scatter(df_g13, CA_PCT, "export_dep", "structural_profile", _STRUC_C,
                               "Export/GDP (ouverture) vs CA% GDP (ρ[E]=+0.41)",
                               "Current Account % GDP", "Export / GDP", height=450)
            _add_refs(fig_g13, CA_PCT, "export_dep",
                      ["USA", "China", "Brazil"], df=df_g13)
            st.plotly_chart(fig_g13, use_container_width=True)
        with col_g13b:
            df_g13b = df_m[["country_name", EXPENSE, "export_dep",
                             "structural_profile"]].dropna()
            df_g13b = df_g13b[(df_g13b["export_dep"] > 0) & (df_g13b["export_dep"] < 1.5)]
            fig_g13b = _scatter(df_g13b, EXPENSE, "export_dep",
                                "structural_profile", _STRUC_C,
                                "Export/GDP vs Expense % GDP (ρ[E]=+0.28)",
                                "Expense % GDP", "Export / GDP", height=390)
            st.plotly_chart(fig_g13b, use_container_width=True)
        finding(
            "<strong>Trois signaux [E] pour l'ouverture export</strong> : "
            "CA% ρ=+0.41 (causal : surplus → export structurel), "
            "GDP_pc ρ=+0.30 (richesse → capacité productive), "
            "Expense% ρ=+0.28 (États développés — welfare state + industrie). "
            "Les grands pays (USA, Chine, Brésil) ont une Export/GDP modérée — "
            "l'économie domestique prime sur le commerce externe."
        )
        thin_divider()

        # G.1.4 Export/capita vs GDP_pc + Expense%
        _question("Les pays riches et à forte dépense publique exportent-ils "
                  "davantage par habitant ?")
        col_g14a, col_g14b = st.columns(2)
        with col_g14a:
            df_g14 = df_m[["country_name", GDP_pc, "export_pc",
                            "combined_profile"]].dropna()
            df_g14 = df_g14[(df_g14[GDP_pc] > 0) & (df_g14["export_pc"] > 0)]
            fig_g14 = _scatter(df_g14, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                               "Export per Capita vs GDP/capita (ρ[E]=+0.82)",
                               "GDP per capita (US$, log)", "Export per Capita (US$, log)",
                               xlog=True, ylog=True, height=450, x_fmt="usd", y_fmt="usd")
            _add_refs(fig_g14, GDP_pc, "export_pc",
                      ["Germany", "France", "Sweden", "Norway"], df=df_g14,
                      x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_g14, use_container_width=True)
        with col_g14b:
            df_g14b = df_m[["country_name", EXPENSE, "export_pc",
                             "combined_profile"]].dropna()
            df_g14b = df_g14b[df_g14b["export_pc"] > 0]
            fig_g14b = _scatter(df_g14b, EXPENSE, "export_pc", "combined_profile", _COMB_C,
                                "Export per Capita vs Expense % GDP (ρ[E]=+0.50)",
                                "Expense % GDP", "Export per Capita (US$, log)",
                                ylog=True, height=390, y_fmt="usd")
            st.plotly_chart(fig_g14b, use_container_width=True)
        finding(
            "<strong>Deux signaux [E] dominants pour export/capita</strong> : "
            "GDP_pc ρ=+0.82 (slope log-log = 1.22 : +1% revenu → +1.22% exports par tête) ; "
            "Expense% GDP ρ=+0.50 : signal État développé — "
            "Allemagne, France, Nordiques exportent massivement par habitant. "
            "CA% ρ=+0.55, Real Rate ρ=-0.38 complètent la hiérarchie."
        )
        thin_divider()

        # ── G.2-G.5 Export Dynamics ──────────────────
        section_label("G.2–G.5 · Export Dynamics — Idiosyncratiques")
        st.markdown(
            "<p style='font-size:13px;color:#7090B0;'>Croissance, consistance, stabilité "
            "et résilience des exports ne répondent pas aux indicateurs macro. "
            "Ce résultat nul est <strong>analytiquement central</strong> : "
            "le niveau est prévisible, la trajectoire ne l'est pas.</p>",
            unsafe_allow_html=True,
        )

        _question("Le taux de croissance du PIB (CAGR GDP) "
                  "explique-t-il la croissance des exports ?")
        df_g2 = df_m[["country_name", "cagr_export", "cagr_gdp",
                       "size_profile"]].copy()
        df_g2 = df_g2.replace([np.inf, -np.inf], np.nan).dropna()
        df_g2 = df_g2[(df_g2["cagr_gdp"].abs() < 0.25)
                      & (df_g2["cagr_export"].abs() < 0.4)]
        rho_cx = (df_g2[["cagr_gdp", "cagr_export"]]
                  .corr(method="spearman").iloc[0, 1])
        fig_g2 = go.Figure()
        for cat, color in _SIZE_C.items():
            sub = df_g2[df_g2["size_profile"] == cat]
            if sub.empty:
                continue
            fig_g2.add_trace(go.Scatter(
                x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                mode="markers", name=cat,
                marker=dict(color=color, size=7, opacity=0.75),
                text=sub["country_name"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                ),
            ))
        _pl(fig_g2,
            title=f"CAGR Export vs CAGR GDP (ρ={rho_cx:+.2f}) — aucun indicateur [E] significatif",
            xaxis_title="CAGR GDP (%)", yaxis_title="CAGR Export (%)",
            height=420)
        st.plotly_chart(fig_g2, use_container_width=True)
        finding(
            "<strong>Résultat nul analytiquement significatif</strong> : "
            f"CAGR GDP ρ≈{rho_cx:+.2f}. "
            "Tous les indicateurs [E] sont non-significatifs (p>0.05). "
            "KW non-significatif pour tous les profils de taille (H=3.0, p=0.39). "
            "<strong>La croissance export est idiosyncratique</strong> — "
            "ce qu'un pays fait de sa position commerciale dépend de facteurs "
            "non capturés par les 16 indicateurs macro."
        )
        _note(
            "<strong>G.3 · Consistance export (negfreq) — Résultat nul</strong> : "
            "Question testée : «&nbsp;Les pays à meilleurs fondamentaux macro ont-ils "
            "moins d'années négatives ?&nbsp;» — Réponse : non. "
            "RF R²=0.854 (size-driven) vs OLS R²=0.105 — gap massif confirmant que "
            "la relation est non-linéaire et pilotée par la taille, pas par le macro [E]. "
            "Aucun indicateur [E] significatif. La fréquence de baisses est idiosyncratique."
        )
        thin_divider()

        _question("Les pays riches exportent-ils avec plus de stabilité ? "
                  "La résilience export est-elle macro-expliquée ?")
        col_g4a, col_g4b = st.columns(2)
        with col_g4a:
            df_g4 = df_m[["country_name", "volatility_export",
                           "size_profile"]].dropna()
            df_g4 = df_g4[(df_g4["volatility_export"] > 0)
                          & (df_g4["volatility_export"] < 80)]
            fig_g4 = go.Figure()
            for cat in ["Leader", "Major", "Mid", "Small"]:
                sub = df_g4[df_g4["size_profile"] == cat]
                fig_g4.add_trace(go.Box(
                    y=sub["volatility_export"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Volatilité=%{y:.1f}%<extra></extra>",
                ))
            _pl(fig_g4,
                title="Volatilité Export par profil de taille<br>"
                      "<span style='font-size:11px;color:#6890B0;'>"
                      "Leaders = stable, Small = forte variance</span>",
                yaxis_title="Volatilité Export (%)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_g4, use_container_width=True)
        with col_g4b:
            df_g4b = df_m[["country_name", RATE, "volatility_export",
                            "size_profile"]].dropna()
            df_g4b = df_g4b[(df_g4b["volatility_export"] > 0)
                             & (df_g4b["volatility_export"] < 80)
                             & (df_g4b[RATE].abs() < 30)]
            _rho_g4b = (df_g4b[[RATE, "volatility_export"]]
                        .corr(method="spearman").iloc[0, 1])
            fig_g4b = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_g4b[df_g4b["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_g4b.add_trace(go.Scatter(
                    x=sub[RATE], y=sub["volatility_export"],
                    mode="markers", name=cat,
                    marker=dict(color=color, size=6, opacity=0.7),
                    text=sub["country_name"],
                    hovertemplate="<b>%{text}</b><br>Real Rate=%{x:.1f}%<br>Volatilité=%{y:.1f}%<extra></extra>",
                ))
            _xs_g4b = df_g4b[RATE].values
            _ys_g4b = df_g4b["volatility_export"].values
            _xl_g4b = np.linspace(_xs_g4b.min(), _xs_g4b.max(), 100)
            fig_g4b.add_trace(go.Scatter(
                x=_xl_g4b, y=np.polyval(np.polyfit(_xs_g4b, _ys_g4b, 1), _xl_g4b),
                mode="lines", name="OLS",
                line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
            ))
            fig_g4b.add_annotation(
                x=0.5, y=0.98, xref="paper", yref="paper",
                text=f"ρ = {_rho_g4b:+.3f} {'✓' if abs(_rho_g4b) > 0.15 else '✗ n.s.'}",
                showarrow=False, xanchor="center", yanchor="top",
                bgcolor="#162035", bordercolor="#304870",
                font=dict(size=12, color="#F59E0B"),
            )
            _pl(fig_g4b,
                title="Volatilité Export vs Real Interest Rate",
                xaxis_title="Real Interest Rate (%)",
                yaxis_title="Volatilité Export (%)",
                height=390,
                legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                            bgcolor="rgba(0,0,0,0)", borderwidth=0))
            st.plotly_chart(fig_g4b, use_container_width=True)
        finding(
            "<strong>Signal faible mais cohérent</strong> : Real Rate ρ=+0.20 — "
            "taux réels élevés → exports plus volatils (coût financement export). "
            "GDP_pc ρ=-0.15 (p<0.05) — les pays riches exportent plus régulièrement. "
            "Le boxplot confirme : Leaders regroupés sous 20%, Small dispersés 0–80%. "
            "L'effet taille protège la régularité export, mais modestement."
        )
        _note(
            "<strong>G.5 · Résilience export — Résultat nul [E]</strong> : "
            "Question testée : «&nbsp;Les fondamentaux macro protègent-ils la "
            "résilience des exports face aux chocs ?&nbsp;» — Réponse : non. "
            "Seul signal : GDP (US$) ρ=+0.20 [S] (size) — les grands pays "
            "résistent légèrement mieux, mais par leur taille absolue. "
            "OLS R²=0.029 — quasi nul. La résilience export est idiosyncratique "
            "(politiques commerciales, diversification produits/marchés)."
        )
        thin_divider()

        # G.7 Synthesis
        section_label("G.7 · Deux mondes distincts")
        finding(
            "<strong>Monde 1 — Niveau expliqué</strong> : "
            "Export Share ρ[S]=0.94, Export/Capita ρ[E]=0.82, "
            "Export/GDP ρ[E]=0.41. "
            "GDP size, CA%, GDP_pc et Expense% déterminent <em>combien</em> un pays exporte."
        )
        finding(
            "<strong>Monde 2 — Dynamiques non expliquées</strong> : "
            "CAGR Export (tous [E] n.s.), Neg.Freq (RF R²=0.85 mais OLS=0.11 → "
            "la relation est non-linéaire et size-driven), Résilience (ρ_max=0.20). "
            "Ce qu'un pays <em>fait</em> de sa position de départ est idiosyncratique."
        )

    # ════════════════════════════════════════════════════
    # H · IMPORTS
    # ════════════════════════════════════════════════════
    with subtab_h:

        # ── H.1-H.3 Import Level ─────────────────────
        section_label("H.1–H.3 · Import Level — Miroir de G, relation encore plus forte")
        st.markdown(
            "<p style='font-size:13px;color:#7090B0;'>Les imports répondent encore "
            "mieux aux fondamentaux macro que les exports : Import/capita présente "
            "4 signaux [E] forts et simultanés (vs 2–3 pour Export/capita).</p>",
            unsafe_allow_html=True,
        )

        # H.1 Import Share vs GDP
        _question("La taille économique explique-t-elle encore mieux "
                  "les imports que les exports ?")
        col_h1a, col_h1b = st.columns(2)
        with col_h1a:
            df_h1 = df_m[["country_name", GDP, "import_share_structural",
                           "size_profile"]].dropna()
            df_h1 = df_h1[df_h1[GDP] > 0]
            fig_h1 = _scatter(df_h1, GDP, "import_share_structural", "size_profile", _SIZE_C,
                              "Import Share vs GDP (ρ=+0.96 > Export ρ=0.94)",
                              "GDP (US$, log)", "Import Share (%)",
                              xlog=True, height=450, x_fmt="usd")
            _add_refs(fig_h1, GDP, "import_share_structural",
                      ["USA"], df=df_h1, x_fmt="usd")
            st.plotly_chart(fig_h1, use_container_width=True)
        with col_h1b:
            df_h1b = df_m[["country_name", "export_share_structural",
                            "import_share_structural", "size_profile"]].dropna()
            df_h1b = df_h1b[(df_h1b["export_share_structural"] > 0)
                             & (df_h1b["import_share_structural"] > 0)]
            fig_h1b = _scatter(df_h1b, "export_share_structural", "import_share_structural",
                               "size_profile", _SIZE_C,
                               "Import Share vs Export Share — symétrie",
                               "Export Share (%)", "Import Share (%)", height=450)
            _mv = max(df_h1b["export_share_structural"].max(),
                      df_h1b["import_share_structural"].max())
            fig_h1b.add_trace(go.Scatter(
                x=[0, _mv], y=[0, _mv], mode="lines", name="Équilibre (1:1)",
                line=dict(color="#6890B0", dash="dot", width=1.5), hoverinfo="skip",
            ))
            _add_refs(fig_h1b, "export_share_structural", "import_share_structural",
                      ["USA", "China", "Germany", "Japan"], df=df_h1b)
            st.plotly_chart(fig_h1b, use_container_width=True)
        finding(
            "<strong>Relation quasi-mécanique</strong> : ρ=+0.96 vs Export ρ=+0.94. "
            "USA = plus grand importateur (10.7%) malgré une export efficiency < 1 — "
            "le marché domestique aspire le monde entier. "
            "Signal [E] : GDP_pc ρ=+0.45, CA% ρ=+0.42 — les pays riches "
            "et excédentaires importent aussi proportionnellement plus."
        )
        thin_divider()

        # H.3 Import/capita — 4 [E] signals
        _question("Quels indicateurs [E] expliquent le mieux les imports par habitant ? "
                  "(Meilleure variable expliquée de H)")
        col_h3a, col_h3b = st.columns(2)
        with col_h3a:
            df_h3 = df_m[["country_name", GDP_pc, "import_pc",
                           "combined_profile"]].dropna()
            df_h3 = df_h3[(df_h3[GDP_pc] > 0) & (df_h3["import_pc"] > 0)]
            fig_h3 = _scatter(df_h3, GDP_pc, "import_pc", "combined_profile", _COMB_C,
                              "Import per Capita vs GDP/capita (ρ=+0.925)",
                              "GDP per capita (US$, log)", "Import per Capita (US$, log)",
                              xlog=True, ylog=True, height=390, x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_h3, use_container_width=True)
        with col_h3b:
            tc_h3 = _top_correlates("import_pc", n=6, flag_filter="[E]")
            st.plotly_chart(_corr_bar(tc_h3, "Import per Capita × [E] (4 signaux forts)",
                                      height=390),
                            use_container_width=True)
        finding(
            "<strong>Import/capita : meilleure variable expliquée de tout G+H</strong>. "
            "4 signaux [E] simultanément significatifs et forts : "
            "GDP_pc ρ=+0.925 (richesse = pouvoir d'achat = imports), "
            "Expense% ρ=+0.57 (dépense publique → consommation collective), "
            "Inflation ρ=-0.47 (l'inflation comprime les imports par érosion du pouvoir d'achat — "
            "signal plus fort ici que pour export/capita ρ=-0.32), "
            "Real Rate ρ=-0.36 (crédit cher → moins d'imports)."
        )
        thin_divider()

        # H.2 Import/GDP openness
        _question("L'ouverture import est-elle liée à la richesse "
                  "et à la dépense publique ?")
        col_h2a, col_h2b = st.columns(2)
        with col_h2a:
            df_h2 = df_m[["country_name", GDP_pc, "import_dep",
                           "structural_profile"]].dropna()
            df_h2 = df_h2[(df_h2["import_dep"] > 0) & (df_h2["import_dep"] < 1.5)]
            fig_h2 = _scatter(df_h2, GDP_pc, "import_dep", "structural_profile", _STRUC_C,
                              "Import/GDP vs GDP/capita (ρ[E]=+0.22)",
                              "GDP per capita (US$, log)", "Import / GDP",
                              xlog=True, height=450, x_fmt="usd")
            _add_refs(fig_h2, GDP_pc, "import_dep",
                      ["USA", "Japan"], df=df_h2, x_fmt="usd")
            st.plotly_chart(fig_h2, use_container_width=True)
        with col_h2b:
            df_h2b = df_m[["country_name", EXPENSE, "import_dep",
                            "structural_profile"]].dropna()
            df_h2b = df_h2b[(df_h2b["import_dep"] > 0) & (df_h2b["import_dep"] < 1.5)]
            fig_h2b = _scatter(df_h2b, EXPENSE, "import_dep",
                               "structural_profile", _STRUC_C,
                               "Import/GDP vs Expense % GDP (ρ[E]=+0.36)",
                               "Expense % GDP", "Import / GDP", height=390)
            st.plotly_chart(fig_h2b, use_container_width=True)
        finding(
            "<strong>Expense% GDP : signal plus fort pour l'ouverture import qu'export</strong> "
            "(ρ=+0.36 vs +0.28). "
            "L'État développé génère une demande d'importation via la consommation publique. "
            "USA et Japon : sous-tendance — Import/GDP faible malgré richesse et dépense "
            "(production domestique substituant les imports). "
            "Inflation ρ=-0.16 (borderline, p=0.051) — tendance mais non confirmée."
        )
        thin_divider()

        # ── H.4-H.7 Import Dynamics ──────────────────
        section_label("H.4–H.7 · Import Dynamics — Principalement idiosyncratiques")

        _question("Le PIB prédit-il mieux la croissance des imports que des exports ? "
                  "(Comparaison directe CAGR)")
        col_h4a, col_h4b = st.columns(2)
        with col_h4a:
            df_h4 = df_m[["country_name", "cagr_import", "cagr_gdp",
                           "size_profile"]].copy()
            df_h4 = df_h4.replace([np.inf, -np.inf], np.nan).dropna()
            df_h4 = df_h4[(df_h4["cagr_gdp"].abs() < 0.25)
                          & (df_h4["cagr_import"].abs() < 0.4)]
            rho_cm = (df_h4[["cagr_gdp", "cagr_import"]]
                      .corr(method="spearman").iloc[0, 1])
            fig_h4 = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_h4[df_h4["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_h4.add_trace(go.Scatter(
                    x=sub["cagr_gdp"] * 100, y=sub["cagr_import"] * 100,
                    mode="markers", name=cat,
                    marker=dict(color=color, size=7, opacity=0.75),
                    text=sub["country_name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "CAGR GDP=%{x:.1f}% | CAGR Import=%{y:.1f}%<extra></extra>"
                    ),
                ))
            _pl(fig_h4,
                title=f"CAGR Import vs CAGR GDP (ρ={rho_cm:+.2f})",
                xaxis_title="CAGR GDP (%)", yaxis_title="CAGR Import (%)",
                height=450)
            _add_refs(fig_h4, "cagr_gdp", "cagr_import", ["China"],
                      df=df_h4, x_scale=100, y_scale=100,
                      ht="<b>%{text}</b><br>CAGR GDP=%{x:.1f}% | CAGR Import=%{y:.1f}%<extra></extra>")
            st.plotly_chart(fig_h4, use_container_width=True)
        with col_h4b:
            df_h4x = df_m[["country_name", "cagr_export", "cagr_gdp",
                            "size_profile"]].copy()
            df_h4x = df_h4x.replace([np.inf, -np.inf], np.nan).dropna()
            df_h4x = df_h4x[(df_h4x["cagr_gdp"].abs() < 0.25)
                            & (df_h4x["cagr_export"].abs() < 0.4)]
            rho_cx2 = (df_h4x[["cagr_gdp", "cagr_export"]]
                       .corr(method="spearman").iloc[0, 1])
            fig_h4x = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_h4x[df_h4x["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_h4x.add_trace(go.Scatter(
                    x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                    mode="markers", name=cat,
                    marker=dict(color=color, size=7, opacity=0.75),
                    text=sub["country_name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                    ),
                ))
            _pl(fig_h4x,
                title=f"CAGR Export vs CAGR GDP (ρ={rho_cx2:+.2f}) — référence G.2",
                xaxis_title="CAGR GDP (%)", yaxis_title="CAGR Export (%)",
                height=390)
            st.plotly_chart(fig_h4x, use_container_width=True)
        finding(
            f"<strong>Les imports suivent le cycle GDP mieux que les exports</strong> : "
            f"CAGR Import ρ≈{rho_cm:+.2f} vs CAGR Export ρ≈{rho_cx2:+.2f}. "
            "Interprétation : quand l'économie croît, la demande domestique "
            "tire les imports mécaniquement. L'offre d'export dépend de la "
            "compétitivité externe — plus idiosyncratique. "
            "<strong>Chine</strong> : CAGR Import négatif malgré PIB positif — "
            "substitution domestique massive."
        )
        thin_divider()

        # H.5 Import Consistency — paradox
        _question("Les grands pays ont-ils des baisses d'imports plus fréquentes "
                  "que les petits ? (Résultat contre-intuitif)")
        col_h5a, col_h5b = st.columns(2)
        with col_h5a:
            df_h5 = df_m[["country_name", "negfreq_import", "size_profile"]].dropna()
            fig_h5 = go.Figure()
            for cat in ["Leader", "Major", "Mid", "Small"]:
                sub = df_h5[df_h5["size_profile"] == cat]
                fig_h5.add_trace(go.Box(
                    y=sub["negfreq_import"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Neg.Freq Import=%{y:.0%}<extra></extra>",
                ))
            _pl(fig_h5,
                title="Fréquence baisses import par profil de taille<br>"
                      "<span style='font-size:11px;color:#F59E0B;'>"
                      "Paradoxe : Leaders ont plus de baisses que Small</span>",
                yaxis_title="Fréquence années négatives (ratio)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_h5, use_container_width=True)
        with col_h5b:
            df_h5b = df_m[["country_name", "negfreq_import",
                            "structural_profile"]].dropna()
            fig_h5b = go.Figure()
            for cat in ["Strong", "Good", "Moderate", "Weak"]:
                sub = df_h5b[df_h5b["structural_profile"] == cat]
                if sub.empty:
                    continue
                fig_h5b.add_trace(go.Box(
                    y=sub["negfreq_import"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_STRUC_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Neg.Freq Import=%{y:.0%}<extra></extra>",
                ))
            _pl(fig_h5b,
                title="Fréquence baisses import par profil structurel",
                yaxis_title="Fréquence années négatives (ratio)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_h5b, use_container_width=True)
        finding(
            "<strong>Paradoxe H.5</strong> : GDP (US$) ρ=+0.21 — "
            "les <em>grands</em> pays ont plus fréquemment des années de baisse import, "
            "contrairement à ce qu'on attendrait. "
            "Explication : les grandes économies (USA, Chine, UE) absorbent les chocs "
            "mondiaux (crises financières, COVID) de manière plus cyclique — "
            "leurs grandes dépendances import deviennent fragiles lors des récessions. "
            "Les petits pays ont des imports plus stables car plus contraints "
            "(pas de boom/bust cyclique de la demande intérieure). "
            "Aucun signal [E] significatif — le pattern est size-driven [S]."
        )
        thin_divider()

        _question("L'inflation rend-elle les imports plus volatils ? "
                  "(Signal absent pour les exports)")
        col_h6a, col_h6b = st.columns(2)
        with col_h6a:
            df_h6 = df_m[["country_name", INFL, "volatility_import",
                           "size_profile"]].dropna()
            df_h6 = df_h6[(df_h6["volatility_import"] > 0)
                          & (df_h6["volatility_import"] < 80)
                          & (df_h6[INFL].abs() < 30)]
            fig_h6 = _scatter(df_h6, INFL, "volatility_import", "size_profile", _SIZE_C,
                              "Volatilité Import vs Inflation % (ρ[E]=+0.20)",
                              "Inflation % (CPI)", "Volatilité Import (%)", height=390)
            st.plotly_chart(fig_h6, use_container_width=True)
        with col_h6b:
            df_h6b = df_m[["country_name", GDP_pc, "volatility_import",
                            "size_profile"]].dropna()
            df_h6b = df_h6b[(df_h6b["volatility_import"] > 0)
                            & (df_h6b["volatility_import"] < 80)]
            fig_h6b = _scatter(df_h6b, GDP_pc, "volatility_import",
                               "size_profile", _SIZE_C,
                               "Volatilité Import vs GDP/capita (ρ[E]=-0.24)",
                               "GDP per capita (US$, log)", "Volatilité Import (%)",
                               xlog=True, height=390, x_fmt="usd")
            st.plotly_chart(fig_h6b, use_container_width=True)
        finding(
            "<strong>Signal Inflation — spécifique aux imports (H.6)</strong> : "
            "Inflation ρ=+0.20 (p<0.01) pour volatilité import, "
            "absent pour volatilité export (ρ=+0.08 n.s.). "
            "L'inflation crée une incertitude dans les décisions d'achat → "
            "comportement d'importation erratique. "
            "GDP_pc ρ=-0.24 : les pays riches importent plus régulièrement "
            "(buffer financier contre les chocs d'approvisionnement)."
        )
        _note(
            "<strong>H.7 · Résilience import — Résultat nul absolu</strong> : "
            "Question testée : «&nbsp;Les fondamentaux macro protègent-ils la "
            "résilience des imports face aux chocs ?&nbsp;» — Réponse : non, "
            "et paradoxalement GDP (US$) ρ=-0.10 (n.s., légèrement négatif) — "
            "les grands pays sont <em>marginalement moins résilients</em> "
            "(miroir du paradoxe H.5 : grandes dépendances import = plus exposées). "
            "R² OLS=0.013 — quasi nul. C'est le résultat le plus nul de tout G+H. "
            "La résilience import est entièrement idiosyncratique."
        )
        thin_divider()

        # H.8 Synthesis
        section_label("H.8 · Imports vs Exports — Ce qui diffère structurellement")
        finding(
            "<strong>Level : imports encore mieux expliqués que exports</strong> : "
            "Import Share ρ=0.96 > Export Share ρ=0.94. "
            "Import/capita : 4 signaux [E] forts (GDP_pc 0.925, Expense% 0.57, "
            "Inflation -0.47, Real Rate -0.36) vs Export/capita : 2–3 signaux. "
            "L'importation est plus mécaniquement liée aux fondamentaux macro."
        )
        finding(
            "<strong>Dynamics : imports légèrement plus prévisibles</strong> : "
            f"CAGR Import ρ≈{rho_cm:+.2f} vs CAGR Export ρ≈{rho_cx2:+.2f} (CAGR GDP). "
            "Inflation ρ=+0.20 pour volatilité import — signal absent pour exports. "
            "Les imports suivent la demande domestique, les exports suivent "
            "la compétitivité — plus idiosyncratique par nature."
        )
        finding(
            "<strong>Anomalies pays</strong> : "
            "USA = sur-importateur (Import Share 10.7%, Export Efficiency ~0.65×) ; "
            "Germany = sur-exportateur (Efficiency ~2.6×) ; "
            "China = paradoxe import (CAGR négatif malgré GDP positif = substitution). "
            "Ces patterns confirment que macro décrit la structure, pas la trajectoire."
        )

    # ════════════════════════════════════════════════════
    # G+H SYNTHESIS
    # ════════════════════════════════════════════════════
    with subtab_synth:
        section_label("G+H · Synthèse — Niveau vs Dynamiques")

        # Summary table
        st.markdown("""
        <div style="font-size:13px;color:#A8C0D8;line-height:1.9;margin-bottom:1.2rem;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Variable</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Top signal</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">ρ (top)</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Flag</th>
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Conclusion</th>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export Share</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.94</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Taille déterministe</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import Share</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.96</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Encore plus mécanique</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/capita</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP_pc</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.82</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Richesse + Expense% (0.50)</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/capita</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP_pc</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.925</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#60A5FA;font-weight:600;">
              4 signaux [E] forts</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;">CA% GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0.41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Modéré — surplus → ouverture</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;">Expense%</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0.36</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Plus fort ici qu'Export/GDP</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Export</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Idiosyncratique</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Import</td>
            <td style="padding:.35rem .7rem;text-align:center;">CAGR GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">~+0.41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[T]</span></td>
            <td style="padding:.35rem .7rem;">Cycle GDP → plus prévisible</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Volatilité Export</td>
            <td style="padding:.35rem .7rem;text-align:center;">Real Rate</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0.20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Faible</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Volatilité Import</td>
            <td style="padding:.35rem .7rem;text-align:center;">Inflation</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0.20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#A78BFA;">Signal unique à H</td>
          </tr>
          <tr>
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Résilience X/M</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Nulle — idiosyncratique</td>
          </tr>
        </table>
        <div style="margin-top:.7rem;font-size:11px;color:#4A6890;">
          [S] = Size-driven (taille) · [E] = Economic (richesse/structure) ·
          [T] = Tautologique · n.s. = non-significatif
        </div>
        </div>
        """, unsafe_allow_html=True)

        thin_divider()
        section_label("Message Central")
        finding(
            "<strong>Niveau = prévisible par le macro</strong> : "
            "la taille [S] explique qui domine le commerce mondial (ρ>0.94). "
            "La richesse et la structure [E] (GDP_pc, CA%, Expense%, Real Rate) "
            "expliquent l'efficience et l'ouverture (ρ=0.28–0.925). "
            "Ces relations sont <em>descriptives et structurelles</em>."
        )
        finding(
            "<strong>Dynamiques = idiosyncratiques</strong> : "
            "croissance, consistance et résilience des exports ne répondent "
            "pas aux 16 indicateurs macro. "
            "Exception partielle : les imports suivent légèrement le cycle GDP "
            "(CAGR Import ρ≈0.41) car la demande domestique tire mécaniquement les imports. "
            "Ce qu'un pays <em>fait</em> de sa position de départ reste non prévisible."
        )

# ────────────────────────────────────────────────────────────────
# TAB I-J — BALANCE & PROFILS
# ────────────────────────────────────────────────────────────────

with tab_ij:
    _IJ_VARS = {
        "coverage_ratio_structural": "Coverage Ratio (X/M)",
        "balance_dep":               "Balance / GDP (X−M)",
        "trade_intensity":           "Trade Intensity (X+M)/GDP",
    }
    _fig_ij_heat = _tab_heatmap(
        _IJ_VARS,
        "Corrélations [E] — Variables Balance & Ouverture (Tab I) · * = p<0.05",
        flags=["[E]"],
    )
    if _fig_ij_heat:
        st.plotly_chart(_fig_ij_heat, use_container_width=True)
    thin_divider()

    subtab_i, subtab_j = st.tabs(["I · Balance", "J · Quadrants & Profils"])

    # ════════════════════════════════════════════════════
    # I · BALANCE
    # ════════════════════════════════════════════════════
    with subtab_i:
        section_label("I · Balance Commerciale × Macro — CA% comme signal central")
        st.markdown(
            "<p style='font-size:13px;color:#7090B0;'>La section I relie les variables "
            "de balance (coverage ratio, solde/GDP, profil de balance) aux indicateurs "
            "macro. Un signal [E] domine de manière cohérente sur trois angles différents.</p>",
            unsafe_allow_html=True,
        )

        # I.1 Coverage Ratio × CA%
        _question("Les pays excédentaires (CA% > 0) couvrent-ils mieux "
                  "leurs imports par leurs exports ?")
        col_i1a, col_i1b = st.columns(2)
        with col_i1a:
            df_i1 = df_m[["country_name", CA_PCT, "coverage_ratio_structural",
                           "structural_profile"]].dropna()
            df_i1 = df_i1[(df_i1["coverage_ratio_structural"] < 5)
                           & (df_i1[CA_PCT].abs() < 25)]
            fig_i1 = _scatter(df_i1, CA_PCT, "coverage_ratio_structural",
                              "structural_profile", _STRUC_C,
                              "Coverage Ratio (X/M) vs CA% GDP (ρ[E]=+0.53)",
                              "Current Account % GDP", "Coverage Ratio (X/M)",
                              height=390)
            fig_i1.add_hline(y=1, line_color="#6890B0", line_dash="dot", line_width=1,
                             annotation_text="X = M", annotation_font_color="#6890B0")
            fig_i1.add_vline(x=0, line_color="#6890B0", line_dash="dot", line_width=1)
            st.plotly_chart(fig_i1, use_container_width=True)
        with col_i1b:
            tc_i1 = _top_correlates("coverage_ratio_structural", n=6, flag_filter="[E]")
            st.plotly_chart(_corr_bar(tc_i1, "Coverage Ratio × [E]", height=390),
                            use_container_width=True)
        finding(
            "<strong>CA% [E] ρ=+0.53 — signal causal le plus propre de I</strong> : "
            "un excédent courant implique que les exports couvrent structurellement "
            "les imports (CA% inclut revenus + transferts, donc la relation n'est pas "
            "circulaire). Les Strong/Good profiles se concentrent au-dessus de la ligne X=M "
            "et à droite de CA%=0."
        )
        thin_divider()

        # I.3 Balance/GDP × CA%
        _question("Le solde commercial (X−M)/GDP reflète-t-il "
                  "directement le compte courant ?")
        col_i3a, col_i3b = st.columns(2)
        with col_i3a:
            df_i3 = df_m[["country_name", CA_PCT, "balance_dep",
                           "structural_profile"]].dropna()
            df_i3 = df_i3[(df_i3["balance_dep"].abs() < 0.8)
                           & (df_i3[CA_PCT].abs() < 25)]
            fig_i3 = _scatter(df_i3, CA_PCT, "balance_dep",
                              "structural_profile", _STRUC_C,
                              "Balance/GDP vs CA% GDP (ρ[E]=+0.48)",
                              "Current Account % GDP", "(X−M) / GDP",
                              height=390)
            fig_i3.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            fig_i3.add_vline(x=0, line_color="#6890B0", line_dash="dot", line_width=1)
            st.plotly_chart(fig_i3, use_container_width=True)
        with col_i3b:
            df_i3b = df_m[["country_name", GDP_pc, "balance_dep",
                            "combined_profile"]].dropna()
            df_i3b = df_i3b[(df_i3b[GDP_pc] > 0)
                             & (df_i3b["balance_dep"].abs() < 0.8)]
            _rho_i3b = (df_i3b[[GDP_pc, "balance_dep"]]
                        .corr(method="spearman").iloc[0, 1])
            fig_i3b = _scatter(df_i3b, GDP_pc, "balance_dep",
                               "combined_profile", _COMB_C,
                               f"Balance/GDP vs GDP/capita (ρ={_rho_i3b:+.3f})",
                               "GDP per capita (US$, log)", "(X−M) / GDP",
                               xlog=True, height=390, x_fmt="usd")
            fig_i3b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _add_refs(fig_i3b, GDP_pc, "balance_dep",
                      ["USA", "China", "Germany", "Japan", "Brazil"], df=df_i3b,
                      x_fmt="usd")
            st.plotly_chart(fig_i3b, use_container_width=True)
        finding(
            "<strong>CA% ρ=+0.48 — deuxième confirmation indépendante</strong> : "
            "trois angles différents (coverage ratio I.1, balance/GDP I.3, profil I.5) "
            "convergent sur le même signal. La relation n'est pas circulaire : "
            "CA% inclut les revenus d'investissement et les transferts courants, "
            "pas uniquement le solde biens/services."
        )
        thin_divider()

        # I.5 Balance Profile × KW
        _question("Quels indicateurs [E] discriminent le mieux "
                  "les profils de balance commerciale ?")
        col_i5a, col_i5b = st.columns(2)
        _bal_order = ["Strong Surplus", "Slight Surplus", "Slight Deficit",
                      "Structural Deficit"]
        with col_i5a:
            df_i5 = df_m[["balance_profile", CA_PCT]].dropna()
            df_i5 = df_i5[df_i5[CA_PCT].abs() < 25]
            fig_i5a = _kw_box(CA_PCT, "balance_profile", _bal_order, _BAL_C)
            fig_i5a.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_i5a,
                title="CA% GDP par profil de balance (KW η²=0.241)",
                yaxis_title="Current Account % GDP",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_i5a, use_container_width=True)
        with col_i5b:
            fig_i5b = _kw_box(GDP_pc, "balance_profile", _bal_order, _BAL_C, fmt_fn=fmt_usd)
            _pl(fig_i5b,
                title="GDP per Capita par profil de balance (η² modéré)",
                yaxis_title="GDP per Capita (US$)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_i5b, use_container_width=True)
        finding(
            "<strong>Balance Profile × KW — discrimination forte</strong> : "
            "CA% GDP η²=0.241 (large) — le signal le plus fort de toute la section I. "
            "Médians : Strong Surplus ~+2%, Slight Surplus +1.7%, "
            "Slight Deficit -1.1%, Structural Deficit -4.8%. "
            "Progression monotone et robuste (Spearman et KW cohérents). "
            "GDP_pc discrimine modérément : Structural Deficit = pays pauvres en médiane, "
            "mais forte variance (des pays riches ont aussi un déficit structurel)."
        )

        with st.expander("Sections testées — résultats référencés en Tab D"):
            _note(
                "<strong>I.2 · Trade Intensity × Expense% (ρ=+0.34)</strong> : "
                "analysé en Tab D · D.1 Trade Intensity. "
                "Résultat : Expense% [E] ρ=+0.34 — les États développés ont une "
                "économie plus ouverte au commerce. Voir D.1 pour l'analyse complète."
            )
            _note(
                "<strong>I.4 · Trade Elasticity × Macro</strong> : "
                "analysé en Tab D · D.2. "
                "Résultat : le macro ne prédit pas l'élasticité trade/PIB. "
                "La situation (Export-led, GDP-led, etc.) est idiosyncratique. "
                "Voir D.2 pour la classification des 6 situations."
            )

        thin_divider()
        section_label("I · Synthèse — CA% comme signal central")
        finding(
            "<strong>CA% GDP est le signal [E] le plus cohérent de section I</strong> : "
            "trois analyses indépendantes convergent : "
            "I.1 Coverage Ratio ρ=+0.53, I.3 Balance/GDP ρ=+0.48, "
            "I.5 Balance Profile η²=0.241. "
            "Interprétation : le compte courant capture la position compétitive externe "
            "de manière plus complète que le solde commercial seul."
        )

    # ════════════════════════════════════════════════════
    # J · QUADRANTS & PROFILS
    # ════════════════════════════════════════════════════
    with subtab_j:
        section_label("J · Quadrants & Profils Trade × Macro")
        st.markdown(
            "<p style='font-size:13px;color:#7090B0;'>Section J teste si les différents "
            "systèmes de classification trade (structurel, statique, dynamique, stratégique) "
            "correspondent à des fondamentaux macro distincts. "
            "Règle générale : <strong>statique = expliqué</strong>, "
            "<strong>dynamique = idiosyncratique</strong>.</p>",
            unsafe_allow_html=True,
        )

        # J.1 Structural Profile × KW
        _question("Les profils structurels (Strong/Good/Moderate/Weak) "
                  "correspondent-ils à des fondamentaux macro différents ?")
        col_j1a, col_j1b = st.columns(2)
        _struc_order = ["Strong", "Good", "Moderate", "Weak"]
        with col_j1a:
            fig_j1a = _kw_box(GDP_pc, "structural_profile", _struc_order, _STRUC_C, fmt_fn=fmt_usd)
            _pl(fig_j1a,
                title="GDP/capita par profil structurel (η²[E]=0.137)",
                yaxis_title="GDP per Capita (US$)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j1a, use_container_width=True)
        with col_j1b:
            fig_j1b = _kw_box(CA_PCT, "structural_profile", _struc_order, _STRUC_C)
            fig_j1b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_j1b,
                title="CA% GDP par profil structurel (η²[E]=0.099)",
                yaxis_title="Current Account % GDP",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j1b, use_container_width=True)
        finding(
            "<strong>Profil structurel partiellement macro-expliqué</strong> : "
            "η² GDP_pc=0.137 (large), CA%=0.099 (medium). "
            "Nuance importante : le structural_score est dominé à 60% par z_export_share — "
            "un grand pays peut atteindre 'Strong' par sa taille seule, "
            "indépendamment de sa performance dynamique. "
            "<strong>Contradiction Germany</strong> : score structurel le plus élevé (+4.6) "
            "mais classé Declining en J.3 — la taille masque la dégradation dynamique."
        )
        thin_divider()

        # J.2 Static Quadrant × KW
        _question("Le quadrant statique (Dominant/Balanced/Vulnerable/Marginal) "
                  "est-il le mieux discriminé par le CA% ?")
        col_j2a, col_j2b = st.columns(2)
        _static_order = ["Dominant", "Balanced", "Vulnerable", "Marginal"]
        with col_j2a:
            df_j2 = df_m[["static_quadrant", CA_PCT]].dropna()
            df_j2 = df_j2[df_j2[CA_PCT].abs() < 25]
            fig_j2a = _kw_box(CA_PCT, "static_quadrant", _static_order, _STATIC_C)
            fig_j2a.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_j2a,
                title="CA% GDP par quadrant statique (η²[E]=0.250 — max de J)",
                yaxis_title="Current Account % GDP",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j2a, use_container_width=True)
        with col_j2b:
            fig_j2b = _kw_box(GDP_pc, "static_quadrant", _static_order, _STATIC_C, fmt_fn=fmt_usd)
            _pl(fig_j2b,
                title="GDP/capita par quadrant statique (η²[E]=0.128)",
                yaxis_title="GDP per Capita (US$)",
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j2b, use_container_width=True)
        finding(
            "<strong>Signal le plus fort de toute la section J</strong> : "
            "CA% η²=0.250 (KW H=47.0, p<0.0001). "
            "Cohérence totale avec I.1 (ρ=0.53), I.3 (ρ=0.48), I.5 (η²=0.241) : "
            "quatre analyses indépendantes confirment que CA% est "
            "le déterminant macro du positionnement statique. "
            "Dominant = pays excédentaires en médiane, "
            "Vulnerable = pays déficitaires malgré une grande taille."
        )
        thin_divider()

        # J.3 Dynamic Quadrant — null (note)
        _question("Le quadrant dynamique (Star/Risky Grower/Stable/Declining) "
                  "est-il prévisible par les fondamentaux macro ?")
        _note(
            "<strong>J.3 · Quadrant dynamique — Résultat nul [E]</strong> : "
            "aucun indicateur macro ne discrimine significativement entre "
            "Star, Risky Grower, Stable Stagnant et Declining. "
            "C'est la confirmation la plus forte des résultats G.2/G.3/G.5 : "
            "le quadrant dynamique synthétise la croissance et la stabilité export — "
            "deux dimensions toutes deux idiosyncratiques. "
            "<strong>Nuance Germany</strong> : classé Declining (z_cagr < médiane) "
            "mais en Stars en J.4 (cagr absolu > 0%) — la même performance peut être "
            "Declining en relatif et Growing en absolu selon le seuil."
        )
        thin_divider()

        # J.4 Strategic Quadrant — note
        _note(
            "<strong>J.4 · Quadrant stratégique (Stars/Cash Cow/Emerging/Marginal)</strong> : "
            "GDP_pc η²=0.137, CA% η²=0.096 — signal présent mais modéré. "
            "Membership Stars ≠ qualité structurelle : plusieurs Stars montrent "
            "un coverage ratio faible (I.1) et un déficit CA (I.3). "
            "Le cadre BCG capture taille × croissance mais ignore la dimension balance — "
            "complémentaire au quadrant statique J.2, pas substituable."
        )
        thin_divider()

        # J synthesis
        section_label("J · Synthèse — Statique expliqué, Dynamique idiosyncratique")
        finding(
            "<strong>Positionnement statique = macro-expliqué</strong> : "
            "J.1 (structural profile η²=0.137) et J.2 (static quadrant η²=0.250) "
            "confirment que CA% et GDP_pc déterminent où un pays se positionne "
            "dans le commerce mondial en termes de taille et de balance."
        )
        finding(
            "<strong>Dynamiques = idiosyncratiques</strong> : "
            "J.3 (dynamic quadrant) = zéro signal macro. "
            "Cohérent avec G.2 (CAGR Export n.s.), G.3 (neg.freq n.s.), G.5 (résilience n.s.). "
            "Même conclusion depuis trois niveaux d'analyse différents."
        )
        finding(
            "<strong>Enrichissement G/H/I → CA% convergence</strong> : "
            "I.1 ρ=+0.53, I.3 ρ=+0.48, I.5 η²=0.241, J.2 η²=0.250 — "
            "quatre analyses indépendantes dans deux sections différentes "
            "convergent sur le même indicateur. "
            "CA% GDP est le signal causal [E] le plus robuste de l'ensemble G→J."
        )

# ────────────────────────────────────────────────────────────────
# TAB K — SCORES
# ────────────────────────────────────────────────────────────────
with tab_k:
    section_label("K · Scores Composites × Macro")
    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>Section K analyse les scores composites "
        "(structural, bilateral resilience, combined) : sont-ils macro-expliqués ? "
        "Résultat clé : deux dimensions avec deux logiques analytiques distinctes.</p>",
        unsafe_allow_html=True,
    )
    _K_VARS = {
        "structural_score": "Structural Score (F)",
        "f_norm_rank":      "F Norm Rank",
        "g_norm_rank":      "G Norm Rank (Bilatéral)",
        "combined_score":   "Combined Score",
    }
    _fig_k_heat = _tab_heatmap(
        _K_VARS,
        "Corrélations [E] — Scores Composites (Tab K) · * = p<0.05",
        flags=["[E]"],
    )
    if _fig_k_heat:
        st.plotly_chart(_fig_k_heat, use_container_width=True)
    thin_divider()

    # K.1 — Bilateral Resilience (null result)
    _question("La résilience bilatérale (diversification géographique export) "
              "est-elle liée aux fondamentaux macro ?")
    _note(
        "<strong>K.1 · g_norm_rank (résilience bilatérale) — Résultat nul</strong> : "
        "aucun indicateur macro ne prédit la concentration géographique des exports. "
        "Hypothèse testée : meilleur CA% → meilleure diversification des partenaires. "
        "Résultat : tous ρ<0.18, p>0.05. "
        "La diversification export est le fruit de politiques commerciales, "
        "d'accords bilatéraux et de géographie — pas des fondamentaux macro agrégés."
    )
    thin_divider()

    # K.3 Combined Profile × KW
    _question("Le profil combiné (Champion/Competitive/Vulnerable/Fragile) "
              "discrimine-t-il les fondamentaux macro [E] ?")
    col_k3a, col_k3b = st.columns(2)
    _comb_order = ["Global Champion", "Competitive", "Vulnerable", "Fragile"]
    with col_k3a:
        fig_k3a = _kw_box(GDP_pc, "combined_profile", _comb_order, _COMB_C, fmt_fn=fmt_usd)
        _pl(fig_k3a,
            title="GDP/capita par profil combiné (KW H=19.3, p=0.0002)",
            yaxis_title="GDP per Capita (US$)",
            height=390,
            xaxis=dict(gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_k3a, use_container_width=True)
    with col_k3b:
        df_k3b = df_m[["combined_profile", CA_PCT]].dropna()
        df_k3b = df_k3b[df_k3b[CA_PCT].abs() < 25]
        fig_k3b = _kw_box(CA_PCT, "combined_profile", _comb_order, _COMB_C)
        fig_k3b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
        _pl(fig_k3b,
            title="CA% GDP par profil combiné",
            yaxis_title="Current Account % GDP",
            height=390,
            xaxis=dict(gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_k3b, use_container_width=True)
    finding(
        "<strong>Global Champion clairement distinct</strong> : "
        "GDP_pc médian Global Champion ~$34k vs Competitive/Vulnerable/Fragile à $5–8k — "
        "écart 4–7×. Le profil combiné identifie une élite économique réelle. "
        "CA% discrimine modérément : les Champions tendent vers l'excédent, "
        "les Fragile vers le déficit, mais la variance intra-groupe est élevée. "
        "Le profil combiné est un signal fiable de niveau de développement."
    )
    thin_divider()

    # K.1/K.2 Combined & Structural Score scatter
    _question("Les scores continus (combined_score, structural_score) "
              "sont-ils corrélés aux indicateurs [E] ?")
    col_k1a, col_k1b = st.columns(2)
    with col_k1a:
        df_k1 = df_m[["country_name", GDP_pc, "combined_score",
                       "combined_profile"]].dropna()
        df_k1 = df_k1[df_k1[GDP_pc] > 0]
        _x = np.log10(df_k1[GDP_pc].values)
        _y = df_k1["combined_score"].values
        _poly = np.polyfit(_x, _y, 1)
        _x_line = np.linspace(_x.min(), _x.max(), 100)
        _y_line = np.polyval(_poly, _x_line)
        fig_k1 = _scatter(df_k1, GDP_pc, "combined_score",
                          "combined_profile", _COMB_C,
                          "Combined Score vs GDP/capita (ρ[E]=+0.33)",
                          "GDP per capita (US$, log)", "Combined Score",
                          xlog=True, height=390, x_fmt="usd")
        fig_k1.add_trace(go.Scatter(
            x=10 ** _x_line, y=_y_line, mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=2),
            hoverinfo="skip",
        ))
        st.plotly_chart(fig_k1, use_container_width=True)
    with col_k1b:
        tc_cs = _top_correlates("combined_score", n=6, flag_filter="[E]")
        tc_ss = _top_correlates("structural_score", n=6, flag_filter="[E]")
        tab_cs, tab_ss = st.tabs(["Combined Score", "Structural Score"])
        with tab_cs:
            st.plotly_chart(_corr_bar(tc_cs, "Combined Score × [E]", height=280),
                            use_container_width=True)
        with tab_ss:
            st.plotly_chart(_corr_bar(tc_ss, "Structural Score × [E]", height=280),
                            use_container_width=True)
    finding(
        "<strong>Combined Score ρ[E]=+0.33 (GDP_pc)</strong> : signal modéré. "
        "L'ajout du score de résilience bilatérale (g_norm_rank, non expliqué par macro) "
        "dilue légèrement le signal du structural_score (ρ=+0.32). "
        "La différence Δρ≈0 confirme que la résilience bilatérale n'ajoute ni "
        "n'enlève de signal macro — les deux dimensions sont orthogonales."
    )
    thin_divider()

    # K hierarchy
    section_label("K · Hiérarchie des signaux [E] — Synthèse finale")
    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>"
        "Classement des indicateurs [E] par |ρ| moyen sur toutes les 44 variables "
        "trade — récapitulatif de l'ensemble des sections G à K.</p>",
        unsafe_allow_html=True,
    )
    _e_rank = (df_sp[df_sp["ind_flag"] == "[E]"]
               .groupby("ind_label")["abs_rho"].mean()
               .sort_values(ascending=True))
    _e_colors = ["#10B981" if v > 0.15 else "#F59E0B" if v > 0.08 else "#6890B0"
                 for v in _e_rank]
    fig_hier = go.Figure(go.Bar(
        x=_e_rank.values, y=_e_rank.index,
        orientation="h", marker_color=_e_colors,
        text=_e_rank.values.round(3), textposition="outside",
        hovertemplate="<b>%{y}</b><br>|ρ| moyen = %{x:.3f}<extra></extra>",
    ))
    _pl(fig_hier,
        title="Indicateurs [E] — |ρ| moyen sur toutes les variables trade (G–K)",
        xaxis_title="|ρ| moyen", height=340,
        margin=dict(l=160, r=80, t=50, b=30),
        xaxis=dict(gridcolor="#304870", range=[0, 0.5]))
    st.plotly_chart(fig_hier, use_container_width=True)
    finding(
        "<strong>Hiérarchie G→K confirmée</strong> : "
        "GDP_pc > CA% > Real Rate > Expense% > Inflation > Unemployment. "
        "GDP_pc : signal [E] dominant (richesse → tous les aspects de la position trade). "
        "CA% : deuxième signal, cohérent sur 4+ analyses indépendantes (I.1/I.3/I.5/J.2). "
        "Real Rate et Expense% : signaux secondaires, pertinents pour l'ouverture (X/GDP, M/GDP). "
        "Inflation et Unemployment : signaux faibles, surtout pour la volatilité import."
    )

    # K.5 Two dimensions
    thin_divider()
    section_label("K.5 · Deux dimensions, deux logiques")
    finding(
        "<strong>F (structural score) = expliqué</strong> : "
        "GDP_pc ρ=+0.32, CA% ρ=+0.30 — cohérent sur tout G/H/I/J/K. "
        "Le score structurel mesure la position dans le commerce mondial — "
        "déterminée par la richesse et la balance externe."
    )
    finding(
        "<strong>G (bilateral resilience = g_norm_rank) = non expliqué</strong> : "
        "tous indicateurs n.s. La concentration géographique des partenaires "
        "trade est entièrement idiosyncratique — géographie, histoire, "
        "accords commerciaux, politiques nationales."
    )
    finding(
        "<strong>Implication pour le combined_score</strong> : "
        "en pondérant 70% structural + 30% bilateral, le combined_score hérite "
        "du signal macro du structural et de l'idiosyncrasie du bilateral. "
        "C'est analytiquement correct : les deux dimensions capturent des réalités différentes."
    )

    with st.expander("Conclusion générale G→K"):
        finding(
            "<strong>Structure trade = macro-expliquée (niveau)</strong> : "
            "Export/Import Share [S], Export/Import per Capita [E], "
            "Coverage Ratio [E], profils statiques [E]. "
            "La taille prime (GDP US$ ρ>0.94), la richesse affine (GDP_pc ρ=0.33–0.92)."
        )
        finding(
            "<strong>Dynamiques trade = idiosyncratiques</strong> : "
            "CAGR, volatilité, résilience, consistance, quadrant dynamique — "
            "tous non significatifs [E]. "
            "Ce qu'un pays fait de sa position structurelle dépend de facteurs "
            "non capturés par 16 indicateurs macro sur 10 ans."
        )
        finding(
            "<strong>Taille vs Richesse</strong> : "
            "être grand (GDP US$) prime pour le volume mondial (ρ=0.94–0.96). "
            "Être riche (GDP_pc) prime pour l'efficience per capita (ρ=0.82–0.925). "
            "CA% est le signal causal le plus robuste : excédent → compétitivité externe."
        )

thin_divider()
