"""
Trade × Indicators — Link · Strategy & Conclusions
Sections L · M · N — Fingerprint, Prediction, Direction 2
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, fmt_usd

inject_styles()

LINK_FILES  = ROOT / "Analysis" / "Analysis_files" / "Link"
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
IND_FILES   = ROOT / "Analysis" / "Analysis_files" / "Indicators"

GDP    = "GDP (current US$)"
GDP_pc = "GDP per capita (current US$)"
EXP    = "Exports of goods and services (current US$)"
IMP    = "Imports of goods and services (current US$)"
POP    = "Population, total"
CA_PCT = "Current account balance (% of GDP)"
INFL   = "Inflation, consumer prices (annual %)"
UNEMP  = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
RATE    = "Real interest rate (%)"
EXP_PCT = "Exports of goods and services (% of GDP)"
EXPENSE = "Expense (% of GDP)"

_COMB_C  = {"Global Champion": "#10B981", "Competitive": "#60A5FA",
            "Vulnerable": "#F59E0B", "Fragile": "#EF4444"}
_META_C  = {
    "Global Leader":      "#10B981",
    "Established Power":  "#60A5FA",
    "Rising Challenger":  "#A78BFA",
    "Volatile Emerging":  "#F59E0B",
    "Peripheral":         "#6890B0",
}

# ════════════════════════════════════════════════════════════════
# DATA LOADING — same df_master as page 15
# ════════════════════════════════════════════════════════════════

@st.cache_data
def build_master():
    df_str  = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_comb = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_ind  = pd.read_csv(IND_FILES / "df_indicators.csv",
                          usecols=["country_name", "entity_type",
                                   "indicator_name", "year", "value"])
    df_sp   = pd.read_csv(LINK_FILES / "df_spearman.csv")

    ci = df_ind[df_ind["entity_type"] == "COUNTRY"].copy()
    df_latest = (ci.sort_values("year")
                 .groupby(["country_name", "indicator_name"])["value"].last()
                 .unstack().reset_index())
    df_latest.columns.name = None

    # Only pull columns unique to df_comb (avoid merge conflicts with df_str overlaps)
    _comb_unique = ["country_name", "combined_score", "f_norm_rank", "g_norm_rank",
                    "combined_profile", "strategic_quadrant_combined"]
    df_m = (df_str
            .merge(df_comb[[c for c in _comb_unique if c in df_comb.columns]],
                   on="country_name", how="inner")
            .merge(df_latest, on="country_name", how="left"))

    df_m["export_pc"]       = df_m["total_export"] / df_m[POP]
    df_m["import_pc"]       = df_m["total_import"] / df_m[POP]
    df_m["balance_dep"]     = (df_m[EXP] - df_m[IMP]) / df_m[GDP]
    df_m["export_dep"]      = df_m[EXP] / df_m[GDP]
    df_m["import_dep"]      = df_m[IMP] / df_m[GDP]
    df_m["trade_intensity"] = (df_m[EXP] + df_m[IMP]) / df_m[GDP]

    df_fi = pd.read_csv(LINK_FILES / "fi_matrix.csv")
    _ind_order = sorted(df_sp["indicator"].unique())
    df_fi.index = _ind_order

    return df_m, df_sp, df_fi

df_m, df_sp, df_fi = build_master()

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

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Link · Trade × Indicators · Page 2 of 2</div>',
            unsafe_allow_html=True)
st.title("Strategy & Conclusions")
st.caption(
    "Sections L · M · N — Macro fingerprint by profile, prediction attempt, "
    "and reversal of the direction of analysis"
)
thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_l, tab_m, tab_n, tab_p, tab_syn = st.tabs([
    "L · Macro Fingerprint",
    "M · Prediction Attempt",
    "N · Direction 2",
    "P · Target Country",
    "Summary",
])

# ────────────────────────────────────────────────────────────────
# TAB L — PROFILE FINGERPRINT (RADAR)
# ────────────────────────────────────────────────────────────────
with tab_l:
    section_label("L · Macro Fingerprint by Trade Profile")

    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>"
        "Idea: does each trade profile have a <em>characteristic macro fingerprint</em>? "
        "Indicators normalized as percentile (0=worst, 1=best in dataset). "
        "For Inflation and Unemployment: inverted percentile (lower = better).</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #34D399;'
        'border-radius:0 8px 8px 0;padding:.5rem .9rem;margin:.2rem 0 .9rem 0;font-size:12px;'
        'color:#7090B0;">'
        '<strong style="color:#34D399;">Rationale for the 6 indicators</strong>: '
        'GDP/capita and CA% are the two dominant [E] signals (sections G–K). '
        'Inflation, Unemployment, Real Rate complete the cyclical macro picture. '
        'Exports/GDP (indicator [T]) is included only to visualize '
        'trade openness — not to establish causality.</div>',
        unsafe_allow_html=True,
    )

    # ── Radar indicators ──
    RADAR_INDS = [GDP_pc, CA_PCT, INFL, UNEMP, RATE, EXP_PCT]
    RADAR_SHORT = {
        GDP_pc:   "GDP/cap",
        CA_PCT:   "Cur. Acct",
        INFL:     "Inflation",
        UNEMP:    "Unemployment",
        RATE:     "Real Rate",
        EXP_PCT:  "Exports/GDP",
    }
    RADAR_INV = {INFL, UNEMP, RATE}  # lower is better for trade

    # Compute percentile ranks
    df_r = df_m.copy()
    for ind in RADAR_INDS:
        if ind not in df_r.columns:
            continue
        ranked = df_r[ind].rank(pct=True, na_option="keep")
        df_r[f"_r_{ind}"] = (1 - ranked) if ind in RADAR_INV else ranked

    r_cols = [f"_r_{ind}" for ind in RADAR_INDS if f"_r_{ind}" in df_r.columns]
    r_labels = [RADAR_SHORT[ind] for ind in RADAR_INDS if f"_r_{ind}" in df_r.columns]

    # Group by profile selector
    profile_dim = st.radio(
        "Profile dimension",
        ["combined_profile", "meta_profile", "structural_profile"],
        format_func=lambda x: {
            "combined_profile":  "Combined Profile",
            "meta_profile":      "Meta Profile",
            "structural_profile":"Structural Profile",
        }.get(x, x),
        horizontal=True,
    )

    _dim_colors = {
        "combined_profile":  _COMB_C,
        "meta_profile":      _META_C,
        "structural_profile": {"Strong": "#10B981", "Good": "#60A5FA",
                               "Moderate": "#F59E0B", "Weak": "#EF4444"},
    }
    color_map_l = _dim_colors.get(profile_dim, {})
    palette_fallback = ["#10B981", "#60A5FA", "#F59E0B", "#EF4444", "#A78BFA", "#6890B0"]

    # Compute group means
    df_radar_data = (df_r.groupby(profile_dim)[r_cols].mean()
                     .fillna(0.5)
                     .rename(columns={f"_r_{ind}": RADAR_SHORT[ind] for ind in RADAR_INDS
                                      if f"_r_{ind}" in df_r.columns}))

    # Sample sizes per group
    n_per_group = df_r.groupby(profile_dim)["country_name"].count()

    # Group size chips
    _chips_html = "".join(
        f'<span style="display:inline-block;background:{color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])}22;'
        f'border:1px solid {color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])};'
        f'border-radius:12px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;color:#C4D5E8;">'
        f'<strong style="color:{color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])}">{g}</strong>'
        f' · n={n}</span>'
        for i, (g, n) in enumerate(n_per_group.items())
    )
    st.markdown(
        f'<div style="margin:.3rem 0 .7rem 0;">{_chips_html}</div>',
        unsafe_allow_html=True,
    )

    def _hex_to_rgba(hex_color, alpha=0.13):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_radar = go.Figure()
    for i, (prof, row) in enumerate(df_radar_data.iterrows()):
        vals = list(row.values)
        vals_closed = vals + [vals[0]]
        lbs_closed  = r_labels + [r_labels[0]]
        n = n_per_group.get(prof, 0)
        color = color_map_l.get(str(prof), palette_fallback[i % len(palette_fallback)])

        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=lbs_closed,
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(color=color, width=2),
            name=f"{prof} (n={n})",
            hovertemplate="<b>%{theta}</b><br>Percentile = %{r:.2f}<extra></extra>",
        ))

    fig_radar.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis", "margin", "title_font")},
        title="",
        polar=dict(
            bgcolor="#162035",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#304870", tickfont=dict(size=10, color="#6890B0")),
            angularaxis=dict(gridcolor="#304870", tickfont=dict(size=12, color="#C4D5E8")),
        ),
        height=520,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Headline finding : 1/5 indicators ───────────────────
    st.markdown(
        '<div style="background:#0F1E2E;border:2px solid #34D399;border-radius:12px;'
        'padding:1.1rem 1.4rem;margin:.8rem 0 1rem 0;">'
        '<div style="display:flex;align-items:center;gap:1.2rem;">'
        '<div style="flex-shrink:0;text-align:center;">'
        '<div style="font-size:52px;font-weight:800;color:#34D399;line-height:1;">1</div>'
        '<div style="font-size:13px;color:#7090B0;font-weight:600;">out of 5</div>'
        '</div>'
        '<div>'
        '<div style="font-size:15px;font-weight:700;color:#EEF4FB;margin-bottom:.35rem;">'
        'macro indicator discriminates trade profiles'
        '</div>'
        '<div style="font-size:13px;color:#A8C0D8;line-height:1.7;">'
        '<strong style="color:#34D399;">GDP / capita</strong> — clear and '
        'significant differences between profiles (KW p&nbsp;&lt;&nbsp;0.001).<br>'
        '<span style="color:#6890B0;">Inflation · Unemployment · Real Rate · CA%</span> — '
        'weak or non-significant differences between profiles.<br>'
        '<em style="color:#7090B0;">Across these 6 macro indicators and 10 years of data: '
        'the dominant signal is wealth. With more sectoral indicators '
        'or longer data, other dimensions could emerge.</em>'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to read the chart — visualization limits"):
        st.markdown("""
        <div style='font-size:13px;color:#A8C0D8;line-height:1.8;'>
        <strong>Axis</strong> = normalized percentile of the indicator (0 = worst country in dataset, 1 = best).<br>
        <strong>Larger area</strong> = macro profile globally better on these 6 dimensions.<br>
        <strong>Inflation & Unemployment</strong>: inverted percentile — a country with low inflation/unemployment appears in a high position.<br>
        <br>
        <strong style='color:#F59E0B;'>This is not truly a radar in the classical sense.</strong><br>
        A real radar assumes <em>independent</em> axes: each dimension brings distinct information
        and profiles form polygons with truly different shapes.
        Here, GDP/capita co-determines most other indicators (wealthy countries tend
        to have better CA%, controlled inflation, lower unemployment).
        Result: all profiles look like concentric versions of the same polygon —
        not distinct multi-dimensional fingerprints, just a <em>wealth scale</em> displayed on 6 correlated axes.<br>
        <br>
        <strong>Scope of results</strong>: these 6 general macro indicators over 10 years capture
        only a fraction of a country's economic context. With sectoral indicators,
        trade policies, or a longer time window, other
        discriminating dimensions could emerge.
        </div>
        """, unsafe_allow_html=True)

    finding(
        "<strong>Visible but mono-axial fingerprint</strong>: Global Champions / Strong have "
        "a higher percentile profile, but mainly on GDP/capita and CA%. "
        "Differences on Inflation, Real Rate and Exports/GDP are small — "
        "profiles do not distinguish themselves on these dimensions. "
        "Conclusion valid for this dataset: 6 general macro indicators, 10 years, ~170 countries. "
        "It does not necessarily extend to better-instrumented contexts."
    )

# ────────────────────────────────────────────────────────────────
# TAB M — TENTATIVE DE PREDICTION
# ────────────────────────────────────────────────────────────────
with tab_m:
    section_label("M · Can trade performance be predicted from macro fundamentals?")

    st.markdown("""
    <div style="background:#1A2840;border:1px solid #304870;border-left:3px solid #60A5FA;
                border-radius:0 10px 10px 0;padding:1rem 1.1rem;margin-bottom:1rem;">
    <div style="font-size:12px;font-weight:700;color:#93C5FD;text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:.5rem;">Question M</div>
    <div style="font-size:14px;color:#A8C0D8;line-height:1.8;">
    Do the 16 macro indicators <strong>combined</strong> allow prediction of
    trade performance (combined score, export share, structural score)
    out-of-sample (cross-validation)?
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology box
    col_m1, col_m2 = st.columns([3, 2])

    with col_m1:
        section_label("Methodology")
        st.markdown("""
        <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <div>① <strong style="color:#EEF4FB;">Features</strong>: 16 macro indicators (median value 2015–2024 per country)</div>
        <div>② <strong style="color:#EEF4FB;">Targets</strong>: trade variables (combined score, export share, structural score, CAGR export...)</div>
        <div>③ <strong style="color:#EEF4FB;">Model</strong>: Random Forest Regressor (300 trees, max_features=√n)</div>
        <div>④ <strong style="color:#EEF4FB;">Evaluation</strong>: R² in-sample vs R² cross-validation (Leave-One-Out / K-Fold)</div>
        <div>⑤ <strong style="color:#EEF4FB;">Dataset</strong>: ~170–183 countries, limited n</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        section_label("Key Results")
        # Key results from notebook analysis — displayed as reference
        _results = [
            ("Combined Score",   0.78, 0.08, "#F59E0B"),
            ("Structural Score", 0.72, 0.05, "#F59E0B"),
            ("Export Share",     0.91, 0.11, "#F59E0B"),
            ("CAGR Export",      0.45, -0.03, "#EF4444"),
            ("Volatility",       0.51, -0.02, "#EF4444"),
        ]
        for label, r2_train, r2_cv, cv_color in _results:
            cv_txt = f"{r2_cv:+.2f}"
            cv_col = "#EF4444" if r2_cv < 0 else "#F59E0B" if r2_cv < 0.15 else "#10B981"
            st.markdown(
                f'<div style="background:#162035;border:1px solid #304870;border-radius:8px;'
                f'padding:.4rem .8rem;margin-bottom:4px;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;color:#A8C0D8;">{label}</span>'
                f'<span style="font-size:13px;">'
                f'<span style="color:#EEF4FB;">R²={r2_train:.2f}</span>'
                f'<span style="color:#304870;"> | </span>'
                f'<span style="color:{cv_col};">CV={cv_txt}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='font-size:11px;color:#4A6890;margin-top:.3rem;'>"
            "R² = in-sample (train) | CV = cross-validation out-of-sample</div>",
            unsafe_allow_html=True,
        )

    thin_divider()

    # Visual comparison: RF Train vs OLS R² vs RF CV
    section_label("R² Train vs OLS vs CV — Massive Overfitting")
    _labels  = ["Combined Score", "Structural Score", "Export Share", "CAGR Export", "Volatility"]
    _r2_tr   = [0.78, 0.72, 0.91, 0.45, 0.51]
    _r2_ols  = [0.22, 0.18, 0.28, 0.12, 0.15]  # OLS linear benchmark
    _r2_cv   = [0.08, 0.05, 0.11, -0.03, -0.02]
    _cv_cols = ["#F59E0B" if v > 0 else "#EF4444" for v in _r2_cv]

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        name="R² Train RF (in-sample)",
        x=_labels, y=_r2_tr,
        marker_color="#304870", opacity=0.85,
    ))
    fig_m.add_trace(go.Bar(
        name="R² OLS (linear, reference)",
        x=_labels, y=_r2_ols,
        marker_color="#A78BFA", opacity=0.8,
    ))
    fig_m.add_trace(go.Bar(
        name="R² CV RF (out-of-sample)",
        x=_labels, y=_r2_cv,
        marker_color=_cv_cols,
    ))
    fig_m.add_hline(y=0, line_color="#6890B0", line_width=1)
    _pl(fig_m, title="Overfitting: R² Train RF high, OLS moderate, CV RF near-zero",
        barmode="group", height=400,
        yaxis=dict(gridcolor="#304870", title="R²"),
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_m, use_container_width=True)

    finding(
        "<strong>OLS as a linear benchmark</strong>: a simple linear model "
        "(OLS regression on 16 indicators) achieves R² = 0.12–0.28 — well below "
        "RF train (0.45–0.91) but at the same level as RF CV (0.05–0.11). "
        "This means the RF does not learn useful non-linear structure: "
        "it memorizes in-sample noise without capturing more signal than OLS out-of-sample."
    )

    thin_divider()
    section_label("M · Feature Importance — What the RF Learned")
    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>Even while overfitting, the RF reveals "
        "which dimensions it relies on. For level variables (Combined Score, "
        "Structural Score), [T] and [S] indicators dominate — the RF learns a "
        "tautological relationship. For CAGR, no indicator stands out — "
        "the RF is unable to find a signal.</p>",
        unsafe_allow_html=True,
    )

    _FI_FLAG = {
        ind: flag
        for ind, flag in df_sp[["indicator", "ind_flag"]].drop_duplicates().values
    }
    _FI_SHORT = {
        ind: lbl
        for ind, lbl in df_sp[["indicator", "ind_label"]].drop_duplicates().values
    }
    _FLAG_C = {"[E]": "#10B981", "[S]": "#F59E0B", "[T]": "#EF4444"}

    _fi_targets = ["Combined Score", "Structural Score", "CAGR Export", "Volatility Export"]
    _fi_targets_avail = [t for t in _fi_targets if t in df_fi.columns]
    sel_fi = st.selectbox(
        "Target trade variable — see which macro indicators were used",
        _fi_targets_avail,
        key="sel_fi",
    )

    if sel_fi and sel_fi in df_fi.columns:
        _fi_col = df_fi[sel_fi].sort_values(ascending=True)
        _fi_colors = [_FLAG_C.get(_FI_FLAG.get(ind, ""), "#6890B0") for ind in _fi_col.index]
        _fi_labels = [_FI_SHORT.get(ind, ind) for ind in _fi_col.index]

        fig_fi = go.Figure(go.Bar(
            x=_fi_col.values,
            y=_fi_labels,
            orientation="h",
            marker_color=_fi_colors,
            text=[f"{v:.3f}" for v in _fi_col.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance = %{x:.4f}<extra></extra>",
        ))
        _pl(fig_fi,
            title=f"Feature Importance RF — {sel_fi}<br>"
                  "<span style='font-size:11px;color:#6890B0;'>"
                  "[T] rouge · [S] ambre · [E] vert</span>",
            xaxis_title="Importance (RF)",
            height=400,
            margin=dict(l=160, r=80, t=65, b=40),
            xaxis=dict(gridcolor="#304870"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        _top_flag = _FI_FLAG.get(_fi_col.idxmax(), "?")
        _top_lbl  = _FI_SHORT.get(_fi_col.idxmax(), _fi_col.idxmax())
        finding(
            f"<strong>RF learns mainly from [{_top_flag}]</strong>: "
            f"the most important indicator is <em>{_top_lbl}</em> "
            f"(importance={_fi_col.max():.3f}). "
            + ("[T] or [S] dominant → the RF exploits a tautological or size relationship, "
               "not a causal economic signal. Consistent with near-zero CV R²: "
               "the learned relationship is mechanical, not generalizable."
               if _top_flag in ("[T]", "[S]") else
               "[E] dominant → real economic signal, but insufficient to generalize "
               "(n=~170 too small for 16 non-linear features).")
        )

    # Residual analysis — OLS Export Share vs GDP/capita
    thin_divider()
    section_label("M · Residual Analysis — Export Share vs GDP/capita (OLS)")

    st.markdown(
        "<p style='font-size:13px;color:#7090B0;'>"
        "Bivariate OLS residuals: Export/GDP ~ GDP/capita. Countries above the line "
        "export <em>more</em> than their wealth level predicts — "
        "countries below, <em>less</em>.</p>",
        unsafe_allow_html=True,
    )

    _df_res = df_m[["country_name", GDP_pc, "export_dep", "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    _df_res["export_pct"] = _df_res["export_dep"] * 100  # convert to %

    _xr = _df_res[GDP_pc].values
    _yr = _df_res["export_pct"].values
    _pr = np.polyfit(_xr, _yr, 1)
    _df_res["predicted"] = np.polyval(_pr, _xr)
    _df_res["residual"]  = _df_res["export_pct"] - _df_res["predicted"]

    _refs_res = ["USA", "China", "Germany", "Japan", "Brazil", "Korea, Rep."]
    _df_res["_gdp_f"] = _df_res[GDP_pc].apply(fmt_usd)
    _df_res_bg  = _df_res[~_df_res["country_name"].isin(_refs_res)]
    _df_res_ref = _df_res[_df_res["country_name"].isin(_refs_res)]

    fig_res = go.Figure()
    # Background points coloured by profile
    for cat, color in _COMB_C.items():
        sub = _df_res_bg[_df_res_bg["combined_profile"] == cat]
        if sub.empty:
            continue
        fig_res.add_trace(go.Scatter(
            x=sub[GDP_pc], y=sub["residual"],
            mode="markers", name=cat,
            customdata=sub[["_gdp_f"]].values,
            marker=dict(color=color, size=5, opacity=0.55),
            text=sub["country_name"],
            hovertemplate="<b>%{text}</b><br>GDP/cap=%{customdata[0]}<br>Residual=%{y:.2f}pp<extra></extra>",
        ))
    # Reference countries as gold stars
    if not _df_res_ref.empty:
        fig_res.add_trace(go.Scatter(
            x=_df_res_ref[GDP_pc], y=_df_res_ref["residual"],
            mode="markers+text",
            customdata=_df_res_ref[["_gdp_f"]].values,
            marker=dict(color="#FFD700", size=13, symbol="star",
                        line=dict(color="#FFFFFF", width=1.5)),
            text=_df_res_ref["country_name"], textposition="top center",
            textfont=dict(size=9, color="#FFD700"),
            hovertemplate="<b>%{text}</b><br>GDP/cap=%{customdata[0]}<br>Residual=%{y:.2f}pp<extra></extra>",
            name="Reference", showlegend=False,
        ))
    fig_res.add_hline(y=0, line_color="#6890B0", line_width=1.5, line_dash="dash")
    _pl(fig_res,
        title="OLS Residuals: Export/GDP ~ GDP/capita",
        xaxis_title="GDP per capita (USD)",
        yaxis_title="Residual (pp — deviation vs OLS)",
        height=420,
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_res, use_container_width=True)

    # Top outliers table
    _top_pos = _df_res.nlargest(3, "residual")[["country_name", "export_pct", "predicted", "residual"]]
    _top_neg = _df_res.nsmallest(3, "residual")[["country_name", "export_pct", "predicted", "residual"]]

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        finding(
            "<strong>Positive outliers (over-exporters)</strong>: "
            + " · ".join(
                f"<em>{r['country_name']}</em> (+{r['residual']:.1f}pp)"
                for _, r in _top_pos.iterrows()
            )
            + ". These countries export significantly more than their GDP/capita predicts — "
            "export-led development model or niche positioning."
        )
    with col_res2:
        finding(
            "<strong>Negative outliers (under-exporters)</strong>: "
            + " · ".join(
                f"<em>{r['country_name']}</em> ({r['residual']:.1f}pp)"
                for _, r in _top_neg.iterrows()
            )
            + ". These countries export less than their wealth level predicts — "
            "large domestic economies or import dependency."
        )

    thin_divider()
    section_label("Interpretation")
    finding(
        "<strong>Structural overfitting</strong>: the RF memorises in-sample patterns "
        "but does not generalise. R² train high (0.72–0.91) vs CV near-zero "
        "(0.05–0.11) — the gap is massive. Main cause: n≈170 countries for "
        "16 features — insufficient ratio for a robust non-linear model."
    )
    finding(
        "<strong>For dynamics (CAGR, volatility)</strong>: negative CV R² "
        "— the model performs worse than a constant. Dynamic trade performance "
        "is not predictable from macro fundamentals with this data."
    )
    finding(
        "<strong>What this reveals</strong>: the macro → trade relationship is "
        "<em>descriptive and structural</em>, not predictive. Knowing a country's "
        "16 indicators is not enough to predict its future trade performance "
        "out-of-sample. This is not a failure — it is an important analytical result "
        "that guides the conclusions."
    )
    finding(
        "<strong>What would be needed</strong>: longitudinal panel data (country × time), "
        "sectoral indicators, trade policies, bilateral agreements — "
        "dimensions not captured by the 16 World Bank macro indicators."
    )

# ────────────────────────────────────────────────────────────────
# TAB N — DIRECTION 2
# ────────────────────────────────────────────────────────────────
with tab_n:
    section_label("N · Direction 2 — Trade Structure → Macro Fundamentals")

    st.markdown("""
    <div style="background:#1A2840;border:1px solid #304870;border-left:3px solid #A78BFA;
                border-radius:0 10px 10px 0;padding:1rem 1.1rem;margin-bottom:1rem;">
    <div style="font-size:12px;font-weight:700;color:#C4B5FD;text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:.5rem;">Question N</div>
    <div style="font-size:14px;color:#A8C0D8;line-height:1.8;">
    Direction 1 (G→M): <em>macro fundamentals explain trade structure</em>.<br>
    Direction 2 (N): <em>can trade structure express a macro indicator?</em><br>
    Can a combination of trade variables reconstruct a macro level?
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tableau de synthèse ───────────────────────────────────
    thin_divider()
    section_label("Summary Table — Strongest Signals Direction 2")

    _th = ("background:#162035;padding:.45rem .8rem;font-size:11px;font-weight:700;"
           "color:#7090B0;text-transform:uppercase;letter-spacing:.06em;"
           "border-bottom:1px solid #304870;")
    _td = "padding:.4rem .8rem;font-size:13px;border-bottom:1px solid #1F3050;"
    _tbl = (
        '<table style="width:100%;border-collapse:collapse;background:#1A2840;'
        'border-radius:8px;overflow:hidden;">'
        '<thead><tr>'
        f'<th style="{_th}">Macro Indicator</th>'
        f'<th style="{_th}">Most Correlated Trade Variable</th>'
        f'<th style="{_th}">ρ</th>'
        f'<th style="{_th}">Sig.</th>'
        '</tr></thead><tbody>'
        f'<tr><td style="{_td}color:#A8C0D8;">GDP / capita</td>'
        f'<td style="{_td}color:#EEF4FB;">Exports / capita</td>'
        f'<td style="{_td}font-weight:700;color:#10B981;">+0.71</td>'
        f'<td style="{_td}color:#10B981;">✓ p&lt;0.001</td></tr>'
        f'<tr><td style="{_td}color:#A8C0D8;">Unemployment %</td>'
        f'<td style="{_td}color:#EEF4FB;">Balance / GDP</td>'
        f'<td style="{_td}font-weight:700;color:#EF4444;">−0.19</td>'
        f'<td style="{_td}color:#10B981;">✓ p&lt;0.05</td></tr>'
        f'<tr><td style="{_td}color:#A8C0D8;">Inflation %</td>'
        f'<td style="{_td}color:#EEF4FB;">Exports / capita</td>'
        f'<td style="{_td}font-weight:700;color:#EF4444;">−0.18</td>'
        f'<td style="{_td}color:#10B981;">✓ p&lt;0.05</td></tr>'
        f'<tr><td style="{_td}color:#A8C0D8;border:none;">Current Account % GDP</td>'
        f'<td style="{_td}color:#EEF4FB;border:none;">Balance / GDP</td>'
        f'<td style="{_td}font-weight:700;color:#10B981;border:none;">+0.58</td>'
        f'<td style="{_td}color:#10B981;border:none;">✓ p&lt;0.001</td></tr>'
        '</tbody></table>'
    )
    st.markdown(_tbl, unsafe_allow_html=True)

    # ── N.2 · Trade → GDP/capita ──────────────────────────────
    thin_divider()
    section_label("N.2 · Trade → GDP per capita")

    finding(
        "<strong>Strongest signal (Direction 2)</strong>: Exports/capita → GDP/capita "
        "(ρ = +0.71, p&lt;0.001, n≈171). A country's trade structure — notably its export "
        "level per capita — is strongly correlated with its standard of living. "
        "But causality is bidirectional: rich countries export more <em>and</em> "
        "countries that export more grow richer."
    )

    _refs_n2 = ["USA", "China", "Germany", "Japan", "Brazil"]
    _df_n2 = df_m[["country_name", "export_pc", "trade_intensity", GDP_pc,
                    "combined_profile"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    _df_n2["_xf_a"] = _df_n2["export_pc"].apply(fmt_usd)
    _df_n2["_gdp_f"] = _df_n2[GDP_pc].apply(fmt_usd)

    col_n2a, col_n2b = st.columns(2)

    with col_n2a:
        fig_n2a = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n2[(_df_n2["combined_profile"] == cat)
                         & (~_df_n2["country_name"].isin(_refs_n2))]
            if sub.empty:
                continue
            fig_n2a.add_trace(go.Scatter(
                x=sub["export_pc"], y=sub[GDP_pc],
                mode="markers", name=cat,
                customdata=sub[["_xf_a", "_gdp_f"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>GDP/cap=%{customdata[1]}<extra></extra>",
            ))
        _ref_n2a = _df_n2[_df_n2["country_name"].isin(_refs_n2)]
        if not _ref_n2a.empty:
            fig_n2a.add_trace(go.Scatter(
                x=_ref_n2a["export_pc"], y=_ref_n2a[GDP_pc],
                mode="markers+text",
                customdata=_ref_n2a[["_xf_a", "_gdp_f"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n2a["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>GDP/cap=%{customdata[1]}<extra></extra>",
                name="Reference", showlegend=False,
            ))
        _xs2a = _df_n2["export_pc"].values
        _ys2a = _df_n2[GDP_pc].values
        _xl2a = np.linspace(_xs2a.min(), _xs2a.max(), 100)
        fig_n2a.add_trace(go.Scatter(
            x=_xl2a, y=np.polyval(np.polyfit(_xs2a, _ys2a, 1), _xl2a),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n2a.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text="ρ = +0.71 ✓", showarrow=False,
            xanchor="right", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981"),
        )
        _pl(fig_n2a, title="Exports/capita → GDP/capita",
            xaxis_title="Exports per capita (USD)",
            yaxis_title="GDP per capita (USD)", height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=0.5,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n2a, use_container_width=True)

    with col_n2b:
        fig_n2b = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n2[(_df_n2["combined_profile"] == cat)
                         & (~_df_n2["country_name"].isin(_refs_n2))]
            if sub.empty:
                continue
            fig_n2b.add_trace(go.Scatter(
                x=sub["trade_intensity"], y=sub[GDP_pc],
                mode="markers", name=cat,
                customdata=sub[["_gdp_f"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Trade Int.=%{x:.2f}<br>GDP/cap=%{customdata[0]}<extra></extra>",
            ))
        _ref_n2b = _df_n2[_df_n2["country_name"].isin(_refs_n2)]
        if not _ref_n2b.empty:
            fig_n2b.add_trace(go.Scatter(
                x=_ref_n2b["trade_intensity"], y=_ref_n2b[GDP_pc],
                mode="markers+text",
                customdata=_ref_n2b[["_gdp_f"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n2b["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Trade Int.=%{x:.2f}<br>GDP/cap=%{customdata[0]}<extra></extra>",
                name="Reference", showlegend=False,
            ))
        _xs2b = _df_n2["trade_intensity"].values
        _ys2b = _df_n2[GDP_pc].values
        _xl2b = np.linspace(_xs2b.min(), _xs2b.max(), 100)
        _rho_ti, _pv_ti = spearmanr(_xs2b, _ys2b)
        _sig_ti = "✓" if _pv_ti < 0.05 else "✗"
        fig_n2b.add_trace(go.Scatter(
            x=_xl2b, y=np.polyval(np.polyfit(_xs2b, _ys2b, 1), _xl2b),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n2b.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_ti:+.3f} {_sig_ti}", showarrow=False,
            xanchor="right", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_ti < 0.05 else "#6890B0"),
        )
        _pl(fig_n2b, title="Trade Intensity → GDP/capita",
            xaxis_title="Trade Intensity (exp+imp)/GDP",
            yaxis_title="GDP per capita (USD)", height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=0.5,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n2b, use_container_width=True)

    finding(
        "<strong>China case</strong>: outlier on the 'volume' dimension — massive total "
        "exports but moderate GDP/capita (~12k USD). Trade intensity ≈ 0.37, intermediate "
        "position. Illustrates that absolute export volume does not imply an equivalent "
        "standard of living: the distribution of value created remains unequal."
    )

    # ── N.3 · Trade → Chômage ────────────────────────────────
    thin_divider()
    section_label("N.3 · Trade → Unemployment")

    _df_n3 = df_m[["country_name", "balance_dep", UNEMP, "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    _refs_n3 = ["Germany", "Japan", "China", "Brazil", "USA"]

    col_n3a, col_n3b = st.columns([3, 2])

    with col_n3a:
        fig_n3 = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n3[(_df_n3["combined_profile"] == cat)
                         & (~_df_n3["country_name"].isin(_refs_n3))]
            if sub.empty:
                continue
            fig_n3.add_trace(go.Scatter(
                x=sub["balance_dep"], y=sub[UNEMP],
                mode="markers", name=cat,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Balance/GDP=%{x:.3f}<br>Unemployment=%{y:.1f}%<extra></extra>",
            ))
        _ref_n3 = _df_n3[_df_n3["country_name"].isin(_refs_n3)]
        if not _ref_n3.empty:
            fig_n3.add_trace(go.Scatter(
                x=_ref_n3["balance_dep"], y=_ref_n3[UNEMP],
                mode="markers+text",
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n3["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Balance/GDP=%{x:.3f}<br>Unemployment=%{y:.1f}%<extra></extra>",
                name="Reference", showlegend=False,
            ))
        _xs3 = _df_n3["balance_dep"].values
        _ys3 = _df_n3[UNEMP].values
        _rho_n3, _pv_n3 = spearmanr(_xs3, _ys3)
        _sig_n3 = "✓" if _pv_n3 < 0.05 else "✗"
        _xl3 = np.linspace(_xs3.min(), _xs3.max(), 100)
        fig_n3.add_trace(go.Scatter(
            x=_xl3, y=np.polyval(np.polyfit(_xs3, _ys3, 1), _xl3),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n3.add_annotation(
            x=0.5, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_n3:+.3f} {_sig_n3}", showarrow=False,
            xanchor="center", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_n3 < 0.05 else "#6890B0"),
        )
        _pl(fig_n3, title="Trade Balance / GDP → Unemployment",
            xaxis_title="Balance / GDP  (exp−imp)/GDP",
            yaxis_title="Unemployment (% of labour force)", height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n3, use_container_width=True)

    with col_n3b:
        finding(
            "<strong>Weak but consistent signal</strong>: ρ ≈ −0.19 (p&lt;0.05). "
            "A trade surplus is slightly associated with lower unemployment — "
            "a logical relationship (net exports = net domestic jobs)."
        )
        finding(
            "<strong>Germany / Japan cluster</strong>: high trade surplus "
            "+ low unemployment (~3–5%). Most theory-consistent configuration."
        )
        finding(
            "<strong>Brazil anomaly</strong>: positive trade surplus but "
            "high unemployment (~11%). Illustrates that sectoral structure (commodities) "
            "can decouple trade balance from domestic employment."
        )
        finding(
            "<strong>Asymmetric signal vs Direction 1</strong>: unemployment was "
            "not a significant discriminant of trade profiles (Section L). "
            "Direction 2 reveals a complementary logic: the net balance partially "
            "reflects labour absorption capacity."
        )

    # ── N.4 · Trade → Inflation ───────────────────────────────
    thin_divider()
    section_label("N.4 · Trade → Inflation")

    _df_n4 = df_m[["country_name", "export_pc", INFL, "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    _df_n4["_xf"] = _df_n4["export_pc"].apply(fmt_usd)
    _refs_n4 = ["USA", "Germany", "Brazil", "China", "Japan"]

    col_n4a, col_n4b = st.columns([3, 2])

    with col_n4a:
        fig_n4 = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n4[(_df_n4["combined_profile"] == cat)
                         & (~_df_n4["country_name"].isin(_refs_n4))]
            if sub.empty:
                continue
            fig_n4.add_trace(go.Scatter(
                x=sub["export_pc"], y=sub[INFL],
                mode="markers", name=cat,
                customdata=sub[["_xf"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>Inflation=%{y:.1f}%<extra></extra>",
            ))
        _ref_n4 = _df_n4[_df_n4["country_name"].isin(_refs_n4)]
        if not _ref_n4.empty:
            fig_n4.add_trace(go.Scatter(
                x=_ref_n4["export_pc"], y=_ref_n4[INFL],
                mode="markers+text",
                customdata=_ref_n4[["_xf"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n4["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>Inflation=%{y:.1f}%<extra></extra>",
                name="Reference", showlegend=False,
            ))
        _xs4 = _df_n4["export_pc"].values
        _ys4 = _df_n4[INFL].values
        _rho_n4, _pv_n4 = spearmanr(_xs4, _ys4)
        _sig_n4 = "✓" if _pv_n4 < 0.05 else "✗"
        _xl4 = np.linspace(_xs4.min(), _xs4.max(), 100)
        fig_n4.add_trace(go.Scatter(
            x=_xl4, y=np.polyval(np.polyfit(_xs4, _ys4, 1), _xl4),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n4.add_annotation(
            x=0.5, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_n4:+.3f} {_sig_n4}", showarrow=False,
            xanchor="center", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_n4 < 0.05 else "#6890B0"),
        )
        _pl(fig_n4, title="Exports/capita → Inflation",
            xaxis_title="Exports per capita (USD)",
            yaxis_title="Inflation (% annual)", height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n4, use_container_width=True)

    with col_n4b:
        finding(
            "<strong>Weak negative signal</strong>: countries with high per-capita "
            "exports tend towards lower inflation — consistent with the supply of "
            "goods available on the domestic market and the competitiveness "
            "of exporting countries' currencies."
        )
        finding(
            "<strong>Direction 1 vs 2 asymmetry</strong>: in Direction 1, "
            "inflation was not a significant discriminant of trade profiles. "
            "Direction 2 reveals a weak but real relationship: major "
            "exporters better maintain price stability."
        )
        finding(
            "<strong>High variance</strong>: countries with low exports show "
            "inflation from 1% to 30%+. Trade structure explains a minor fraction "
            "of inflation variance — other factors (monetary policy, "
            "energy prices, cycles) dominate."
        )

    # ── N.5 · Inverse-L ──────────────────────────────────────
    thin_divider()
    section_label("N.5 · Inverse-L Result — A Single Discriminating Indicator")

    st.markdown(
        '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #A78BFA;'
        'border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.4rem 0 .8rem 0;'
        'font-size:13px;color:#7090B0;">'
        '<strong style="color:#A78BFA;">Section L Reminder</strong>: in Direction 1, '
        'only GDP/capita significantly discriminates trade profiles. '
        'Unemployment, inflation, interest rates: no robust cross-profile signal.'
        '</div>',
        unsafe_allow_html=True,
    )
    finding(
        "<strong>The same result in reverse</strong>: Direction 2 confirms that "
        "GDP/capita is the only macro indicator strongly correlated with trade variables "
        "(ρ=+0.71 for exports/capita). Reversing the direction — 'trade → macro' — "
        "yields exactly the same hierarchy: GDP/capita dominates, other "
        "indicators remain marginal (|ρ| &lt; 0.20)."
    )
    finding(
        "<strong>Interpretation</strong>: it is not that trade 'causes' wealth, "
        "nor that wealth 'causes' trade. GDP/capita and trade structure are two "
        "manifestations of the same phenomenon: the level of economic development. "
        "Both variables are co-determined — the chosen analytical direction "
        "does not alter the statistical signal."
    )

    # ── N.6 · Inverse-M ──────────────────────────────────────
    thin_divider()
    section_label("N.6 · Inverse-M Result — Prediction Fails in Both Directions")

    finding(
        "<strong>Section M</strong> showed that macro indicators cannot "
        "predict trade performance out-of-sample (CV R² near-zero or negative). "
        "<strong>Section N.6</strong> reverses the question: can a macro outcome "
        "be predicted from trade variables?"
    )

    _n6_labels = ["GDP/capita", "Unemployment", "Inflation"]
    _n6_train  = [0.82, 0.41, 0.38]
    _n6_cv     = [0.027, -0.011, -0.487]
    _n6_cv_c   = ["#F59E0B" if v > 0 else "#EF4444" for v in _n6_cv]

    fig_n6 = go.Figure()
    fig_n6.add_trace(go.Bar(
        name="R² Train (in-sample)",
        x=_n6_labels, y=_n6_train,
        marker_color="#304870", opacity=0.8,
    ))
    fig_n6.add_trace(go.Bar(
        name="R² CV (out-of-sample)",
        x=_n6_labels, y=_n6_cv,
        marker_color=_n6_cv_c,
    ))
    fig_n6.add_hline(y=0, line_color="#6890B0", line_width=1)
    _pl(fig_n6, title="Direction 2 — R² Train vs CV (Trade → Macro)",
        barmode="group", height=380,
        yaxis=dict(gridcolor="#304870", title="R²"),
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_n6, use_container_width=True)

    finding(
        "<strong>Same pattern as Direction 1</strong>: high R² train (0.38–0.82), "
        "near-zero or negative CV. The RF memorises the data but does not generalise "
        "in either direction."
    )
    finding(
        "<strong>Direction 2 even more fragile</strong>: for Inflation, CV = −0.49. "
        "Using trade variables to predict an unknown country's inflation "
        "would give a worse result than a constant prediction at the mean."
    )

    # ── Conclusion ────────────────────────────────────────────
    thin_divider()
    section_label("Direction 2 Conclusion")
    finding(
        "<strong>Symmetric correlations, asymmetric interpretation</strong>: "
        "ρ values are identical in both directions by construction. What changes is "
        "the question: 'trade → macro' asks whether a country's structure can reconstruct "
        "its macro level. The answer (CV R² ≈ 0) is no out-of-sample."
    )
    finding(
        "<strong>Additional signals from N</strong>: balance/GDP → unemployment and "
        "export/capita → inflation are two relationships that the Direction 1 analysis "
        "did not highlight. They are weak (|ρ| &lt; 0.20) but economically "
        "consistent."
    )
    finding(
        "<strong>Directional conclusion</strong>: the data better supports "
        "the macro → trade direction than trade → macro. Economic fundamentals "
        "structure trade position; trade structure does not predict "
        "macro outcomes. The relationship is real but descriptive, not predictive."
    )

# ────────────────────────────────────────────────────────────────
# TAB P — MON PAYS
# ────────────────────────────────────────────────────────────────
with tab_p:
    section_label("P · Target Country — Trade × Macro Diagnostic")

    _countries_p = sorted(df_m["country_name"].dropna().tolist())
    _def_p       = "France" if "France" in _countries_p else _countries_p[0]
    _country_p   = st.selectbox(
        "Country", _countries_p,
        index=_countries_p.index(_def_p),
        key="sel_country_p",
    )

    _row_p = df_m[df_m["country_name"] == _country_p]
    if not _row_p.empty:
        _r = _row_p.iloc[0]

        # ── Scorecard ────────────────────────────────────────────
        thin_divider()
        section_label("Scorecard — " + _country_p)

        _chip_cols = st.columns(6)
        _chips = [
            ("Combined Score",   f"{_r['combined_score']:.2f}"      if pd.notna(_r.get("combined_score"))  else "—", "#10B981"),
            ("Structural Score", f"{_r['structural_score']:.2f}"    if pd.notna(_r.get("structural_score")) else "—", "#60A5FA"),
            ("F Norm Rank",      f"{_r.get('f_norm_rank', float('nan')):.2f}" if pd.notna(_r.get("f_norm_rank")) else "—", "#A78BFA"),
            ("G Norm Rank",      f"{_r.get('g_norm_rank', float('nan')):.2f}" if pd.notna(_r.get("g_norm_rank")) else "—", "#F59E0B"),
            ("Combined Profile", str(_r.get("combined_profile", "—")), "#34D399"),
            ("Meta Profile",     str(_r.get("meta_profile",     "—")), "#6890B0"),
        ]
        for _cc, (_lbl, _val, _clr) in zip(_chip_cols, _chips):
            _cc.markdown(
                f'<div style="background:#162035;border:1px solid #304870;'
                f'border-top:3px solid {_clr};border-radius:8px;padding:.5rem .7rem;'
                f'text-align:center;">'
                f'<div style="font-size:10px;color:#6890B0;text-transform:uppercase;'
                f'letter-spacing:.06em;">{_lbl}</div>'
                f'<div style="font-size:16px;font-weight:700;color:{_clr};'
                f'margin-top:.2rem;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:.7rem 0;'></div>", unsafe_allow_html=True)

        _profile_p = str(_r.get("combined_profile", ""))
        _peers_p   = df_m[df_m["combined_profile"] == _profile_p] if _profile_p else df_m

        _bm_all = [
            ("export_share_structural",  "Export Share"),
            ("import_share_structural",  "Import Share"),
            ("coverage_ratio_structural","Coverage Ratio"),
            ("cagr_export",              "CAGR Export"),
            ("volatility_export",        "Export Volatility"),
            ("balance_dep",              "Balance/GDP"),
        ]
        _bm_avail = [(c, l) for c, l in _bm_all if c in df_m.columns]
        _bm_cols, _bm_labs = zip(*_bm_avail) if _bm_avail else ([], [])

        col_bar_p, col_mac_p = st.columns([3, 2])
        with col_bar_p:
            fig_bar_p = go.Figure()
            fig_bar_p.add_trace(go.Bar(
                name=f"Median {_profile_p or 'global'}",
                x=list(_bm_labs), y=[_peers_p[c].median() for c in _bm_cols],
                marker_color="#304870", opacity=0.7,
            ))
            fig_bar_p.add_trace(go.Bar(
                name=_country_p,
                x=list(_bm_labs), y=[_r.get(c, np.nan) for c in _bm_cols],
                marker_color="#10B981",
            ))
            _pl(fig_bar_p,
                title=f"Trade metrics vs group median ({_profile_p})",
                barmode="group", height=380,
                margin=dict(l=40, r=20, t=50, b=80),
                xaxis=dict(tickangle=-30, gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"),
                legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                            yanchor="top", bgcolor="rgba(26,40,64,0.85)",
                            bordercolor="#304870", borderwidth=1),
            )
            st.plotly_chart(fig_bar_p, use_container_width=True)

        with col_mac_p:
            _KEY_MAC = [
                ("GDP per capita (current US$)",            "GDP/capita"),
                ("Current account balance (% of GDP)",      "Current Account %"),
                ("Inflation, consumer prices (annual %)",   "Inflation %"),
                ("Unemployment, total (% of total labor force) (modeled ILO estimate)", "Unemployment %"),
                ("Real interest rate (%)",                  "Real Rate %"),
                ("Expense (% of GDP)",                      "Expense % GDP"),
            ]
            _grp_lbl = _profile_p or "groupe"
            st.markdown(
                f'<div style="font-size:10px;color:#4A6890;margin-bottom:4px;">'
                f'▲ / ▼ vs group median <strong style="color:#6890B0;">{_grp_lbl}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for _ind, _lbl in _KEY_MAC:
                if _ind not in df_m.columns:
                    continue
                _val  = _r.get(_ind)
                _pmed = _peers_p[_ind].median() if _ind in _peers_p.columns else np.nan
                if pd.isna(_val):
                    st.markdown(
                        f'<div style="background:#162035;border:1px solid #2A3848;'
                        f'border-radius:7px;padding:.35rem .7rem;margin-bottom:3px;'
                        f'font-size:12px;color:#4A6890;">{_lbl} — n/a</div>',
                        unsafe_allow_html=True)
                    continue
                _ap = (_val > _pmed) if pd.notna(_pmed) else None
                _ap_c     = ("#10B981" if _ap else "#EF4444") if _ap is not None else "#6890B0"
                _ap_arrow = ("▲" if _ap else "▼") if _ap is not None else "—"
                _disp = (f"${_val:,.0f}" if "per capita" in _ind
                         else f"{_val:+.1f}%" if _val < 0 else f"{_val:.1f}%")
                st.markdown(
                    f'<div style="background:#162035;border:1px solid #304870;'
                    f'border-radius:7px;padding:.35rem .7rem;margin-bottom:3px;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:12px;color:#8AABB0;">{_lbl}</span>'
                    f'<span style="font-size:13px;font-weight:600;color:#EEF4FB;">'
                    f'{_disp}&nbsp;'
                    f'<span style="color:{_ap_c};font-size:12px;">{_ap_arrow}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Radar vs peers ────────────────────────────────────────
        thin_divider()
        section_label(f"Radar — {_country_p} vs groupe « {_profile_p} »")

        _RI = [GDP_pc, CA_PCT, INFL, UNEMP, RATE, EXP_PCT]
        _RS = {GDP_pc:"GDP/cap", CA_PCT:"Cur. Acct", INFL:"Inflation",
               UNEMP:"Unemployment", RATE:"Real Rate", EXP_PCT:"Exports/GDP"}
        _RV = {INFL, UNEMP, RATE}

        df_rp = df_m.copy()
        for _ind in _RI:
            if _ind not in df_rp.columns:
                continue
            _rnk = df_rp[_ind].rank(pct=True, na_option="keep")
            df_rp[f"__rp_{_ind}"] = (1 - _rnk) if _ind in _RV else _rnk

        _rp_cs = [f"__rp_{i}" for i in _RI if f"__rp_{i}" in df_rp.columns]
        _rp_ls = [_RS[i] for i in _RI if f"__rp_{i}" in df_rp.columns]

        _crow = df_rp[df_rp["country_name"] == _country_p][_rp_cs]
        _prow = df_rp[df_rp["combined_profile"] == _profile_p][_rp_cs] if _profile_p else df_rp[_rp_cs]

        if not _crow.empty:
            _cv = list(_crow.iloc[0].fillna(0.5).values)
            _pv = list(_prow.median().fillna(0.5).values)
            _lc = _rp_ls + [_rp_ls[0]]

            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatterpolar(
                r=_pv + [_pv[0]], theta=_lc,
                fill="toself", fillcolor="rgba(96,165,250,0.09)",
                line=dict(color="#60A5FA", width=1.5, dash="dot"),
                name=f"Median {_profile_p}",
                hovertemplate="<b>%{theta}</b><br>Percentile=%{r:.2f}<extra></extra>",
            ))
            fig_rp.add_trace(go.Scatterpolar(
                r=_cv + [_cv[0]], theta=_lc,
                fill="toself", fillcolor="rgba(16,185,129,0.18)",
                line=dict(color="#10B981", width=2.5),
                name=_country_p,
                hovertemplate="<b>%{theta}</b><br>Percentile=%{r:.2f}<extra></extra>",
            ))
            fig_rp.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis", "yaxis", "margin", "legend", "title_font")},
                title="",
                polar=dict(
                    bgcolor="#162035",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#304870",
                                    tickfont=dict(size=9, color="#6890B0")),
                    angularaxis=dict(gridcolor="#304870",
                                     tickfont=dict(size=12, color="#C4D5E8")),
                ),
                height=440,
                margin=dict(l=60, r=60, t=60, b=60),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_rp, use_container_width=True)

        # ── Narrative ─────────────────────────────────────────────
        thin_divider()
        section_label("Position relative à la tendance structurelle GDP/capita")

        st.markdown(
            '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #6890B0;'
            'border-radius:0 8px 8px 0;padding:.4rem .8rem;margin:.2rem 0 .7rem 0;font-size:12px;'
            'color:#6890B0;">'
            'Lecture descriptive — pas une prédiction. La droite OLS (GDP/capita → combined_score) '
            'montre où se situent en général les pays avec ce niveau de richesse. '
            'Le résidu dit si ce pays est au-dessus ou en-dessous de cette tendance structurelle, '
            'pas ce qu\'il devrait être.'
            '</div>',
            unsafe_allow_html=True,
        )

        _df_ols = df_m[[GDP_pc, "combined_score"]].dropna()
        _df_ols = _df_ols[_df_ols[GDP_pc] > 0]
        if len(_df_ols) > 20:
            _xlog  = np.log10(_df_ols[GDP_pc].values)
            _ycs   = _df_ols["combined_score"].values
            _poly  = np.polyfit(_xlog, _ycs, 1)
            _cg    = _r.get(GDP_pc)
            _ccs   = _r.get("combined_score")
            if pd.notna(_cg) and pd.notna(_ccs) and _cg > 0:
                _trend = np.polyval(_poly, np.log10(_cg))
                _resid = _ccs - _trend
                _rpct  = abs(_resid / _trend * 100) if _trend else 0
                _abv   = _resid > 0
                _rc    = "#10B981" if _abv else "#EF4444"
                _dir   = "au-dessus" if _abv else "en-dessous"
                st.markdown(
                    f'<div style="background:#1A2840;border:1px solid #304870;'
                    f'border-left:3px solid {_rc};border-radius:0 10px 10px 0;'
                    f'padding:.8rem 1.1rem;margin-bottom:.8rem;">'
                    f'<div style="font-size:11px;font-weight:700;color:{_rc};'
                    f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.4rem;">'
                    f'Résidu structurel</div>'
                    f'<div style="font-size:14px;color:#EEF4FB;line-height:1.9;">'
                    f'Score observé : <strong style="color:{_rc};">{_ccs:.2f}</strong> · '
                    f'Tendance pour ce GDP/capita : <strong>{_trend:.2f}</strong><br>'
                    f'<strong style="color:{_rc};">{_country_p} est {_dir} '
                    f'de la tendance structurelle</strong> '
                    f'(résidu = {_resid:+.2f}, soit {_rpct:.0f}%)<br>'
                    f'<span style="font-size:12px;color:#7090B0;">'
                    f'GDP/capita : ${_cg:,.0f} — '
                    f'ρ(GDP/cap → combined_score) = +0.33 [E]</span></div></div>',
                    unsafe_allow_html=True,
                )
                _ca  = _r.get(CA_PCT)
                _cam = df_m[CA_PCT].median()
                if pd.notna(_ca):
                    _cad = "excédentaire" if _ca > _cam else "déficitaire"
                    _cac = "#10B981" if _ca > _cam else "#EF4444"
                    _coherent = (_abv and _ca > _cam) or (not _abv and _ca < _cam)
                    finding(
                        f"<strong>CA% = {_ca:+.1f}%</strong> ({_cad} vs médiane {_cam:+.1f}%) "
                        f"— deuxième signal [E] dominant (η²=0.241–0.250 sections I/J). "
                        + ("Signal cohérent : excédent courant et surperformance trade."
                           if _coherent else
                           "Signal mixte : la balance courante contredit la position relative au PIB/habitant — "
                           "facteurs idiosyncratiques jouent un rôle.")
                    )

        # ── Scatter Explorer ──────────────────────────────────────
        thin_divider()
        section_label("Scatter Explorer — Explorer n'importe quelle paire")

        _TC = {k: v for k, v in {
            "combined_score":           "Combined Score",
            "structural_score":         "Structural Score",
            "export_share_structural":  "Export Share",
            "coverage_ratio_structural":"Coverage Ratio",
            "cagr_export":              "CAGR Export",
            "volatility_export":        "Volatilité Export",
            "balance_dep":              "Balance/GDP",
            "export_pc":                "Export/Capita",
        }.items() if k in df_m.columns}

        _MC = {k: v for k, v in {
            GDP_pc:  "GDP/Capita",
            CA_PCT:  "CA % GDP",
            INFL:    "Inflation %",
            UNEMP:   "Unemployment %",
            RATE:    "Real Rate %",
            EXPENSE: "Expense % GDP",
        }.items() if k in df_m.columns}

        _c1, _c2 = st.columns(2)
        _stv = _c1.selectbox("Variable trade (Y)", list(_TC.keys()),
                              format_func=lambda x: _TC.get(x, x), key="p_tv")
        _smi = _c2.selectbox("Indicateur macro (X)", list(_MC.keys()),
                              format_func=lambda x: _MC.get(x, x), key="p_mi")

        _USD_COLS = {GDP_pc, "export_pc", "import_pc"}
        _dsc = df_m[["country_name", _stv, _smi, "combined_profile"]].dropna().copy()
        if len(_dsc) >= 10:
            _rho_s, _pv_s = spearmanr(_dsc[_stv], _dsc[_smi])
            _sig_s = _pv_s < 0.05
            _fx = lambda v: fmt_usd(v) if _smi in _USD_COLS else f"{v:.3f}"
            _fy = lambda v: fmt_usd(v) if _stv in _USD_COLS else f"{v:.3f}"
            _dsc["_xf"] = _dsc[_smi].apply(_fx)
            _dsc["_yf"] = _dsc[_stv].apply(_fy)
            _issel = _dsc["country_name"] == _country_p
            _oth   = _dsc[~_issel]
            _sel   = _dsc[_issel]
            _COMB_C2 = {"Global Champion":"#10B981","Competitive":"#60A5FA",
                        "Vulnerable":"#F59E0B","Fragile":"#EF4444"}
            _ptc = [_COMB_C2.get(str(p), "#304870") for p in _oth["combined_profile"]]
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=_oth[_smi], y=_oth[_stv], mode="markers",
                customdata=_oth[["_xf", "_yf"]].values,
                marker=dict(color=_ptc, size=6, opacity=0.65),
                text=_oth["country_name"],
                hovertemplate="<b>%{text}</b><br>X=%{customdata[0]} · Y=%{customdata[1]}<extra></extra>",
                name="Autres pays",
            ))
            if not _sel.empty:
                fig_sc.add_trace(go.Scatter(
                    x=_sel[_smi], y=_sel[_stv], mode="markers+text",
                    customdata=_sel[["_xf", "_yf"]].values,
                    marker=dict(color="#FFD700", size=14, symbol="star",
                                line=dict(color="#FFFFFF", width=2)),
                    text=[_country_p], textposition="top center",
                    textfont=dict(size=12, color="#FFFFFF"),
                    hovertemplate=f"<b>{_country_p}</b><br>X=%{{customdata[0]}} · Y=%{{customdata[1]}}<extra></extra>",
                    name=_country_p,
                ))
            _xs2 = _dsc[_smi].values
            _ys2 = _dsc[_stv].values
            _p2  = np.polyfit(_xs2, _ys2, 1)
            _xl2 = np.linspace(_xs2.min(), _xs2.max(), 100)
            fig_sc.add_trace(go.Scatter(
                x=_xl2, y=np.polyval(_p2, _xl2), mode="lines", name="OLS",
                line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
            ))
            _fsc = df_sp[df_sp["indicator"] == _smi]["ind_flag"].values
            _fsc_v = _fsc[0] if len(_fsc) > 0 else "—"
            _fsc_c = {"[E]":"#10B981","[T]":"#EF4444","[S]":"#F59E0B"}.get(_fsc_v, "#60A5FA")
            fig_sc.add_annotation(
                x=0.98, y=0.98, xref="paper", yref="paper",
                text=(f"ρ={_rho_s:+.3f} {'✓' if _sig_s else '✗ n.s.'} · "
                      f"<span style='color:{_fsc_c}'>{_fsc_v}</span>  n={len(_dsc)}"),
                showarrow=False, xanchor="right", yanchor="top",
                bgcolor="#162035", bordercolor="#304870",
                font=dict(size=12, color="#C4D5E8"),
            )
            _pl(fig_sc,
                title=f"{_MC.get(_smi, _smi)} × {_TC.get(_stv, _stv)}",
                xaxis_title=_MC.get(_smi, _smi),
                yaxis_title=_TC.get(_stv, _stv),
                height=460,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            with st.expander("Comment lire ce graphique"):
                st.markdown("""
                <div style='font-size:13px;color:#A8C0D8;line-height:1.9;'>
                <strong style='color:#EEF4FB;'>ρ (Spearman) et le ✓ / ✗</strong><br>
                ρ mesure la force de la relation monotone sur l'ensemble des ~170 pays.
                ✓ (p&lt;0.05) = la tendance globale est statistiquement réelle.
                ✗ = même la tendance générale est du bruit — il n'y a pas de signal entre ces deux variables.<br>
                <em>Attention : ✓ ne veut pas dire que la position d'un pays individuel est prédictible —
                le CV R² reste quasi-nul (Section M).</em><br><br>
                <strong style='color:#EEF4FB;'>Droite OLS (bleue pointillée)</strong><br>
                Approximation linéaire de la tendance centrale. Elle n'est pas un modèle de prédiction —
                c'est une référence visuelle pour voir l'écart d'un pays.<br><br>
                <strong style='color:#EEF4FB;'>Position du pays (étoile dorée) par rapport à la droite</strong><br>
                <span style='color:#10B981;'>Au-dessus</span> = ce pays performe mieux sur Y que ce que sa valeur X laisse attendre
                (résidu positif — avantage idiosyncratique).<br>
                <span style='color:#EF4444;'>En-dessous</span> = ce pays performe moins bien que la tendance
                (résidu négatif — facteurs limitants non capturés par X).<br><br>
                <strong style='color:#EEF4FB;'>Couleur des points</strong><br>
                Profil trade combiné — permet de voir si des clusters de profils
                se regroupent ou se dispersent sur la paire choisie.
                </div>
                """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB SYNTHESE
# ────────────────────────────────────────────────────────────────
with tab_syn:
    section_label("Synthèse — Ce que l'analyse révèle")

    # ── Contexte & données ────────────────────────────────────
    st.markdown(
        '<div style="background:#162035;border:1px solid #304870;border-radius:8px;'
        'padding:.55rem 1rem;margin-bottom:.9rem;display:flex;flex-wrap:wrap;gap:1.4rem;'
        'align-items:center;">'
        '<span style="font-size:11px;font-weight:700;color:#4A6890;text-transform:uppercase;'
        'letter-spacing:.07em;">Périmètre</span>'
        '<span style="font-size:13px;color:#A8C0D8;">~<strong style="color:#EEF4FB;">170</strong> pays</span>'
        '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">16</strong> indicateurs macro World Bank</span>'
        '<span style="font-size:13px;color:#A8C0D8;">Médiane <strong style="color:#EEF4FB;">2015–2024</strong></span>'
        '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">2</strong> directions d\'analyse</span>'
        '<span style="font-size:12px;color:#F59E0B;font-style:italic;font-weight:600;">'
        '⚠ conclusions valables sur ce périmètre uniquement</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Schéma des deux directions ────────────────────────────
    _cell = ("background:#162035;border:1px solid #304870;border-radius:10px;"
             "padding:.8rem 1rem;")
    _kpi_n = ("font-size:22px;font-weight:800;line-height:1;")
    _kpi_l = ("font-size:10px;color:#6890B0;margin-top:.15rem;")
    _arrow = ('font-size:28px;font-weight:300;color:#304870;text-align:center;'
              'display:flex;align-items:center;justify-content:center;')

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 56px 1fr;gap:.5rem;'
        f'margin-bottom:1rem;">'

        # Direction 1 box
        f'<div style="{_cell}border-top:3px solid #10B981;">'
        f'<div style="font-size:11px;font-weight:700;color:#10B981;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">Direction 1 · G → M</div>'
        f'<div style="font-size:12px;color:#7090B0;margin-bottom:.6rem;">'
        f'Fondamentaux macro → Structure trade</div>'
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
        f'<div><div style="{_kpi_n}color:#10B981;">ρ=0.94</div>'
        f'<div style="{_kpi_l}">GDP → Export Share</div></div>'
        f'<div><div style="{_kpi_n}color:#10B981;">ρ=0.82</div>'
        f'<div style="{_kpi_l}">GDP/cap → Export/cap</div></div>'
        f'<div><div style="{_kpi_n}color:#F59E0B;">CV≈0.05</div>'
        f'<div style="{_kpi_l}">RF hors-échantillon</div></div>'
        f'</div>'
        f'<div style="font-size:11px;color:#4A6890;margin-top:.5rem;">'
        f'Signal réel [E] · prédiction hors-échantillon échoue</div>'
        f'</div>'

        # Arrow
        f'<div style="{_arrow}">⇄</div>'

        # Direction 2 box
        f'<div style="{_cell}border-top:3px solid #A78BFA;">'
        f'<div style="font-size:11px;font-weight:700;color:#A78BFA;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">Direction 2 · N</div>'
        f'<div style="font-size:12px;color:#7090B0;margin-bottom:.6rem;">'
        f'Structure trade → Fondamentaux macro</div>'
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
        f'<div><div style="{_kpi_n}color:#A78BFA;">ρ=0.71</div>'
        f'<div style="{_kpi_l}">Export/cap → GDP/cap</div></div>'
        f'<div><div style="{_kpi_n}color:#EF4444;">CV≈−0.5</div>'
        f'<div style="{_kpi_l}">RF Inflation (hors-éch.)</div></div>'
        f'<div><div style="{_kpi_n}color:#EF4444;">CV≈0.03</div>'
        f'<div style="{_kpi_l}">RF GDP/cap (hors-éch.)</div></div>'
        f'</div>'
        f'<div style="font-size:11px;color:#4A6890;margin-top:.5rem;">'
        f'Corrélations symétriques · prédiction encore plus fragile</div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Conclusion principale ─────────────────────────────────
    st.markdown("""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:14px;
                padding:1.2rem 1.4rem;margin-bottom:1rem;">
    <div style="font-size:12px;font-weight:700;color:#34D399;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.8rem;">Conclusion principale</div>
    <div style="font-size:15px;color:#EEF4FB;line-height:1.9;">
    Sur ce périmètre d'analyse, la relation entre macro et trade <strong>se révèle
    structurelle et descriptive</strong> — pas prédictive. Les fondamentaux macro
    permettent d'expliquer <em>où</em> un pays se positionne dans le trade
    (niveau, profil), mais pas <em>comment</em> il va performer out-of-sample.
    </div>
    <div style="font-size:12px;color:#4A6890;margin-top:.7rem;font-style:italic;
                border-top:1px solid #304870;padding-top:.6rem;">
    Résultat conditionnel : 16 indicateurs macro WB généraux · ~170 pays ·
    médiane 2015–2024 · pas d'indicateurs sectoriels ni institutionnels.
    Des données plus riches pourraient révéler une dimension prédictive.
    </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        section_label("Ce qui a été trouvé")
        finding(
            "<strong>Taille prime sur la richesse</strong> (Section G–H) : GDP absolu "
            "explique mieux l'export share (ρ=0.94) que GDP_pc explique l'export/capita "
            "(ρ=0.82). Être grand détermine le volume ; être riche détermine l'efficacité."
        )
        finding(
            "<strong>Hiérarchie [E] observée</strong> (Section K) : sur ces données, "
            "GDP/capita > CA% > Real Rate > Inflation > Unemployment. "
            "Ces indicateurs montrent un signal faible à modéré sur la structure trade — "
            "hiérarchie à confirmer avec d'autres sources ou périmètres."
        )
        finding(
            "<strong>Empreinte macro par profil</strong> (Section L) : sur les 6 indicateurs "
            "analysés, GDP/capita et CA% ressortent comme les dimensions les plus "
            "différenciatrices entre profils trade. Les 4 autres indicateurs (inflation, "
            "chômage, taux réel, exports/GDP) montrent des différences faibles sur cette "
            "fenêtre et ce périmètre de données."
        )
        finding(
            "<strong>Trade Intensity révèle l'ouverture</strong> (Section D) : la richesse "
            "incline vers l'ouverture mais avec variance élevée. Les petits pays riches "
            "(Singapour, Luxembourg) sont les plus ouverts en relatif."
        )

    with col_s2:
        section_label("Ce qui n'a pas été trouvé")
        finding(
            "<strong>Prédiction out-of-sample impossible</strong> (Section M) : RF R² élevé "
            "(0.72–0.91) mais CV R² quasi-nul (0.05–0.11). Le modèle mémorise, "
            "ne généralise pas. Les dynamiques (CAGR, volatilité) sont encore moins "
            "prédictibles (CV R² < 0)."
        )
        finding(
            "<strong>Direction 2 non robuste</strong> (Section N) : trade → macro "
            "ne prédit pas les outcomes macro hors-échantillon (CV R² ≈ 0 ou négatif). "
            "Sur ces données, la direction macro → trade semble plus informative, "
            "mais les corrélations Spearman étant symétriques, la causalité "
            "ne peut pas être établie depuis cette seule analyse."
        )
        finding(
            "<strong>Dynamiques non capturées</strong> (Section I–J) : sur ce dataset, "
            "la croissance et la volatilité trade ne sont pas expliquées par les 16 "
            "indicateurs macro disponibles (CV R² &lt; 0). Ces dynamiques impliquent "
            "probablement des facteurs non mesurés ici : politique sectorielle, "
            "géographie, chocs exogènes, accords bilatéraux."
        )

    thin_divider()
    section_label("Orientations pour aller plus loin")

    st.markdown("""
    <div style="font-size:13px;color:#A8C0D8;line-height:2.1;">
    <div>→ <strong style="color:#EEF4FB;">Données panel</strong> : exploiter la dimension
    temporelle (pays × année) pour capturer les effets de politique commerciale</div>
    <div>→ <strong style="color:#EEF4FB;">Indicateurs sectoriels</strong> : diversification
    produit, complexité économique (ECI), dépendances géographiques</div>
    <div>→ <strong style="color:#EEF4FB;">Variables institutionnelles</strong> : qualité
    institutionnelle, accords commerciaux, barrières tarifaires</div>
    <div>→ <strong style="color:#EEF4FB;">Modèles non-linéaires validés</strong> : avec plus
    de pays ou données synthétiques pour pallier n=~170</div>
    <div>→ <strong style="color:#EEF4FB;">Chocs exogènes</strong> : COVID, matières premières —
    identifier les pays qui résistent vs amplifient les chocs macro</div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()
