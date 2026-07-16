"""
Trade Analysis — T6 · Typology & Final Scoring
H.4  Combined scoring (F × G)
H.5  Dynamic Country Profile Card
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding

inject_styles()

TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_combined  = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_final     = pd.read_csv(TRADE_FILES / "df_final.csv")
    df_fp        = pd.read_csv(TRADE_FILES / "df_full_periode.csv")
    df_structural = pd.read_csv(TRADE_FILES / "df_structural.csv")
    _extra = df_fp[["country_name", "stability_label", "drift_magnitude",
                     "profile_recent", "score_recent"]]
    df_combined = df_combined.merge(_extra, on="country_name", how="left")
    _sq = df_structural[["country_name", "static_quadrant",
                          "dynamic_quadrant", "strategic_quadrant"]]
    df_combined = df_combined.merge(_sq, on="country_name", how="left")
    return df_combined, df_final

df_combined, df_final = load_data()

COUNTRIES = sorted(df_combined["country_name"].dropna().unique().tolist())
N_COUNTRIES = len(df_combined)

# ════════════════════════════════════════════════════════════════
# THEME & COLORS
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

def _lay(**kw):
    return {**PLOTLY_LAYOUT, **kw}


COMBINED_COLORS = {
    "Global Champion": "#10B981",
    "Competitive":     "#60A5FA",
    "Vulnerable":      "#F59E0B",
    "Fragile":         "#EF4444",
}

PROFILE_COLORS = {
    "Robust":                  "#10B981",
    "Stable but Concentrated": "#A78BFA",
    "Vulnerable":              "#F59E0B",
    "Critical Dependency":     "#EF4444",
}

STRUCTURAL_COLORS = {
    "Strong":   "#10B981",
    "Good":     "#60A5FA",
    "Moderate": "#F59E0B",
    "Weak":     "#EF4444",
}

STRATEGIC_COLORS = {
    "Strong & Resilient":  "#10B981",
    "Strong but Exposed":  "#F59E0B",
    "Diversified Minor":   "#60A5FA",
    "Weak & Vulnerable":   "#EF4444",
}

STATIC_Q_COLORS = {
    "Dominant":  "#10B981",
    "Balanced":  "#60A5FA",
    "Vulnerable": "#F59E0B",
    "Marginal":  "#EF4444",
}
DYNAMIC_Q_COLORS = {
    "Star":            "#10B981",
    "Risky Grower":    "#F59E0B",
    "Stable Stagnant": "#60A5FA",
    "Declining":       "#EF4444",
}
STRATEGIC_F_COLORS = {
    "Star":     "#10B981",
    "Emerging": "#60A5FA",
    "Marginal": "#EF4444",
}

STABILITY_COLORS = {
    "Very Stable": "#10B981",
    "Stable":      "#A78BFA",
    "Moderate":    "#F59E0B",
    "Unstable":    "#EF4444",
}

RADAR_DIMS   = ["size_profile", "balance_profile", "growth_profile",
                "stability_profile", "consistency_profile"]
RADAR_LABELS = ["Size", "Balance", "Growth", "Stability", "Consistency"]

RADAR_SCALE = {
    "size_profile":        {"Leader": 1.0, "Major": 0.67, "Mid": 0.33, "Small": 0.0},
    "balance_profile":     {"Strong Surplus": 1.0, "Slight Surplus": 0.67,
                            "Slight Deficit": 0.33, "Structural Deficit": 0.0},
    "growth_profile":      {"Strong Grower": 1.0, "Moderate Grower": 0.67,
                            "Weak Grower": 0.33, "Declining": 0.0},
    "stability_profile":   {"Stable": 1.0, "Moderate": 0.67,
                            "Volatile": 0.33, "Very Volatile": 0.0},
    "consistency_profile": {"Very Consistent": 1.0, "Consistent": 0.67,
                            "Inconsistent": 0.33, "Very Inconsistent": 0.0},
}

_PROF_ORDER = ["Robust", "Stable but Concentrated", "Vulnerable", "Critical Dependency"]
_PROF_NUM   = {p: i for i, p in enumerate(_PROF_ORDER)}
_PROF_CLR   = ["#10B981", "#A78BFA", "#F59E0B", "#EF4444"]
_TL_COLORSCALE = [
    [0.00, "#10B981"], [0.249, "#10B981"],
    [0.25, "#A78BFA"], [0.499, "#A78BFA"],
    [0.50, "#F59E0B"], [0.749, "#F59E0B"],
    [0.75, "#EF4444"], [1.000, "#EF4444"],
]

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Trade Analysis · Page 6 of 6</div>',
            unsafe_allow_html=True)
st.title("Typology & Final Scoring")
st.caption("Combined F×G scoring · Strategic quadrant · Country Profile Card")
thin_divider()

# ── Global KPI cards ─────────────────────────────────────────
n_champ = (df_combined["combined_profile"] == "Global Champion").sum()
n_comp  = (df_combined["combined_profile"] == "Competitive").sum()
n_vuln  = (df_combined["combined_profile"] == "Vulnerable").sum()
n_frag  = (df_combined["combined_profile"] == "Fragile").sum()

_kc = st.columns(4)
_kc[0].markdown(kpi("Global Champions", str(n_champ), "teal"),  unsafe_allow_html=True)
_kc[1].markdown(kpi("Competitive",      str(n_comp),  "blue"),  unsafe_allow_html=True)
_kc[2].markdown(kpi("Vulnerable",       str(n_vuln),  "amber"), unsafe_allow_html=True)
_kc[3].markdown(kpi("Fragile",          str(n_frag),  ""),      unsafe_allow_html=True)

thin_divider()

# ── Global country selector ───────────────────────────────────
_sel_col, _ = st.columns([1, 3])
with _sel_col:
    _default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox("Select country", COUNTRIES,
                           index=_default_idx, key="t6_country")

_row = df_combined[df_combined["country_name"] == country]
if _row.empty:
    st.warning(f"No data for {country}.")
    st.stop()
_row = _row.iloc[0]

# Derived colors for selected country
_cprof   = _row.get("combined_profile", "—")
_cc      = COMBINED_COLORS.get(_cprof, "#C4D5E8")
_sprof   = _row.get("structural_profile", "—")
_sc      = STRUCTURAL_COLORS.get(_sprof, "#C4D5E8")
_bprof   = _row.get("profile_full_periode", "—")
_bc      = PROFILE_COLORS.get(_bprof, "#C4D5E8")
_strat_q = _row.get("strategic_quadrant_combined", "—")
_sq_c    = STRATEGIC_COLORS.get(_strat_q, "#C4D5E8")
_stab    = _row.get("stability_label", "—")
_stab_c  = STABILITY_COLORS.get(_stab, "#C4D5E8")
_drift   = _row.get("drift", "—")
_drift_c = {"Worsening": "#EF4444", "Improving": "#10B981",
            "Stable": "#6B7280"}.get(_drift, "#C4D5E8")

# Country rank by combined score
_rank_series = df_combined.sort_values("combined_score", ascending=False).reset_index(drop=True)
_rank_idx    = _rank_series[_rank_series["country_name"] == country].index
_rank_pos    = int(_rank_idx[0]) + 1 if len(_rank_idx) else "—"

thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Combined Scoring & Profiles",
    "🎯 Quadrant & Stress Test",
    "🏅 Country Profile Card",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — COMBINED OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:

    # ── 1. Target country card ────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1A2C48,#1F3050);
            border:1.5px solid {_cc};border-radius:14px;
            padding:1rem 1.5rem;margin-bottom:1rem;">
  <div style="display:grid;grid-template-columns:auto 1fr 1fr 1fr 1fr 1fr;
              align-items:center;gap:1.5rem;">
    <div style="border-right:1px solid #304870;padding-right:1.5rem;">
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:4px;">Target Country</div>
      <div style="font-size:22px;font-weight:700;color:#EEF4FB;white-space:nowrap;">
        {country}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">Combined Score</div>
      <div style="font-size:26px;font-weight:700;color:{_cc};line-height:1.1;">
        {_row['combined_score']:.3f}</div>
      <div style="font-size:11px;color:#6890B0;margin-top:2px;">Rank #{_rank_pos} / {N_COUNTRIES}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">Combined Profile</div>
      <div style="font-size:14px;font-weight:600;color:{_cc};margin-top:6px;">{_cprof}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">Strategic Quadrant</div>
      <div style="font-size:14px;font-weight:600;color:{_sq_c};margin-top:6px;">{_strat_q}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">F Rank</div>
      <div style="font-size:20px;font-weight:700;color:#60A5FA;margin-top:4px;">
        {_row['f_norm_rank']:.3f}</div>
      <div style="font-size:10px;color:#6890B0;margin-top:2px;">{_sprof}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">G Rank</div>
      <div style="font-size:20px;font-weight:700;color:#A78BFA;margin-top:4px;">
        {_row['g_norm_rank']:.3f}</div>
      <div style="font-size:10px;color:#6890B0;margin-top:2px;">{_bprof}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 2. Definition block ───────────────────────────────────
    _d1, _d2 = st.columns(2)

    with _d1:
        st.markdown("""
<div style="background:#162035;border:1px solid #304870;border-radius:10px;padding:.9rem 1rem;">
  <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.6rem;">Combined Score</div>
  <div style="font-size:13px;color:#C4D5E8;margin-bottom:.6rem;">
    <code style="background:#1F3050;padding:2px 6px;border-radius:4px;color:#A78BFA;">
      0.70 × F_norm_rank &nbsp;+&nbsp; 0.30 × G_norm_rank</code>
  </div>
  <div style="font-size:12px;color:#6890B0;margin-bottom:.5rem;">
    F = Trade Structure score (size, growth, balance, coverage)
    &nbsp;·&nbsp; G = Bilateral Risk score (HHI, BDIS, critical share)
  </div>
  <table style="width:100%;font-size:12px;border-collapse:collapse;">
    <tr style="color:#6890B0;border-bottom:1px solid #304870;">
      <th style="text-align:left;padding:3px 6px;">Threshold</th>
      <th style="text-align:left;padding:3px 6px;">Combined Profile</th>
    </tr>
    <tr><td style="padding:3px 6px;color:#6890B0;">≥ 0.80</td>
        <td style="padding:3px 6px;font-weight:600;color:#10B981;">Global Champion</td></tr>
    <tr><td style="padding:3px 6px;color:#6890B0;">≥ 0.55</td>
        <td style="padding:3px 6px;font-weight:600;color:#60A5FA;">Competitive</td></tr>
    <tr><td style="padding:3px 6px;color:#6890B0;">≥ 0.30</td>
        <td style="padding:3px 6px;font-weight:600;color:#F59E0B;">Vulnerable</td></tr>
    <tr><td style="padding:3px 6px;color:#6890B0;">&lt; 0.30</td>
        <td style="padding:3px 6px;font-weight:600;color:#EF4444;">Fragile</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    with _d2:
        st.markdown("""
<div style="background:#162035;border:1px solid #304870;border-radius:10px;padding:.9rem 1rem;">
  <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.6rem;">Strategic Quadrant</div>
  <div style="font-size:12px;color:#6890B0;margin-bottom:.6rem;">
    Position in F (trade power) × G (resilience) space — threshold at 0.55
  </div>
  <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:center;">
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;"></td>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:center;">
        G Low<br><span style="color:#EF4444;font-size:10px;">More Risk</span></td>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:center;">
        G High<br><span style="color:#10B981;font-size:10px;">Resilient</span></td>
    </tr>
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:right;">
        F High<br><span style="color:#60A5FA;font-size:10px;">More Power</span></td>
      <td style="padding:6px;background:#231800;border-radius:6px;
                 font-weight:600;color:#F59E0B;text-align:center;">Strong but Exposed</td>
      <td style="padding:6px;background:#082E18;border-radius:6px;
                 font-weight:600;color:#10B981;text-align:center;">Strong & Resilient</td>
    </tr>
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:right;">
        F Low<br><span style="color:#6B7280;font-size:10px;">Less Power</span></td>
      <td style="padding:6px;background:#2D0E0E;border-radius:6px;
                 font-weight:600;color:#EF4444;text-align:center;">Weak & Vulnerable</td>
      <td style="padding:6px;background:#0F2240;border-radius:6px;
                 font-weight:600;color:#60A5FA;text-align:center;">Diversified Minor</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

    thin_divider()

    # ── 3. F × G Cross-table ──────────────────────────────────
    section_label("F × G · Typology Matrix")

    _cross_l, _cross_r = st.columns(2)

    with _cross_l:
        df_cross = pd.crosstab(
            df_combined["structural_profile"],
            df_combined["profile_full_periode"]
        )
        _row_ord = ["Strong", "Good", "Moderate", "Weak"]
        _col_ord = ["Robust", "Stable but Concentrated", "Vulnerable", "Critical Dependency"]
        df_cross = df_cross.reindex(
            index=[r for r in _row_ord if r in df_cross.index],
            columns=[c for c in _col_ord if c in df_cross.columns],
            fill_value=0,
        )
        fig_cross = go.Figure(go.Heatmap(
            z=df_cross.values,
            x=df_cross.columns.tolist(),
            y=df_cross.index.tolist(),
            colorscale=[[0, "#162035"], [0.5, "#3B82F6"], [1, "#10B981"]],
            text=df_cross.values,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#EEF4FB"),
            showscale=False,
        ))
        fig_cross.update_layout(**_lay(
            title="Structural Profile (F) × Bilateral Profile (G)",
            xaxis_title="Bilateral Profile (G)",
            yaxis_title="Structural Profile (F)",
            margin=dict(l=80, r=20, t=50, b=110),
            height=340,
        ))
        st.plotly_chart(fig_cross, width='stretch')

    with _cross_r:
        # Combined profile distribution
        _comb_cnt = df_combined["combined_profile"].value_counts().reset_index()
        _comb_cnt.columns = ["Profile", "Count"]
        _comb_cnt["_o"] = _comb_cnt["Profile"].map(
            {p: i for i, p in enumerate(COMBINED_COLORS)}).fillna(99)
        _comb_cnt = _comb_cnt.sort_values("_o")

        fig_comb = go.Figure(go.Bar(
            x=_comb_cnt["Count"], y=_comb_cnt["Profile"],
            orientation="h",
            marker_color=[COMBINED_COLORS.get(p, "#6B7280") for p in _comb_cnt["Profile"]],
            text=_comb_cnt["Count"], textposition="outside",
            opacity=0.85,
        ))
        fig_comb.update_layout(**_lay(
            title="Combined Profile Distribution",
            xaxis_title="Countries",
            height=165,
            margin=dict(l=10, r=50, t=50, b=30),
        ))
        st.plotly_chart(fig_comb, width='stretch')

        # Strategic quadrant distribution
        _strat_cnt = df_combined["strategic_quadrant_combined"].value_counts().reset_index()
        _strat_cnt.columns = ["Quadrant", "Count"]

        fig_strat = go.Figure(go.Bar(
            x=_strat_cnt["Count"], y=_strat_cnt["Quadrant"],
            orientation="h",
            marker_color=[STRATEGIC_COLORS.get(q, "#6B7280") for q in _strat_cnt["Quadrant"]],
            text=_strat_cnt["Count"], textposition="outside",
            opacity=0.85,
        ))
        fig_strat.update_layout(**_lay(
            title="Strategic Quadrant Distribution",
            xaxis_title="Countries",
            height=175,
            margin=dict(l=10, r=50, t=50, b=30),
        ))
        st.plotly_chart(fig_strat, width='stretch')

    thin_divider()

    # ── 4. Ranking bar ────────────────────────────────────────
    section_label("Combined Score — Ranking")

    _rk_col, _rk_bar = st.columns([1, 3])
    with _rk_col:
        n_top = st.selectbox("Show top N", [5, 10, 20, 30, 50], index=1, key="t1_n")
        view  = st.radio("View", ["Top performers", "Bottom performers"], key="t1_view")

    _df_rank = (df_combined.nlargest(n_top, "combined_score") if "Top" in view
                else df_combined.nsmallest(n_top, "combined_score"))
    _df_rank = _df_rank.sort_values("combined_score")

    with _rk_bar:
        fig_rank = go.Figure(go.Bar(
            x=_df_rank["combined_score"],
            y=_df_rank["country_name"],
            orientation="h",
            marker_color=[
                COMBINED_COLORS.get(p, "#6B7280")
                if cn != country
                else _cc
                for cn, p in zip(_df_rank["country_name"], _df_rank["combined_profile"])
            ],
            marker_line_color=[
                "#FFFFFF" if cn == country else "rgba(0,0,0,0)"
                for cn in _df_rank["country_name"]
            ],
            marker_line_width=[2 if cn == country else 0 for cn in _df_rank["country_name"]],
            text=_df_rank["combined_score"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
            customdata=_df_rank[["combined_profile", "f_norm_rank",
                                  "g_norm_rank", "strategic_quadrant_combined"]],
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<br>"
                          "Profile: %{customdata[0]}<br>"
                          "F: %{customdata[1]:.3f} · G: %{customdata[2]:.3f}<br>"
                          "Quadrant: %{customdata[3]}<extra></extra>",
            opacity=0.85,
        ))
        fig_rank.update_layout(**PLOTLY_LAYOUT,
            title=f"Combined Score — {view}  (0.70 × F + 0.30 × G)",
            xaxis_title="Combined Score",
            height=max(380, n_top * 22 + 80),
        )
        st.plotly_chart(fig_rank, width='stretch')

    thin_divider()

    # ── 5. Summary table — all countries ─────────────────────
    section_label("All Countries — Combined Scoring Summary")

    _sc_tog = st.pills(
        "Add columns", ["F & G Detail"],
        selection_mode="multi", default=[], key="t1_sc_tog",
    )
    _show_sc = "F & G Detail" in (_sc_tog or [])

    # Base: combined view only (4 cols)
    # Toggle ON: interleaved F→combined→G decomposition
    if not _show_sc:
        _all_sel = [
            ("country_name",               "Country"),
            ("combined_score",             "Combined Score"),
            ("combined_profile",           "Combined Profile"),
            ("strategic_quadrant_combined","Strategic Quadrant"),
        ]
    else:
        _all_sel = [
            ("country_name",               "Country"),
            ("combined_score",             "Combined Score"),
            ("combined_profile",           "Combined Profile"),
            ("strategic_quadrant_combined","Strategic Quadrant"),
            ("structural_profile",         "Structural Profile (F)"),
            ("f_norm_rank",                "F Score"),
            ("profile_full_periode",       "Profile Full Period (G)"),
            ("g_norm_rank",                "G Score"),
        ]
    _all_sel = [(s, l) for s, l in _all_sel if s in df_combined.columns]

    _df_tbl = (df_combined[[s for s, _ in _all_sel]]
               .copy()
               .sort_values("combined_score", ascending=False)
               .reset_index(drop=True))
    _df_tbl.columns = [l for _, l in _all_sel]

    # Styling
    def _clr_combined(val):
        c = COMBINED_COLORS.get(val, "")
        return f"color:{c};font-weight:600;" if c else ""

    def _clr_quadrant(val):
        c = STRATEGIC_COLORS.get(val, "")
        return f"color:{c};font-weight:600;" if c else ""

    def _clr_fp(val):
        c = PROFILE_COLORS.get(val, "")
        return f"color:{c};font-weight:600;" if c else ""

    def _clr_struct(val):
        c = STRUCTURAL_COLORS.get(val, "")
        return f"color:{c};font-weight:600;" if c else ""

    def _highlight_target(row):
        if row["Country"] == country:
            return ["background-color:rgba(16,185,129,0.10);"] * len(row)
        return [""] * len(row)

    _st = _df_tbl.style
    _cols = _df_tbl.columns.tolist()

    _st = _st.apply(_highlight_target, axis=1)
    for _col, _fn in [("Combined Profile",        _clr_combined),
                       ("Strategic Quadrant",       _clr_quadrant),
                       ("Profile Full Period (G)",  _clr_fp),
                       ("Structural Profile (F)",   _clr_struct)]:
        if _col in _cols:
            _st = _st.map(_fn, subset=[_col])

    _num_fmt = {c: "{:.3f}" for c in ["Combined Score", "F Score", "G Score"]
                if c in _cols}
    if _num_fmt:
        _st = _st.format(_num_fmt)

    _COL_W = {
        "Country": 90, "Combined Score": 90, "Combined Profile": 110,
        "Strategic Quadrant": 120, "Structural Profile (F)": 100,
        "Profile Full Period (G)": 125, "F Score": 80, "G Score": 80,
    }
    _col_cfg = {c: st.column_config.Column(width=_COL_W[c]) for c in _cols if c in _COL_W}

    st.dataframe(_st, width='stretch', column_config=_col_cfg, height=500)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — QUADRANT & STRESS TEST
# ══════════════════════════════════════════════════════════════════
with tab2:

    # ── Highlight selector ────────────────────────────────────
    _DEFAULT_HL = [country] + [c for c in [
        "USA", "China", "Germany", "Japan", "Canada",
        "Brazil", "Nigeria", "Qatar", "France", "Egypt",
    ] if c in COUNTRIES and c != country]

    # Reset highlight list whenever the target country changes
    if st.session_state.get("_t2_prev_country") != country:
        st.session_state["t2_hl"] = _DEFAULT_HL
        st.session_state["_t2_prev_country"] = country

    _hl = st.multiselect(
        "Highlight countries",
        options=COUNTRIES,
        default=_DEFAULT_HL,
        key="t2_hl",
    )

    _df_q = df_combined[["country_name", "f_norm_rank", "g_norm_rank",
                          "combined_score", "combined_profile",
                          "structural_score", "score_full_periode",
                          "strategic_quadrant_combined"]].dropna().copy()
    _df_q["_sz"]  = (_df_q["combined_score"] / _df_q["combined_score"].max() * 25 + 6).clip(6, 31)
    _df_q["_lbl"] = _df_q["country_name"].apply(lambda x: x if x in _hl else "")

    # ── Strategic Quadrant (colored by quadrant) ─────────────
    section_label("Strategic Quadrant — F × G Ranks")

    fig_q = go.Figure()
    for _quad, _color in STRATEGIC_COLORS.items():
        _dp = _df_q[_df_q["strategic_quadrant_combined"] == _quad]
        if _dp.empty:
            continue
        fig_q.add_trace(go.Scatter(
            x=_dp["f_norm_rank"], y=_dp["g_norm_rank"],
            mode="markers+text", name=_quad,
            marker=dict(color=_color, size=_dp["_sz"], opacity=0.7,
                        line=dict(width=0.5, color="#304870")),
            text=_dp["_lbl"], textposition="top center",
            textfont=dict(size=10, color="#EEF4FB"),
            customdata=_dp[["country_name", "combined_score", "combined_profile"]],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "F: %{x:.3f} · G: %{y:.3f}<br>"
                          "Score: %{customdata[1]:.3f}<br>"
                          "Profile: %{customdata[2]}<extra></extra>",
        ))

    # Threshold = 0.5 (midpoint of normalized 0-1 rank scale)
    _t = 0.5
    fig_q.add_vline(x=_t, line_dash="dash", line_color="#64748B", line_width=1.2)
    fig_q.add_hline(y=_t, line_dash="dash", line_color="#64748B", line_width=1.2)

    for _xa, _ya, _txt, _clr in [
        (0.75, 0.76, "Strong & Resilient",  STRATEGIC_COLORS["Strong & Resilient"]),
        (0.75, 0.24, "Strong but Exposed",   STRATEGIC_COLORS["Strong but Exposed"]),
        (0.25, 0.76, "Diversified Minor",    STRATEGIC_COLORS["Diversified Minor"]),
        (0.25, 0.24, "Weak & Vulnerable",    STRATEGIC_COLORS["Weak & Vulnerable"]),
    ]:
        fig_q.add_annotation(
            x=_xa, y=_ya, text=f"<b>{_txt}</b>",
            showarrow=False,
            font=dict(size=11, color=_clr),
            bgcolor="rgba(10,20,40,0.55)",
            bordercolor=_clr,
            borderwidth=1,
            borderpad=5,
            xref="x", yref="y",
        )

    fig_q.update_layout(**_lay(
        title="F Rank × G Rank — colored by strategic quadrant",
        xaxis_title="F Normalized Rank (trade power →)",
        yaxis_title="G Normalized Rank (resilience →)",
        height=540,
    ))
    st.plotly_chart(fig_q, width='stretch')

    thin_divider()

    # ── F score vs G score (raw values) ──────────────────────
    section_label("F Score vs G Score — Raw Values")

    fig_s = go.Figure()
    for _prof, _color in COMBINED_COLORS.items():
        _dp = _df_q[_df_q["combined_profile"] == _prof]
        if _dp.empty:
            continue
        fig_s.add_trace(go.Scatter(
            x=_dp["structural_score"], y=_dp["score_full_periode"],
            mode="markers+text", name=_prof,
            marker=dict(color=_color, size=_dp["_sz"], opacity=0.7,
                        line=dict(width=0.5, color="#304870")),
            text=_dp["_lbl"], textposition="top center",
            textfont=dict(size=10, color="#EEF4FB"),
            customdata=_dp[["country_name", "combined_score",
                            "strategic_quadrant_combined"]],
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "F score: %{x:.3f}<br>G score: %{y:.3f}<br>"
                          "Combined: %{customdata[1]:.3f}<br>"
                          "Quadrant: %{customdata[2]}<extra></extra>",
        ))

    _f_med = float(_df_q["structural_score"].median())
    _g_med = float(_df_q["score_full_periode"].median())
    fig_s.add_vline(x=_f_med, line_dash="dash", line_color="#475569",
                    annotation_text=f"Median F {_f_med:.2f}",
                    annotation_font=dict(color="#475569", size=10))
    fig_s.add_hline(y=_g_med, line_dash="dash", line_color="#475569",
                    annotation_text=f"Median G {_g_med:.2f}",
                    annotation_font=dict(color="#475569", size=10))

    fig_s.update_layout(**PLOTLY_LAYOUT,
        title="Structural Score (F) × Bilateral Risk Score (G) — colored by combined profile",
        xaxis_title="Structural Score (F) →",
        yaxis_title="Score Full Period (G) →",
        height=540,
    )
    st.plotly_chart(fig_s, width='stretch')

    # ── Weight Sensitivity Analysis ───────────────────────────
    thin_divider()
    section_label("Weight Sensitivity Analysis")

    with st.expander("Scoring methodology"):
        st.markdown("""
**Combined Score** = `0.70 × F_norm_rank + 0.30 × G_norm_rank`

The 70/30 split reflects a deliberate prioritization of **trade power** (size, growth, export coverage) over **bilateral resilience** (partner concentration, dependency). It can be read as: *a country with strong structural trade position matters more globally than one that is merely well-diversified but small.*

**G score formula:** `0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`

The stress test below shows how sensitive the ranking is to this choice.
        """)

    _wf = st.slider(
        "Weight on F (trade power)", min_value=0.50, max_value=0.90,
        value=0.70, step=0.05, key="stress_wf", format="%.2f",
    )
    _wg = round(1.0 - _wf, 2)

    _scen_lbl = {0.50: "Equal weight", 0.60: "Slight F priority",
                 0.70: "Baseline (70/30)", 0.80: "Strong F priority",
                 0.90: "F dominant"}.get(_wf, f"{_wf:.0%} F")
    st.caption(f"**{_scen_lbl}** — F: {_wf:.0%} · G: {_wg:.0%}")

    # Compute baseline ranks (70/30) and simulated ranks
    _df_st = df_combined[["country_name", "f_norm_rank", "g_norm_rank",
                           "combined_profile"]].dropna().copy()
    _df_st["score_base"] = 0.70 * _df_st["f_norm_rank"] + 0.30 * _df_st["g_norm_rank"]
    _df_st["score_sim"]  = _wf  * _df_st["f_norm_rank"] + _wg  * _df_st["g_norm_rank"]
    _df_st["rank_base"]  = _df_st["score_base"].rank(ascending=False).astype(int)
    _df_st["rank_sim"]   = _df_st["score_sim"].rank(ascending=False).astype(int)
    _df_st["rank_delta"] = _df_st["rank_base"] - _df_st["rank_sim"]  # positive = climbed

    _df_st_top20 = _df_st.nlargest(20, "score_sim").copy()
    _country_in_top20 = country in _df_st_top20["country_name"].values

    if _country_in_top20:
        _df_st_plot = _df_st_top20.sort_values("score_sim")
        _separator_idx = None
    else:
        # Add target country row below the top 20 (will appear at bottom of chart)
        _target_row = _df_st[_df_st["country_name"] == country].copy()
        _df_st_plot = pd.concat([_target_row, _df_st_top20], ignore_index=True)
        _df_st_plot = _df_st_plot.sort_values("score_sim")
        _separator_idx = float(_target_row["score_sim"].iloc[0]) if not _target_row.empty else None

    def _delta_lbl(d):
        if d > 0:   return f"▲+{d}"
        elif d < 0: return f"▼{d}"
        else:        return "="

    _df_st_plot["_dlbl"]    = _df_st_plot["rank_delta"].apply(_delta_lbl)
    _df_st_plot["_bar_txt"] = (_df_st_plot["score_sim"].apply(lambda v: f"{v:.3f}")
                                + "  " + _df_st_plot["_dlbl"])

    # Colors: target country gets its profile color + white border
    _bar_colors  = []
    _border_clrs = []
    _border_w    = []
    _opacities   = []
    for _, _r in _df_st_plot.iterrows():
        _is_target = _r["country_name"] == country
        _bar_colors.append(COMBINED_COLORS.get(_r["combined_profile"], "#6B7280"))
        _border_clrs.append("#FFFFFF" if _is_target else "rgba(0,0,0,0)")
        _border_w.append(2 if _is_target else 0)
        _opacities.append(1.0 if _is_target else (0.85 if _country_in_top20 else
                          (0.85 if _r["country_name"] != country else 0.5)))

    _n_bars = len(_df_st_plot)
    fig_st = go.Figure(go.Bar(
        x=_df_st_plot["score_sim"],
        y=_df_st_plot["country_name"],
        orientation="h",
        marker=dict(
            color=_bar_colors,
            line=dict(color=_border_clrs, width=_border_w),
            opacity=_opacities,
        ),
        text=_df_st_plot["_bar_txt"],
        textposition="outside",
        customdata=_df_st_plot[["rank_sim", "rank_base", "rank_delta", "combined_profile"]],
        hovertemplate="<b>%{y}</b><br>"
                      "Simulated score: %{x:.3f}<br>"
                      "Simulated rank: #%{customdata[0]}<br>"
                      "Baseline rank (70/30): #%{customdata[1]}<br>"
                      "Rank change: %{customdata[2]:+d}<br>"
                      "Profile: %{customdata[3]}<extra></extra>",
    ))

    # If target added outside top 20, draw a separator line
    if not _country_in_top20 and _separator_idx is not None:
        _top20_min = float(_df_st_top20["score_sim"].min())
        fig_st.add_hline(
            y=0.5,  # between first bar (target) and second bar
            line_dash="dot", line_color="#475569",
            annotation_text=f"← Target country (rank #{int(_df_st[_df_st['country_name']==country]['rank_sim'].iloc[0])})",
            annotation_font=dict(color="#475569", size=10),
            annotation_position="bottom right",
        )

    fig_st.update_layout(**_lay(
        title=f"Simulated Ranking — {_wf:.0%} F + {_wg:.0%} G  ·  score  Δrank vs 70/30",
        xaxis_title="Simulated Combined Score",
        height=max(560, _n_bars * 28 + 80),
        margin=dict(l=130, r=130, t=50, b=40),
    ))
    st.plotly_chart(fig_st, width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 3 — COUNTRY PROFILE CARD
# ══════════════════════════════════════════════════════════════════
with tab3:
    section_label("H.5 · Country Profile Card")

    thin_divider()

    # ── Card header ───────────────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1A2C48,#1F3050);
            border:2px solid {_cc};border-radius:16px;
            padding:1.25rem 1.5rem;margin-bottom:1rem;">
  <div style="display:flex;align-items:center;
              justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div style="font-size:11px;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:4px;">Country Profile Card</div>
      <div style="font-size:28px;font-weight:700;color:#EEF4FB;">{country}</div>
      <div style="font-size:14px;color:{_cc};font-weight:600;margin-top:4px;">{_cprof}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#6890B0;text-transform:uppercase;">Combined Score</div>
      <div style="font-size:40px;font-weight:700;color:{_cc};line-height:1.1;">
        {_row['combined_score']:.3f}</div>
      <div style="font-size:12px;color:#6890B0;margin-top:2px;">
        F: {_row['f_norm_rank']:.3f} &nbsp;·&nbsp; G: {_row['g_norm_rank']:.3f}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Three cards: F / G / H — equal height via CSS grid ───────
    _prof_rec       = str(_row.get("profile_recent", "—"))
    _pr_c           = PROFILE_COLORS.get(_prof_rec, "#C4D5E8")
    _conv           = _row.get("convergence_rate", None)
    _conv_str       = f"{_conv:.1%}" if pd.notna(_conv) else "—"
    _stab_score     = _row.get("stability_score", None)
    _stab_score_str = f"{_stab_score:.3f}" if pd.notna(_stab_score) else "—"
    _dm             = _row.get("drift_magnitude", None)
    _dm_str         = f"{_dm:.0f}" if pd.notna(_dm) else "—"
    _static_q       = str(_row.get("static_quadrant", "—"))
    _static_q_c     = STATIC_Q_COLORS.get(_static_q, "#C4D5E8")
    _dynamic_q      = str(_row.get("dynamic_quadrant", "—"))
    _dynamic_q_c    = DYNAMIC_Q_COLORS.get(_dynamic_q, "#C4D5E8")
    _strategic_qf   = str(_row.get("strategic_quadrant", "—"))
    _strategic_qf_c = STRATEGIC_F_COLORS.get(_strategic_qf, "#C4D5E8")

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;
            gap:1rem;align-items:stretch;margin-bottom:.75rem;">

  <!-- F · Trade Structure -->
  <div style="background:#231800;border:1.5px solid #D97706;
              border-radius:12px;padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:11px;font-weight:700;color:#FCD34D;
                text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">
      F · Trade Structure</div>
    <div style="font-size:13px;color:#FBBf24;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">Structural score</span> &nbsp;→&nbsp;
           {_row.get("structural_score", 0):.3f}</div>
      <div><span style="color:#6890B0;">Profile</span> &nbsp;→&nbsp;
           <span style="color:{_sc};font-weight:600;">{_sprof}</span></div>
      <div><span style="color:#6890B0;">Static quadrant</span> &nbsp;→&nbsp;
           <span style="color:{_static_q_c};font-weight:600;">{_static_q}</span></div>
      <div><span style="color:#6890B0;">Dynamic quadrant</span> &nbsp;→&nbsp;
           <span style="color:{_dynamic_q_c};font-weight:600;">{_dynamic_q}</span></div>
      <div><span style="color:#6890B0;">Strategic quadrant</span> &nbsp;→&nbsp;
           <span style="color:{_strategic_qf_c};font-weight:600;">{_strategic_qf}</span></div>
      <div><span style="color:#6890B0;">Meta</span> &nbsp;→&nbsp;
           {_row.get("meta_profile","—")}</div>
      <div><span style="color:#6890B0;">Export share</span> &nbsp;→&nbsp;
           {_row["export_share_structural"]:.2f}%</div>
      <div><span style="color:#6890B0;">Coverage</span> &nbsp;→&nbsp;
           {_row["coverage_ratio_structural"]:.3f}</div>
      <div><span style="color:#6890B0;">CAGR export</span> &nbsp;→&nbsp;
           {_row["cagr_export"]*100:.1f}%</div>
      <div><span style="color:#6890B0;">Volatility</span> &nbsp;→&nbsp;
           {_row["volatility_export"]:.2f}</div>
    </div>
  </div>

  <!-- G · Bilateral Risk -->
  <div style="background:#0F2240;border:1.5px solid #3B82F6;
              border-radius:12px;padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:11px;font-weight:700;color:#93C5FD;
                text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">
      G · Bilateral Risk</div>
    <div style="font-size:13px;color:#BAD7F8;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">Score (Full Period)</span> &nbsp;→&nbsp;
           {_row.get("score_full_periode", 0):.3f}</div>
      <div><span style="color:#6890B0;">Profile (Full Period)</span> &nbsp;→&nbsp;
           <span style="color:{_bc};font-weight:600;">{_bprof}</span></div>
      <div><span style="color:#6890B0;">Profile (Recent)</span> &nbsp;→&nbsp;
           <span style="color:{_pr_c};font-weight:600;">{_prof_rec}</span></div>
      <div><span style="color:#6890B0;">Convergence Rate</span> &nbsp;→&nbsp; {_conv_str}</div>
      <div><span style="color:#6890B0;">Score Import</span> &nbsp;→&nbsp;
           {_row.get("score_import", 0):.3f}</div>
      <div><span style="color:#6890B0;">Score Export</span> &nbsp;→&nbsp;
           {_row.get("score_export", 0):.3f}</div>
      <div><span style="color:#6890B0;">BDIS</span> &nbsp;→&nbsp; {_row["bdis"]:.3f}</div>
      <div><span style="color:#6890B0;">HHI Import</span> &nbsp;→&nbsp;
           {_row["hhi_import"]:.3f}</div>
      <div><span style="color:#6890B0;">HHI Export</span> &nbsp;→&nbsp;
           {_row["hhi_export"]:.3f}</div>
      <div><span style="color:#6890B0;">Critical share</span> &nbsp;→&nbsp;
           {_row["critical_share"]:.1%}</div>
      <div><span style="color:#6890B0;">Top partner</span> &nbsp;→&nbsp;
           {_row.get("top_partner","—")}</div>
    </div>
  </div>

  <!-- H · Combined -->
  <div style="background:#082E18;border:1.5px solid #10B981;
              border-radius:12px;padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:11px;font-weight:700;color:#34D399;
                text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">
      H · Combined</div>
    <div style="font-size:13px;color:#6EE7B7;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">Combined score</span> &nbsp;→&nbsp;
           <span style="color:{_cc};font-weight:700;">{_row["combined_score"]:.3f}</span></div>
      <div><span style="color:#6890B0;">Quadrant</span> &nbsp;→&nbsp;
           <span style="color:{_sq_c};font-weight:600;">{_strat_q}</span></div>
      <div><span style="color:#6890B0;">Stability</span> &nbsp;→&nbsp;
           <span style="color:{_stab_c};font-weight:600;">{_stab}</span></div>
      <div><span style="color:#6890B0;">Stability score</span> &nbsp;→&nbsp; {_stab_score_str}</div>
      <div><span style="color:#6890B0;">Profile Drift</span> &nbsp;→&nbsp;
           <span style="color:{_drift_c};font-weight:600;">{_drift}</span></div>
      <div><span style="color:#6890B0;">Drift Magnitude</span> &nbsp;→&nbsp; {_dm_str}</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.75rem;'></div>", unsafe_allow_html=True)

    # ── Radar + Scorecard ─────────────────────────────────────
    _cr, _cs_col = st.columns(2)

    with _cr:
        section_label("Dimension Radar")
        _radar_default = [c for c in ["Australia", "Belgium"] if c in COUNTRIES and c != country]
        if st.session_state.get("_radar_prev_country") != country:
            st.session_state["card_compare"] = _radar_default
            st.session_state["_radar_prev_country"] = country
        compare_c = st.multiselect(
            "Compare with",
            [c for c in COUNTRIES if c != country],
            default=_radar_default, max_selections=3,
            key="card_compare",
        )

        _PAL = ["#F59E0B", "#60A5FA", "#10B981", "#EF4444"]
        fig_r = go.Figure()

        for _i, _cn in enumerate([country] + compare_c):
            _rrow = df_combined[df_combined["country_name"] == _cn]
            if _rrow.empty:
                continue
            _rrow = _rrow.iloc[0]
            _vals = [RADAR_SCALE[d].get(_rrow.get(d, ""), 0) for d in RADAR_DIMS]
            _clr  = _cc if _cn == country else _PAL[_i % 4]
            _r, _g, _b = int(_clr[1:3], 16), int(_clr[3:5], 16), int(_clr[5:7], 16)

            fig_r.add_trace(go.Scatterpolar(
                r=_vals + [_vals[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
                fill="toself", name=_cn,
                line=dict(color=_clr, width=2),
                fillcolor=f"rgba({_r},{_g},{_b},0.12)",
                opacity=0.9,
            ))

        fig_r.update_layout(
            paper_bgcolor="#1F3050",
            font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
            polar=dict(
                bgcolor="#162035",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#304870"),
                angularaxis=dict(gridcolor="#304870"),
            ),
            legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
            height=380, margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_r, width='stretch')

    with _cs_col:
        section_label("Profile Scorecard")

        _DIMS = [
            ("Size",        "size_profile"),
            ("Balance",     "balance_profile"),
            ("Growth",      "growth_profile"),
            ("Stability",   "stability_profile"),
            ("Consistency", "consistency_profile"),
        ]
        _LVL_C = {1: "#10B981", 2: "#60A5FA", 3: "#F59E0B", 4: "#EF4444"}
        _LVL_M = {
            "size_profile":        {"Leader": 1, "Major": 2, "Mid": 3, "Small": 4},
            "balance_profile":     {"Strong Surplus": 1, "Slight Surplus": 2,
                                    "Slight Deficit": 3, "Structural Deficit": 4},
            "growth_profile":      {"Strong Grower": 1, "Moderate Grower": 2,
                                    "Weak Grower": 3, "Declining": 4},
            "stability_profile":   {"Stable": 1, "Moderate": 2,
                                    "Volatile": 3, "Very Volatile": 4},
            "consistency_profile": {"Very Consistent": 1, "Consistent": 2,
                                    "Inconsistent": 3, "Very Inconsistent": 4},
        }
        _BAR_W = {1: "100%", 2: "67%", 3: "33%", 4: "15%"}

        for _lbl, _col in _DIMS:
            _val = _row.get(_col, "—")
            _lvl = _LVL_M.get(_col, {}).get(_val, 2)
            _clr = _LVL_C.get(_lvl, "#C4D5E8")
            _bw  = _BAR_W.get(_lvl, "50%")
            st.markdown(f"""
<div style="margin-bottom:.5rem;">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
    <span style="color:#A8C0D8;">{_lbl}</span>
    <span style="color:{_clr};font-weight:600;">{_val}</span>
  </div>
  <div style="background:#162035;border-radius:4px;height:6px;">
    <div style="background:{_clr};width:{_bw};height:6px;border-radius:4px;opacity:.85;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="margin-top:1rem;background:#1A2C48;border:1px solid #304870;
            border-left:4px solid {_cc};border-radius:0 10px 10px 0;
            padding:.75rem 1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:11px;color:#6890B0;">Structural Score</div>
      <div style="font-size:22px;font-weight:700;color:{_sc};">
        {_row['structural_score']:.3f}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#6890B0;">BDIS</div>
      <div style="font-size:22px;font-weight:700;color:{_bc};">
        {_row['bdis']:.3f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Annual Profile Timeline ───────────────────────────────
    thin_divider()
    section_label("Annual Profile Timeline")

    _df_tl = df_final[df_final["country_name"] == country].copy()

    if not _df_tl.empty:
        _df_tl = _df_tl.sort_values("year")
        _years = [int(y) for y in _df_tl["year"].tolist()]

        # Single heatmap with 2 rows: Score (bottom) / Logic (top)
        # .astype(str) converts Arrow-backed strings to regular Python strings
        _sc_num = _df_tl["profile_score_global"].astype(str).map(_PROF_NUM).tolist()
        _lg_num = _df_tl["profile_logic_global"].astype(str).map(_PROF_NUM).tolist()
        _sc_lbl = _df_tl["profile_score_global"].astype(str).tolist()
        _lg_lbl = _df_tl["profile_logic_global"].astype(str).tolist()

        fig_tl = go.Figure(go.Heatmap(
            z=[_lg_num, _sc_num],
            x=_years,
            y=["Logic Profile", "Score Profile"],
            colorscale=_TL_COLORSCALE, zmin=0, zmax=3,
            text=[_lg_lbl, _sc_lbl],
            texttemplate="%{text}",
            textfont=dict(size=9, color="#EEF4FB"),
            showscale=False,
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Profile: %{text}<extra></extra>",
            xgap=2, ygap=4,
        ))
        fig_tl.update_layout(**_lay(
            title=f"Annual Bilateral Profile — {country}",
            xaxis=dict(
                tickmode="array", tickvals=_years,
                ticktext=[str(y) for y in _years],
                gridcolor="rgba(0,0,0,0)", linecolor="#304870",
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="#304870"),
            height=200,
            margin=dict(l=110, r=20, t=50, b=40),
        ))
        st.plotly_chart(fig_tl, width='stretch')

        _leg = '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:2px;">'
        for _p, _c in zip(_PROF_ORDER, _PROF_CLR):
            _leg += (f'<span style="display:flex;align-items:center;gap:5px;font-size:12px;">'
                     f'<span style="width:12px;height:12px;border-radius:2px;'
                     f'background:{_c};display:inline-block;"></span>'
                     f'<span style="color:#A8C0D8;">{_p}</span></span>')
        _leg += '</div>'
        st.markdown(_leg, unsafe_allow_html=True)
    else:
        st.info(f"No annual data available for {country}.")
