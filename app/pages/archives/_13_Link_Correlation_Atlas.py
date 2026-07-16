"""
Trade × Indicators — Link · Correlation Atlas
Spearman correlations, RF Feature Importance, Kruskal-Wallis profiles
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding

inject_styles()

LINK_FILES = ROOT / "Analysis" / "Analysis_files" / "Link"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_sp  = pd.read_csv(LINK_FILES / "df_spearman.csv")
    df_fi  = pd.read_csv(LINK_FILES / "fi_matrix.csv", index_col=0)
    df_kw  = pd.read_csv(LINK_FILES / "df_kw.csv")
    return df_sp, df_fi, df_kw

df_sp, df_fi, df_kw = load_data()

# ── Reconstruct fi_matrix indicator labels (alphabetical order, same as notebook) ──
_ind_alpha_labels = sorted(
    df_sp[["indicator", "ind_label"]].drop_duplicates()["ind_label"].tolist()
)
_ind_alpha_full = sorted(
    df_sp[["indicator", "ind_label"]].drop_duplicates()["indicator"].tolist()
)
df_fi = df_fi.copy()
df_fi.index = _ind_alpha_labels

_flag_map = (df_sp[["ind_label", "ind_flag"]].drop_duplicates()
             .set_index("ind_label")["ind_flag"].to_dict())

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

def _pl(fig, **kw):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kw})
    return fig

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Link · Trade × Indicators · Page 1 of 2</div>',
            unsafe_allow_html=True)
st.title("Correlation Atlas")
st.caption(
    "Spearman correlations, Random Forest feature importance, and "
    "Kruskal-Wallis profiles — 16 macro indicators × 44 trade variables"
)
thin_divider()

# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

_n_pairs   = len(df_sp)
_n_sig     = int(df_sp["sig"].sum())
_sig_rate  = _n_sig / _n_pairs * 100
_top_e     = (df_sp[df_sp["ind_flag"] == "[E]"]
              .groupby("ind_label")["abs_rho"].mean()
              .idxmax())
_top_e_rho = (df_sp[df_sp["ind_flag"] == "[E]"]
              .groupby("ind_label")["abs_rho"].mean()
              .max())

cols_kpi = st.columns(4)
for col, (label, value, color) in zip(cols_kpi, [
    ("Pairs Tested",   f"{_n_pairs:,}",          "blue"),
    ("Significant",    f"{_n_sig:,}",             "teal"),
    ("Sig. Rate",      f"{_sig_rate:.0f}%",       "amber"),
    ("Top [E] Signal", _top_e,                    ""),
]):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# METHODOLOGY NOTE
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex;gap:12px;margin-bottom:1rem;">
  <div style="background:#1F3050;border:1px solid #304870;border-left:3px solid #EF4444;
              border-radius:0 10px 10px 0;padding:.6rem 1rem;flex:1;font-size:13px;color:#A8C0D8;">
    <strong style="color:#FCA5A5;">[T] Tautological</strong> — indicator is directly derived from trade flows
    (e.g. Exports USD). High correlation is structural, not causal.
  </div>
  <div style="background:#1F3050;border:1px solid #304870;border-left:3px solid #F59E0B;
              border-radius:0 10px 10px 0;padding:.6rem 1rem;flex:1;font-size:13px;color:#A8C0D8;">
    <strong style="color:#FCD34D;">[S] Size-driven</strong> — indicator reflects country size (GDP, Population).
    Correlation with trade volumes is scale effect, not economic signal.
  </div>
  <div style="background:#1F3050;border:1px solid #304870;border-left:3px solid #10B981;
              border-radius:0 10px 10px 0;padding:.6rem 1rem;flex:1;font-size:13px;color:#A8C0D8;">
    <strong style="color:#34D399;">[E] Economic</strong> — genuine macro signal independent of trade definition
    (e.g. GDP/capita, Inflation). Correlations here are analytically meaningful.
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_a, tab_b, tab_c = st.tabs([
    "A · Spearman Heatmap",
    "B · RF Feature Importance",
    "C · Kruskal-Wallis Profiles",
])

# ────────────────────────────────────────────────────────────────
# TAB A — SPEARMAN HEATMAP
# ────────────────────────────────────────────────────────────────
with tab_a:
    section_label("A · Spearman ρ — Indicator × Trade Variable")

    # Core trade variables for default display
    _CORE_TRADE_VARS = [
        "export_share_structural", "import_share_structural", "coverage_ratio_structural",
        "cagr_export", "cagr_import", "volatility_export", "volatility_import",
        "resilience_export", "structural_score",
        "combined_score", "f_norm_rank", "g_norm_rank",
        "export_pc", "import_pc", "balance_dep",
    ]

    all_tv = (df_sp[["trade_var", "trade_label"]].drop_duplicates()
              .sort_values("trade_label"))
    default_labels = [r["trade_label"] for _, r in all_tv.iterrows()
                      if r["trade_var"] in _CORE_TRADE_VARS]

    col_a1, col_a2, col_a3 = st.columns([2, 2, 4])
    flag_a = col_a1.radio("Indicator type", ["All", "[E]", "[T]", "[S]"],
                          horizontal=True, key="sp_flag")
    sig_a  = col_a2.toggle("Significant only", value=True, key="sp_sig")
    sel_tv = col_a3.multiselect("Trade variables",
                                all_tv["trade_label"].tolist(),
                                default=default_labels,
                                key="sp_tv")

    # Build filtered dataset
    df_a = df_sp.copy()
    if flag_a != "All":
        df_a = df_a[df_a["ind_flag"] == flag_a]
    if sel_tv:
        sel_vars = all_tv[all_tv["trade_label"].isin(sel_tv)]["trade_var"].tolist()
        df_a = df_a[df_a["trade_var"].isin(sel_vars)]

    if df_a.empty:
        st.info("No correlations match the current filters.")
    else:
        # Pivot: show all rho values, mask non-sig with NaN when toggle active
        df_plot_a = df_a.copy()
        if sig_a:
            df_plot_a.loc[~df_plot_a["sig"], "rho"] = float("nan")

        pivot_a = df_plot_a.pivot_table(
            index="ind_label", columns="trade_label", values="rho", aggfunc="first"
        )

        # Sort rows by mean abs_rho
        row_order = (df_a.groupby("ind_label")["abs_rho"].mean()
                     .reindex(pivot_a.index).fillna(0).sort_values().index.tolist())
        col_order = (df_a.groupby("trade_label")["abs_rho"].mean()
                     .reindex(pivot_a.columns).fillna(0).sort_values().index.tolist())
        pivot_a = pivot_a.loc[row_order, col_order]

        # Color annotation per flag
        _flag_colors = {"[E]": "#10B981", "[T]": "#EF4444", "[S]": "#F59E0B"}
        y_labels = [
            f'<span style="color:{_flag_colors.get(_flag_map.get(i,""), "#C4D5E8")}">'
            f'{i}</span>'
            for i in pivot_a.index
        ]

        fig_sp = go.Figure(go.Heatmap(
            z=pivot_a.values,
            x=pivot_a.columns.tolist(),
            y=pivot_a.index.tolist(),
            colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="ρ", len=0.8),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        ))
        _pl(fig_sp,
            title=f"Spearman ρ {'(significant only)' if sig_a else '(all pairs)'}",
            height=max(400, len(pivot_a) * 32 + 120),
            margin=dict(l=160, r=40, t=60, b=120),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11), gridcolor="#304870"),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#304870"),
        )
        st.plotly_chart(fig_sp, use_container_width=True)

        # Top findings
        _top5 = (df_a[df_a["sig"] & (df_a["ind_flag"] == "[E]")]
                 .nlargest(5, "abs_rho")[["ind_label", "trade_label", "rho", "ind_flag"]])
        with st.expander("Top significant [E] correlations"):
            if _top5.empty:
                st.write("No significant [E] correlations with current filter.")
            else:
                for _, row in _top5.iterrows():
                    direction = "↑" if row["rho"] > 0 else "↓"
                    finding(
                        f"<strong>{row['ind_label']}</strong> × {row['trade_label']} "
                        f"— ρ = <strong>{row['rho']:+.3f}</strong> {direction}"
                    )

# ────────────────────────────────────────────────────────────────
# TAB B — RF FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────────
with tab_b:
    section_label("B · Random Forest Feature Importance")

    st.markdown(
        "<p style='font-size:13px;color:#7090B0;margin-bottom:.5rem;'>"
        "One RF model per trade variable. Feature importance = mean decrease in impurity "
        "across all trees. [T] and [S] indicators dominate — the true macro signal "
        "comes from [E] indicators after filtering them out.</p>",
        unsafe_allow_html=True,
    )

    _CORE_FI_COLS = [
        "Export Share", "Import Share", "Coverage Ratio",
        "CAGR Export", "CAGR Import", "Volatility Export", "Volatility Import",
        "Resilience Export", "Structural Score",
        "Combined Score", "F Norm Rank", "G Norm Rank",
        "Export per Capita", "Import per Capita", "Balance / GDP",
    ]
    avail_fi_cols = [c for c in _CORE_FI_COLS if c in df_fi.columns]

    col_b1, col_b2 = st.columns([2, 2])
    flag_b = col_b1.radio("Highlight type", ["All", "[E]", "[T]", "[S]"],
                          horizontal=True, key="fi_flag")

    df_fi_disp = df_fi[avail_fi_cols].copy()

    col_heat, col_bar = st.columns([3, 1])

    with col_heat:
        _alpha = 0.9
        _colors_b = []
        for ind in df_fi_disp.index:
            f = _flag_map.get(ind, "")
            if flag_b == "All" or flag_b == f:
                _colors_b.append(ind)
            # all rows shown, just filter highlighting via bar chart

        fig_fi_h = go.Figure(go.Heatmap(
            z=df_fi_disp.values,
            x=df_fi_disp.columns.tolist(),
            y=df_fi_disp.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title="FI", len=0.8),
            hovertemplate="<b>%{y}</b> → %{x}<br>FI = %{z:.4f}<extra></extra>",
        ))
        _pl(fig_fi_h,
            title="Feature Importance — Indicator × Trade Variable",
            height=520,
            margin=dict(l=160, r=40, t=60, b=120),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11), gridcolor="#304870"),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#304870"),
        )
        st.plotly_chart(fig_fi_h, use_container_width=True)

    with col_bar:
        mean_fi = df_fi_disp.mean(axis=1).sort_values()
        _bar_colors = []
        for ind in mean_fi.index:
            f = _flag_map.get(ind, "")
            if flag_b != "All" and f != flag_b:
                _bar_colors.append("#2A3D5C")
            elif f == "[T]":
                _bar_colors.append("#EF4444")
            elif f == "[S]":
                _bar_colors.append("#F59E0B")
            else:
                _bar_colors.append("#10B981")

        fig_fi_bar = go.Figure(go.Bar(
            y=mean_fi.index,
            x=mean_fi.values,
            orientation="h",
            marker_color=_bar_colors,
            text=mean_fi.values.round(3),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Mean FI = %{x:.4f}<extra></extra>",
        ))
        _pl(fig_fi_bar,
            title="Mean FI",
            height=520,
            margin=dict(l=120, r=60, t=60, b=40),
            xaxis=dict(gridcolor="#304870", tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#304870"),
        )
        st.plotly_chart(fig_fi_bar, use_container_width=True)

    with st.expander("Key interpretation"):
        finding(
            "<strong>Dominance of [T]/[S]</strong>: Imports & Exports in USD and Population "
            "explain most variance because they are mechanically linked to trade volume variables. "
            "This is expected and does not imply economic causality."
        )
        finding(
            "<strong>Signal from [E] indicators</strong>: Among economically independent "
            "indicators, GDP/capita and Current Account % GDP carry the most signal — "
            "confirmed by both RF importance and Spearman ρ."
        )
        finding(
            "<strong>Structure is macro-explained</strong>: trade <em>position</em> (combined "
            "score, structural score) correlates with economic development level (GDP/capita). "
            "Trade <em>dynamics</em> are idiosyncratic — macro explains little of growth/volatility."
        )

# ────────────────────────────────────────────────────────────────
# TAB C — KRUSKAL-WALLIS
# ────────────────────────────────────────────────────────────────
with tab_c:
    section_label("C · Kruskal-Wallis — Profile × Indicator")

    st.markdown(
        "<p style='font-size:13px;color:#7090B0;margin-bottom:.5rem;'>"
        "Kruskal-Wallis H-test: do different trade profiles have significantly different "
        "macro indicator distributions? η² (eta-squared) = effect size "
        "(0–0.06 small, 0.06–0.14 medium, >0.14 large).</p>",
        unsafe_allow_html=True,
    )

    _pv_labels = (df_kw[["profile_var", "profile_label"]].drop_duplicates()
                  .set_index("profile_var")["profile_label"].to_dict())

    col_c1, col_c2, col_c3 = st.columns([3, 2, 2])
    sel_pv = col_c1.selectbox(
        "Trade profile variable",
        list(_pv_labels.keys()),
        format_func=lambda x: _pv_labels.get(x, x),
        key="kw_pv",
    )
    flag_c = col_c2.radio("Indicator type", ["All", "[E]", "[T]", "[S]"],
                          horizontal=True, key="kw_flag")
    sig_c  = col_c3.toggle("Significant only (p < 0.05)", value=True, key="kw_sig")

    df_c = df_kw[df_kw["profile_var"] == sel_pv].copy()
    if flag_c != "All":
        df_c = df_c[df_c["ind_flag"] == flag_c]
    if sig_c:
        df_c = df_c[df_c["sig"]]

    if df_c.empty:
        st.info("No significant results for this profile × indicator combination. Try removing filters.")
    else:
        df_c_sorted = df_c.sort_values("eta2")

        _eff_colors = []
        for v in df_c_sorted["eta2"]:
            if v >= 0.14:
                _eff_colors.append("#10B981")
            elif v >= 0.06:
                _eff_colors.append("#F59E0B")
            else:
                _eff_colors.append("#60A5FA")

        _flag_sym = {"[E]": "●", "[T]": "▲", "[S]": "■"}
        _y_labels = [
            f"{_flag_sym.get(_flag_map.get(i, ''), '?')} {i}"
            for i in df_c_sorted["ind_label"]
        ]

        fig_kw = go.Figure(go.Bar(
            y=_y_labels,
            x=df_c_sorted["eta2"],
            orientation="h",
            marker_color=_eff_colors,
            text=df_c_sorted["eta2"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
            customdata=df_c_sorted[["H_stat", "pvalue", "n_groups"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "η² = %{x:.3f}<br>"
                "H = %{customdata[0]:.1f} | p = %{customdata[1]:.4f}<br>"
                "Groups: %{customdata[2]}<extra></extra>"
            ),
        ))

        # Reference lines
        for xref_val, xref_label, xref_color in [
            (0.06, "Medium (0.06)", "#F59E0B"),
            (0.14, "Large (0.14)", "#10B981"),
        ]:
            fig_kw.add_vline(x=xref_val, line_dash="dash", line_color=xref_color,
                             annotation_text=xref_label,
                             annotation_font_color=xref_color,
                             annotation_position="top right")

        _pl(fig_kw,
            title=f"Kruskal-Wallis η² — {_pv_labels.get(sel_pv, sel_pv)}",
            xaxis_title="η² (Effect Size)",
            height=max(350, len(df_c) * 34 + 120),
            margin=dict(l=180, r=80, t=60, b=40),
            xaxis=dict(gridcolor="#304870"),
            yaxis=dict(tickfont=dict(size=12), gridcolor="#304870"),
        )
        st.plotly_chart(fig_kw, use_container_width=True)

        # Legend
        st.markdown("""
        <div style="display:flex;gap:24px;font-size:12px;color:#6890B0;margin-top:.5rem;">
          <span><span style="color:#10B981">■</span> Large effect (η² ≥ 0.14)</span>
          <span><span style="color:#F59E0B">■</span> Medium effect (η² 0.06–0.14)</span>
          <span><span style="color:#60A5FA">■</span> Small effect (η² < 0.06)</span>
          <span>● [E] economic  ▲ [T] tautological  ■ [S] size</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Interpretation"):
            _top_kw = df_c.nlargest(3, "eta2")
            for _, row in _top_kw.iterrows():
                eff = "large" if row["eta2"] >= 0.14 else "medium" if row["eta2"] >= 0.06 else "small"
                finding(
                    f"<strong>{row['ind_label']}</strong> [{row['ind_flag'].strip('[]')}] "
                    f"— η² = {row['eta2']:.3f} (<em>{eff} effect</em>). "
                    f"Trade profiles differ significantly in this indicator "
                    f"(H = {row['H_stat']:.1f}, p = {row['pvalue']:.4f})."
                )

thin_divider()
