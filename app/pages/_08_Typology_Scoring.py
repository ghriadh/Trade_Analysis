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
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 6 of 6",
        "title": "Typology & Final Scoring",
        "caption": "Combined F×G scoring · Strategic quadrant · Country Profile Card",
        "kpi_global_champions": "Global Champions",
        "kpi_competitive": "Competitive",
        "kpi_vulnerable": "Vulnerable",
        "kpi_fragile": "Fragile",
        "select_country": "Select country",
        "no_data_country": "No data for {country}.",
        "tab1_label": "📊 Combined Scoring & Profiles",
        "tab2_label": "🎯 Quadrant & Stress Test",
        "tab3_label": "🏅 Country Profile Card",
        "lbl_target_country": "Target Country",
        "lbl_combined_score": "Combined Score",
        "lbl_rank": "Rank #{pos} / {n}",
        "lbl_combined_profile": "Combined Profile",
        "lbl_strategic_quadrant": "Strategic Quadrant",
        "lbl_f_rank": "F Rank",
        "lbl_g_rank": "G Rank",
        "def_combined_score_desc": "F = Trade Structure score (size, growth, balance, coverage) "
                                    "&nbsp;·&nbsp; G = Bilateral Risk score (HHI, BDIS, critical share)",
        "def_threshold_header": "Threshold",
        "def_strategic_quadrant_desc": "Position in F (trade power) × G (resilience) space — threshold at 0.55",
        "quad_g_low": "G Low",
        "quad_more_risk": "More Risk",
        "quad_g_high": "G High",
        "quad_resilient": "Resilient",
        "quad_f_high": "F High",
        "quad_more_power": "More Power",
        "quad_f_low": "F Low",
        "quad_less_power": "Less Power",
        "matrix_section": "F × G · Typology Matrix",
        "fig_cross_title": "Structural Profile (F) × Bilateral Profile (G)",
        "fig_cross_xaxis": "Bilateral Profile (G)",
        "fig_cross_yaxis": "Structural Profile (F)",
        "fig_comb_title": "Combined Profile Distribution",
        "axis_countries": "Countries",
        "fig_strat_title": "Strategic Quadrant Distribution",
        "ranking_section": "Combined Score — Ranking",
        "show_top_n_label": "Show top N",
        "view_label": "View",
        "fig_rank_title": "Combined Score — {view}  (0.70 × F + 0.30 × G)",
        "axis_combined_score": "Combined Score",
        "summary_section": "All Countries — Combined Scoring Summary",
        "pills_add_columns_label": "Add columns",
        "col_country": "Country",
        "col_combined_score": "Combined Score",
        "col_combined_profile": "Combined Profile",
        "col_strategic_quadrant": "Strategic Quadrant",
        "col_structural_profile_f": "Structural Profile (F)",
        "col_f_score": "F Score",
        "col_profile_full_period_g": "Profile Full Period (G)",
        "col_g_score": "G Score",
        "tab2_highlight_countries_label": "Highlight countries",
        "tab2_quadrant_section": "Strategic Quadrant — F × G Ranks",
        "fig_q_title": "F Rank × G Rank — colored by strategic quadrant",
        "fig_q_xaxis": "F Normalized Rank (trade power →)",
        "fig_q_yaxis": "G Normalized Rank (resilience →)",
        "fg_score_section": "F Score vs G Score — Raw Values",
        "median_f_annot": "Median F {val}",
        "median_g_annot": "Median G {val}",
        "fig_s_title": "Structural Score (F) × Bilateral Risk Score (G) — colored by combined profile",
        "fig_s_xaxis": "Structural Score (F) →",
        "fig_s_yaxis": "Score Full Period (G) →",
        "weight_sensitivity_section": "Weight Sensitivity Analysis",
        "scoring_methodology_expander": "Scoring methodology",
        "scoring_methodology_md": """
**Combined Score** = `0.70 × F_norm_rank + 0.30 × G_norm_rank`

The 70/30 split reflects a deliberate prioritization of **trade power** (size, growth, export coverage) over **bilateral resilience** (partner concentration, dependency). It can be read as: *a country with strong structural trade position matters more globally than one that is merely well-diversified but small.*

**G score formula:** `0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`

The stress test below shows how sensitive the ranking is to this choice.
        """,
        "weight_on_f_label": "Weight on F (trade power)",
        "scen_equal": "Equal weight",
        "scen_slight_f": "Slight F priority",
        "scen_baseline": "Baseline (70/30)",
        "scen_strong_f": "Strong F priority",
        "scen_f_dominant": "F dominant",
        "target_country_separator": "← Target country (rank #{rank})",
        "stress_title": "Simulated Ranking — {wf} F + {wg} G  ·  score  Δrank vs 70/30",
        "stress_xaxis": "Simulated Combined Score",
        "tab3_section_profile_card": "H.5 · Country Profile Card",
        "lbl_country_profile_card_header": "Country Profile Card",
        "card_f_header": "F · Trade Structure",
        "card_f_structural_score": "Structural score",
        "card_f_profile": "Profile",
        "card_f_static_quadrant": "Static quadrant",
        "card_f_dynamic_quadrant": "Dynamic quadrant",
        "card_f_meta": "Meta",
        "card_f_export_share": "Export share",
        "card_f_coverage": "Coverage",
        "card_f_cagr_export": "CAGR export",
        "card_f_volatility": "Volatility",
        "card_g_header": "G · Bilateral Risk",
        "card_g_score_full_period": "Score (Full Period)",
        "card_g_profile_full_period": "Profile (Full Period)",
        "card_g_profile_recent": "Profile (Recent)",
        "card_g_convergence_rate": "Convergence Rate",
        "card_g_score_import": "Score Import",
        "card_g_score_export": "Score Export",
        "card_g_hhi_import": "HHI Import",
        "card_g_hhi_export": "HHI Export",
        "card_g_critical_share": "Critical share",
        "card_g_top_partner": "Top partner",
        "card_h_header": "H · Combined",
        "card_h_quadrant": "Quadrant",
        "card_h_stability": "Stability",
        "card_h_stability_score": "Stability score",
        "card_h_profile_drift": "Profile Drift",
        "card_h_drift_magnitude": "Drift Magnitude",
        "dimension_radar_section": "Dimension Radar",
        "compare_with_label": "Compare with",
        "radar_size": "Size",
        "radar_balance": "Balance",
        "radar_growth": "Growth",
        "radar_stability": "Stability",
        "radar_consistency": "Consistency",
        "scorecard_section": "Profile Scorecard",
        "annual_timeline_section": "Annual Profile Timeline",
        "heatmap_logic_profile": "Logic Profile",
        "heatmap_score_profile": "Score Profile",
        "fig_tl_title": "Annual Bilateral Profile — {country}",
        "no_annual_data": "No annual data available for {country}.",
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 6 sur 6",
        "title": "Typologie & scoring final",
        "caption": "Scoring combiné F×G · Quadrant stratégique · Fiche profil pays",
        "kpi_global_champions": "Champions mondiaux",
        "kpi_competitive": "Compétitifs",
        "kpi_vulnerable": "Vulnérables",
        "kpi_fragile": "Fragiles",
        "select_country": "Sélectionner un pays",
        "no_data_country": "Aucune donnée pour {country}.",
        "tab1_label": "📊 Scoring combiné & profils",
        "tab2_label": "🎯 Quadrant & test de stress",
        "tab3_label": "🏅 Fiche profil pays",
        "lbl_target_country": "Pays cible",
        "lbl_combined_score": "Score combiné",
        "lbl_rank": "Rang n°{pos} / {n}",
        "lbl_combined_profile": "Profil combiné",
        "lbl_strategic_quadrant": "Quadrant stratégique",
        "lbl_f_rank": "Rang F",
        "lbl_g_rank": "Rang G",
        "def_combined_score_desc": "F = Score de structure commerciale (taille, croissance, équilibre, "
                                    "couverture) &nbsp;·&nbsp; G = Score de risque bilatéral (HHI, BDIS, "
                                    "part critique)",
        "def_threshold_header": "Seuil",
        "def_strategic_quadrant_desc": "Position dans l'espace F (puissance commerciale) × G (résilience) "
                                        "— seuil à 0.55",
        "quad_g_low": "G faible",
        "quad_more_risk": "Risque plus élevé",
        "quad_g_high": "G élevé",
        "quad_resilient": "Résilient",
        "quad_f_high": "F élevé",
        "quad_more_power": "Puissance plus élevée",
        "quad_f_low": "F faible",
        "quad_less_power": "Puissance plus faible",
        "matrix_section": "F × G · Matrice de typologie",
        "fig_cross_title": "Profil structurel (F) × Profil bilatéral (G)",
        "fig_cross_xaxis": "Profil bilatéral (G)",
        "fig_cross_yaxis": "Profil structurel (F)",
        "fig_comb_title": "Répartition des profils combinés",
        "axis_countries": "Pays",
        "fig_strat_title": "Répartition des quadrants stratégiques",
        "ranking_section": "Score combiné — Classement",
        "show_top_n_label": "Afficher le top N",
        "view_label": "Vue",
        "fig_rank_title": "Score combiné — {view}  (0.70 × F + 0.30 × G)",
        "axis_combined_score": "Score combiné",
        "summary_section": "Tous les pays — Résumé du scoring combiné",
        "pills_add_columns_label": "Ajouter des colonnes",
        "col_country": "Pays",
        "col_combined_score": "Score combiné",
        "col_combined_profile": "Profil combiné",
        "col_strategic_quadrant": "Quadrant stratégique",
        "col_structural_profile_f": "Profil structurel (F)",
        "col_f_score": "Score F",
        "col_profile_full_period_g": "Profil période complète (G)",
        "col_g_score": "Score G",
        "tab2_highlight_countries_label": "Pays à mettre en évidence",
        "tab2_quadrant_section": "Quadrant stratégique — Rangs F × G",
        "fig_q_title": "Rang F × Rang G — coloré par quadrant stratégique",
        "fig_q_xaxis": "Rang F normalisé (puissance commerciale →)",
        "fig_q_yaxis": "Rang G normalisé (résilience →)",
        "fg_score_section": "Score F vs Score G — Valeurs brutes",
        "median_f_annot": "Médiane F {val}",
        "median_g_annot": "Médiane G {val}",
        "fig_s_title": "Score structurel (F) × Score de risque bilatéral (G) — coloré par profil combiné",
        "fig_s_xaxis": "Score structurel (F) →",
        "fig_s_yaxis": "Score période complète (G) →",
        "weight_sensitivity_section": "Analyse de sensibilité des pondérations",
        "scoring_methodology_expander": "Méthodologie de scoring",
        "scoring_methodology_md": """
**Score combiné** = `0.70 × F_norm_rank + 0.30 × G_norm_rank`

La répartition 70/30 reflète une priorisation délibérée de la **puissance commerciale** (taille, croissance, couverture export) sur la **résilience bilatérale** (concentration des partenaires, dépendance). On peut le lire ainsi : *un pays avec une position structurelle commerciale forte compte plus globalement qu'un pays simplement bien diversifié mais petit.*

**Formule du score G :** `0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`

Le test de stress ci-dessous montre la sensibilité du classement à ce choix.
        """,
        "weight_on_f_label": "Pondération de F (puissance commerciale)",
        "scen_equal": "Pondération égale",
        "scen_slight_f": "Légère priorité à F",
        "scen_baseline": "Référence (70/30)",
        "scen_strong_f": "Forte priorité à F",
        "scen_f_dominant": "F dominant",
        "target_country_separator": "← Pays cible (rang n°{rank})",
        "stress_title": "Classement simulé — {wf} F + {wg} G  ·  score  Δrang vs 70/30",
        "stress_xaxis": "Score combiné simulé",
        "tab3_section_profile_card": "H.5 · Fiche profil pays",
        "lbl_country_profile_card_header": "Fiche profil pays",
        "card_f_header": "F · Structure commerciale",
        "card_f_structural_score": "Score structurel",
        "card_f_profile": "Profil",
        "card_f_static_quadrant": "Quadrant statique",
        "card_f_dynamic_quadrant": "Quadrant dynamique",
        "card_f_meta": "Méta",
        "card_f_export_share": "Part des exportations",
        "card_f_coverage": "Couverture",
        "card_f_cagr_export": "CAGR export",
        "card_f_volatility": "Volatilité",
        "card_g_header": "G · Risque bilatéral",
        "card_g_score_full_period": "Score (période complète)",
        "card_g_profile_full_period": "Profil (période complète)",
        "card_g_profile_recent": "Profil (récent)",
        "card_g_convergence_rate": "Taux de convergence",
        "card_g_score_import": "Score import",
        "card_g_score_export": "Score export",
        "card_g_hhi_import": "HHI import",
        "card_g_hhi_export": "HHI export",
        "card_g_critical_share": "Part critique",
        "card_g_top_partner": "Partenaire principal",
        "card_h_header": "H · Combiné",
        "card_h_quadrant": "Quadrant",
        "card_h_stability": "Stabilité",
        "card_h_stability_score": "Score de stabilité",
        "card_h_profile_drift": "Dérive du profil",
        "card_h_drift_magnitude": "Ampleur de la dérive",
        "dimension_radar_section": "Radar des dimensions",
        "compare_with_label": "Comparer avec",
        "radar_size": "Taille",
        "radar_balance": "Équilibre",
        "radar_growth": "Croissance",
        "radar_stability": "Stabilité",
        "radar_consistency": "Consistance",
        "scorecard_section": "Fiche de score du profil",
        "annual_timeline_section": "Chronologie annuelle du profil",
        "heatmap_logic_profile": "Profil logique",
        "heatmap_score_profile": "Profil score",
        "fig_tl_title": "Profil bilatéral annuel — {country}",
        "no_annual_data": "Aucune donnée annuelle disponible pour {country}.",
    },
}

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
RADAR_LABELS = [t(STRINGS, "radar_size"), t(STRINGS, "radar_balance"), t(STRINGS, "radar_growth"),
                 t(STRINGS, "radar_stability"), t(STRINGS, "radar_consistency")]

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

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ── Global KPI cards ─────────────────────────────────────────
n_champ = (df_combined["combined_profile"] == "Global Champion").sum()
n_comp  = (df_combined["combined_profile"] == "Competitive").sum()
n_vuln  = (df_combined["combined_profile"] == "Vulnerable").sum()
n_frag  = (df_combined["combined_profile"] == "Fragile").sum()

_kc = st.columns(4)
_kc[0].markdown(kpi(t(STRINGS, "kpi_global_champions"), str(n_champ), "teal"),  unsafe_allow_html=True)
_kc[1].markdown(kpi(t(STRINGS, "kpi_competitive"),      str(n_comp),  "blue"),  unsafe_allow_html=True)
_kc[2].markdown(kpi(t(STRINGS, "kpi_vulnerable"),       str(n_vuln),  "amber"), unsafe_allow_html=True)
_kc[3].markdown(kpi(t(STRINGS, "kpi_fragile"),          str(n_frag),  ""),      unsafe_allow_html=True)

thin_divider()

# ── Global country selector ───────────────────────────────────
_sel_col, _ = st.columns([1, 3])
with _sel_col:
    _default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox(t(STRINGS, "select_country"), COUNTRIES,
                           index=_default_idx, key="t6_country")

_row = df_combined[df_combined["country_name"] == country]
if _row.empty:
    st.warning(t(STRINGS, "no_data_country").format(country=country))
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
    t(STRINGS, "tab1_label"),
    t(STRINGS, "tab2_label"),
    t(STRINGS, "tab3_label"),
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
                  letter-spacing:.08em;margin-bottom:4px;">{t(STRINGS, "lbl_target_country")}</div>
      <div style="font-size:22px;font-weight:700;color:#EEF4FB;white-space:nowrap;">
        {country}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_combined_score")}</div>
      <div style="font-size:26px;font-weight:700;color:{_cc};line-height:1.1;">
        {_row['combined_score']:.3f}</div>
      <div style="font-size:11px;color:#6890B0;margin-top:2px;">{t(STRINGS, "lbl_rank").format(pos=_rank_pos, n=N_COUNTRIES)}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_combined_profile")}</div>
      <div style="font-size:14px;font-weight:600;color:{_cc};margin-top:6px;">{_cprof}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_strategic_quadrant")}</div>
      <div style="font-size:14px;font-weight:600;color:{_sq_c};margin-top:6px;">{_strat_q}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_f_rank")}</div>
      <div style="font-size:20px;font-weight:700;color:#60A5FA;margin-top:4px;">
        {_row['f_norm_rank']:.3f}</div>
      <div style="font-size:10px;color:#6890B0;margin-top:2px;">{_sprof}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_g_rank")}</div>
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
        st.markdown(f"""
<div style="background:#162035;border:1px solid #304870;border-radius:10px;padding:.9rem 1rem;">
  <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.6rem;">{t(STRINGS, "lbl_combined_score")}</div>
  <div style="font-size:13px;color:#C4D5E8;margin-bottom:.6rem;">
    <code style="background:#1F3050;padding:2px 6px;border-radius:4px;color:#A78BFA;">
      0.70 × F_norm_rank &nbsp;+&nbsp; 0.30 × G_norm_rank</code>
  </div>
  <div style="font-size:12px;color:#6890B0;margin-bottom:.5rem;">
    {t(STRINGS, "def_combined_score_desc")}
  </div>
  <table style="width:100%;font-size:12px;border-collapse:collapse;">
    <tr style="color:#6890B0;border-bottom:1px solid #304870;">
      <th style="text-align:left;padding:3px 6px;">{t(STRINGS, "def_threshold_header")}</th>
      <th style="text-align:left;padding:3px 6px;">{t(STRINGS, "lbl_combined_profile")}</th>
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
        st.markdown(f"""
<div style="background:#162035;border:1px solid #304870;border-radius:10px;padding:.9rem 1rem;">
  <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.6rem;">{t(STRINGS, "lbl_strategic_quadrant")}</div>
  <div style="font-size:12px;color:#6890B0;margin-bottom:.6rem;">
    {t(STRINGS, "def_strategic_quadrant_desc")}
  </div>
  <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:center;">
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;"></td>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:center;">
        {t(STRINGS, "quad_g_low")}<br><span style="color:#EF4444;font-size:10px;">{t(STRINGS, "quad_more_risk")}</span></td>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:center;">
        {t(STRINGS, "quad_g_high")}<br><span style="color:#10B981;font-size:10px;">{t(STRINGS, "quad_resilient")}</span></td>
    </tr>
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:right;">
        {t(STRINGS, "quad_f_high")}<br><span style="color:#60A5FA;font-size:10px;">{t(STRINGS, "quad_more_power")}</span></td>
      <td style="padding:6px;background:#231800;border-radius:6px;
                 font-weight:600;color:#F59E0B;text-align:center;">Strong but Exposed</td>
      <td style="padding:6px;background:#082E18;border-radius:6px;
                 font-weight:600;color:#10B981;text-align:center;">Strong & Resilient</td>
    </tr>
    <tr>
      <td style="padding:4px;color:#6890B0;font-size:11px;text-align:right;">
        {t(STRINGS, "quad_f_low")}<br><span style="color:#6B7280;font-size:10px;">{t(STRINGS, "quad_less_power")}</span></td>
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
    section_label(t(STRINGS, "matrix_section"))

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
            title=t(STRINGS, "fig_cross_title"),
            xaxis_title=t(STRINGS, "fig_cross_xaxis"),
            yaxis_title=t(STRINGS, "fig_cross_yaxis"),
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
            title=t(STRINGS, "fig_comb_title"),
            xaxis_title=t(STRINGS, "axis_countries"),
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
            title=t(STRINGS, "fig_strat_title"),
            xaxis_title=t(STRINGS, "axis_countries"),
            height=175,
            margin=dict(l=10, r=50, t=50, b=30),
        ))
        st.plotly_chart(fig_strat, width='stretch')

    thin_divider()

    # ── 4. Ranking bar ────────────────────────────────────────
    section_label(t(STRINGS, "ranking_section"))

    _rk_col, _rk_bar = st.columns([1, 3])
    with _rk_col:
        n_top = st.selectbox(t(STRINGS, "show_top_n_label"), [5, 10, 20, 30, 50], index=1, key="t1_n")
        # NOTE: option values ("Top performers" / "Bottom performers") are left untranslated
        # because the code below matches on the literal substring "Top" in view.
        view  = st.radio(t(STRINGS, "view_label"), ["Top performers", "Bottom performers"], key="t1_view")

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
            title=t(STRINGS, "fig_rank_title").format(view=view),
            xaxis_title=t(STRINGS, "axis_combined_score"),
            height=max(380, n_top * 22 + 80),
        )
        st.plotly_chart(fig_rank, width='stretch')

    thin_divider()

    # ── 5. Summary table — all countries ─────────────────────
    section_label(t(STRINGS, "summary_section"))

    # NOTE: the pill option "F & G Detail" is left untranslated because the code
    # below matches on this literal string to decide which columns to show.
    _sc_tog = st.pills(
        t(STRINGS, "pills_add_columns_label"), ["F & G Detail"],
        selection_mode="multi", default=[], key="t1_sc_tog",
    )
    _show_sc = "F & G Detail" in (_sc_tog or [])

    # Column header labels — built once via t() so every downstream lookup
    # (styling subsets, column widths, number formats) stays in sync with the
    # currently selected language.
    _L_COUNTRY   = t(STRINGS, "col_country")
    _L_SCORE     = t(STRINGS, "col_combined_score")
    _L_PROFILE   = t(STRINGS, "col_combined_profile")
    _L_QUADRANT  = t(STRINGS, "col_strategic_quadrant")
    _L_STRUCT_F  = t(STRINGS, "col_structural_profile_f")
    _L_F_SCORE   = t(STRINGS, "col_f_score")
    _L_PROFILE_G = t(STRINGS, "col_profile_full_period_g")
    _L_G_SCORE   = t(STRINGS, "col_g_score")

    # Base: combined view only (4 cols)
    # Toggle ON: interleaved F→combined→G decomposition
    if not _show_sc:
        _all_sel = [
            ("country_name",               _L_COUNTRY),
            ("combined_score",             _L_SCORE),
            ("combined_profile",           _L_PROFILE),
            ("strategic_quadrant_combined",_L_QUADRANT),
        ]
    else:
        _all_sel = [
            ("country_name",               _L_COUNTRY),
            ("combined_score",             _L_SCORE),
            ("combined_profile",           _L_PROFILE),
            ("strategic_quadrant_combined",_L_QUADRANT),
            ("structural_profile",         _L_STRUCT_F),
            ("f_norm_rank",                _L_F_SCORE),
            ("profile_full_periode",       _L_PROFILE_G),
            ("g_norm_rank",                _L_G_SCORE),
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
        if row[_L_COUNTRY] == country:
            return ["background-color:rgba(16,185,129,0.10);"] * len(row)
        return [""] * len(row)

    _st = _df_tbl.style
    _cols = _df_tbl.columns.tolist()

    _st = _st.apply(_highlight_target, axis=1)
    for _col, _fn in [(_L_PROFILE,   _clr_combined),
                       (_L_QUADRANT, _clr_quadrant),
                       (_L_PROFILE_G,_clr_fp),
                       (_L_STRUCT_F, _clr_struct)]:
        if _col in _cols:
            _st = _st.map(_fn, subset=[_col])

    _num_fmt = {c: "{:.3f}" for c in [_L_SCORE, _L_F_SCORE, _L_G_SCORE]
                if c in _cols}
    if _num_fmt:
        _st = _st.format(_num_fmt)

    _COL_W = {
        _L_COUNTRY: 90, _L_SCORE: 90, _L_PROFILE: 110,
        _L_QUADRANT: 120, _L_STRUCT_F: 100,
        _L_PROFILE_G: 125, _L_F_SCORE: 80, _L_G_SCORE: 80,
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
        t(STRINGS, "tab2_highlight_countries_label"),
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
    section_label(t(STRINGS, "tab2_quadrant_section"))

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
        title=t(STRINGS, "fig_q_title"),
        xaxis_title=t(STRINGS, "fig_q_xaxis"),
        yaxis_title=t(STRINGS, "fig_q_yaxis"),
        height=540,
    ))
    st.plotly_chart(fig_q, width='stretch')

    thin_divider()

    # ── F score vs G score (raw values) ──────────────────────
    section_label(t(STRINGS, "fg_score_section"))

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
                    annotation_text=t(STRINGS, "median_f_annot").format(val=f"{_f_med:.2f}"),
                    annotation_font=dict(color="#475569", size=10))
    fig_s.add_hline(y=_g_med, line_dash="dash", line_color="#475569",
                    annotation_text=t(STRINGS, "median_g_annot").format(val=f"{_g_med:.2f}"),
                    annotation_font=dict(color="#475569", size=10))

    fig_s.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "fig_s_title"),
        xaxis_title=t(STRINGS, "fig_s_xaxis"),
        yaxis_title=t(STRINGS, "fig_s_yaxis"),
        height=540,
    )
    st.plotly_chart(fig_s, width='stretch')

    # ── Weight Sensitivity Analysis ───────────────────────────
    thin_divider()
    section_label(t(STRINGS, "weight_sensitivity_section"))

    with st.expander(t(STRINGS, "scoring_methodology_expander")):
        st.markdown(t(STRINGS, "scoring_methodology_md"))

    _wf = st.slider(
        t(STRINGS, "weight_on_f_label"), min_value=0.50, max_value=0.90,
        value=0.70, step=0.05, key="stress_wf", format="%.2f",
    )
    _wg = round(1.0 - _wf, 2)

    _scen_lbl = {0.50: t(STRINGS, "scen_equal"), 0.60: t(STRINGS, "scen_slight_f"),
                 0.70: t(STRINGS, "scen_baseline"), 0.80: t(STRINGS, "scen_strong_f"),
                 0.90: t(STRINGS, "scen_f_dominant")}.get(_wf, f"{_wf:.0%} F")
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
            annotation_text=t(STRINGS, "target_country_separator").format(
                rank=int(_df_st[_df_st['country_name']==country]['rank_sim'].iloc[0])),
            annotation_font=dict(color="#475569", size=10),
            annotation_position="bottom right",
        )

    fig_st.update_layout(**_lay(
        title=t(STRINGS, "stress_title").format(wf=f"{_wf:.0%}", wg=f"{_wg:.0%}"),
        xaxis_title=t(STRINGS, "stress_xaxis"),
        height=max(560, _n_bars * 28 + 80),
        margin=dict(l=130, r=130, t=50, b=40),
    ))
    st.plotly_chart(fig_st, width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 3 — COUNTRY PROFILE CARD
# ══════════════════════════════════════════════════════════════════
with tab3:
    section_label(t(STRINGS, "tab3_section_profile_card"))

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
                  letter-spacing:.1em;margin-bottom:4px;">{t(STRINGS, "lbl_country_profile_card_header")}</div>
      <div style="font-size:28px;font-weight:700;color:#EEF4FB;">{country}</div>
      <div style="font-size:14px;color:{_cc};font-weight:600;margin-top:4px;">{_cprof}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#6890B0;text-transform:uppercase;">{t(STRINGS, "lbl_combined_score")}</div>
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
      {t(STRINGS, "card_f_header")}</div>
    <div style="font-size:13px;color:#FBBf24;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_structural_score")}</span> &nbsp;→&nbsp;
           {_row.get("structural_score", 0):.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_profile")}</span> &nbsp;→&nbsp;
           <span style="color:{_sc};font-weight:600;">{_sprof}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_static_quadrant")}</span> &nbsp;→&nbsp;
           <span style="color:{_static_q_c};font-weight:600;">{_static_q}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_dynamic_quadrant")}</span> &nbsp;→&nbsp;
           <span style="color:{_dynamic_q_c};font-weight:600;">{_dynamic_q}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "lbl_strategic_quadrant")}</span> &nbsp;→&nbsp;
           <span style="color:{_strategic_qf_c};font-weight:600;">{_strategic_qf}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_meta")}</span> &nbsp;→&nbsp;
           {_row.get("meta_profile","—")}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_export_share")}</span> &nbsp;→&nbsp;
           {_row["export_share_structural"]:.2f}%</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_coverage")}</span> &nbsp;→&nbsp;
           {_row["coverage_ratio_structural"]:.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_cagr_export")}</span> &nbsp;→&nbsp;
           {_row["cagr_export"]*100:.1f}%</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_f_volatility")}</span> &nbsp;→&nbsp;
           {_row["volatility_export"]:.2f}</div>
    </div>
  </div>

  <!-- G · Bilateral Risk -->
  <div style="background:#0F2240;border:1.5px solid #3B82F6;
              border-radius:12px;padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:11px;font-weight:700;color:#93C5FD;
                text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">
      {t(STRINGS, "card_g_header")}</div>
    <div style="font-size:13px;color:#BAD7F8;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_score_full_period")}</span> &nbsp;→&nbsp;
           {_row.get("score_full_periode", 0):.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_profile_full_period")}</span> &nbsp;→&nbsp;
           <span style="color:{_bc};font-weight:600;">{_bprof}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_profile_recent")}</span> &nbsp;→&nbsp;
           <span style="color:{_pr_c};font-weight:600;">{_prof_rec}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_convergence_rate")}</span> &nbsp;→&nbsp; {_conv_str}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_score_import")}</span> &nbsp;→&nbsp;
           {_row.get("score_import", 0):.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_score_export")}</span> &nbsp;→&nbsp;
           {_row.get("score_export", 0):.3f}</div>
      <div><span style="color:#6890B0;">BDIS</span> &nbsp;→&nbsp; {_row["bdis"]:.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_hhi_import")}</span> &nbsp;→&nbsp;
           {_row["hhi_import"]:.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_hhi_export")}</span> &nbsp;→&nbsp;
           {_row["hhi_export"]:.3f}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_critical_share")}</span> &nbsp;→&nbsp;
           {_row["critical_share"]:.1%}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_g_top_partner")}</span> &nbsp;→&nbsp;
           {_row.get("top_partner","—")}</div>
    </div>
  </div>

  <!-- H · Combined -->
  <div style="background:#082E18;border:1.5px solid #10B981;
              border-radius:12px;padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:11px;font-weight:700;color:#34D399;
                text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">
      {t(STRINGS, "card_h_header")}</div>
    <div style="font-size:13px;color:#6EE7B7;line-height:1.75;flex:1;">
      <div><span style="color:#6890B0;">{t(STRINGS, "lbl_combined_score")}</span> &nbsp;→&nbsp;
           <span style="color:{_cc};font-weight:700;">{_row["combined_score"]:.3f}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_h_quadrant")}</span> &nbsp;→&nbsp;
           <span style="color:{_sq_c};font-weight:600;">{_strat_q}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_h_stability")}</span> &nbsp;→&nbsp;
           <span style="color:{_stab_c};font-weight:600;">{_stab}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_h_stability_score")}</span> &nbsp;→&nbsp; {_stab_score_str}</div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_h_profile_drift")}</span> &nbsp;→&nbsp;
           <span style="color:{_drift_c};font-weight:600;">{_drift}</span></div>
      <div><span style="color:#6890B0;">{t(STRINGS, "card_h_drift_magnitude")}</span> &nbsp;→&nbsp; {_dm_str}</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.75rem;'></div>", unsafe_allow_html=True)

    # ── Radar + Scorecard ─────────────────────────────────────
    _cr, _cs_col = st.columns(2)

    with _cr:
        section_label(t(STRINGS, "dimension_radar_section"))
        _radar_default = [c for c in ["Australia", "Belgium"] if c in COUNTRIES and c != country]
        if st.session_state.get("_radar_prev_country") != country:
            st.session_state["card_compare"] = _radar_default
            st.session_state["_radar_prev_country"] = country
        compare_c = st.multiselect(
            t(STRINGS, "compare_with_label"),
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
        section_label(t(STRINGS, "scorecard_section"))

        _DIMS = [
            (t(STRINGS, "radar_size"),        "size_profile"),
            (t(STRINGS, "radar_balance"),     "balance_profile"),
            (t(STRINGS, "radar_growth"),      "growth_profile"),
            (t(STRINGS, "radar_stability"),   "stability_profile"),
            (t(STRINGS, "radar_consistency"), "consistency_profile"),
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
      <div style="font-size:11px;color:#6890B0;">{t(STRINGS, "card_f_structural_score")}</div>
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
    section_label(t(STRINGS, "annual_timeline_section"))

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
            y=[t(STRINGS, "heatmap_logic_profile"), t(STRINGS, "heatmap_score_profile")],
            colorscale=_TL_COLORSCALE, zmin=0, zmax=3,
            text=[_lg_lbl, _sc_lbl],
            texttemplate="%{text}",
            textfont=dict(size=9, color="#EEF4FB"),
            showscale=False,
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Profile: %{text}<extra></extra>",
            xgap=2, ygap=4,
        ))
        fig_tl.update_layout(**_lay(
            title=t(STRINGS, "fig_tl_title").format(country=country),
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
        st.info(t(STRINGS, "no_annual_data").format(country=country))
