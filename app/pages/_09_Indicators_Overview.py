"""
Indicators Analysis — I1 · Overview & Methodology
Covers: B. Executive Summary + C. Data & Methodology + E. Global Snapshot
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Indicators Analysis · Page 1 of 4",
        "title": "Overview & Methodology",
        "caption": "16 macroeconomic indicators · ~180 countries · 2015–2024 · World Bank",
        "kpi_countries": "Countries",
        "kpi_indicators": "Indicators",
        "kpi_period": "Period",
        "kpi_coverage": "Coverage",
        "exec_summary": "Executive Summary",
        "lbl_global": "Global",
        "txt_global": "Economic indicators are **highly concentrated** — a small number of countries "
                       "dominates absolute levels. Cross-indicator stability varies widely between "
                       "level and flow indicators.",
        "lbl_world_profile": "World Profile",
        "txt_world_profile": "The World aggregate **masks heterogeneity** — global trends are driven by a "
                              "handful of large economies whose cycles dominate the average.",
        "lbl_country_profile": "Country Profile",
        "txt_country_profile": "Country-level profiles reveal **structural divergences invisible** at the global "
                                "level — same indicator, very different trajectories depending on size and "
                                "development stage.",
        "lbl_comparisons": "Comparisons",
        "txt_comparisons": "Cross-country comparisons on **index-100 and z-score** reveal convergence and "
                            "divergence patterns that raw values cannot capture.",
        "analytical_framework": "Analytical Framework",
        "card_e_title": "E · Global Vision",
        "card_e_1": "① Trend — global average",
        "card_e_2": "② Distribution cross-country",
        "card_e_3": "③ Stability (CV + rolling)",
        "card_e_4": "④ Concentration (Gini)",
        "card_e_5": "⑤ Growth dynamics",
        "card_e_6": "⑥ Top & worst countries",
        "card_fg_title": "F–I · Deep Dive",
        "card_fg_1": "F · World aggregate profile",
        "card_fg_2": "G · Country-level profile",
        "card_fg_3": "H · Cross-country comparison",
        "card_fg_4": "I · Growth dynamics",
        "card_fg_sub": "CAGR · volatility · shocks<br>\n          index-100 · z-score",
        "card_hln_title": "J–N · Advanced",
        "card_hln_1": "J · Correlation & PCA",
        "card_hln_2": "K · Convergence (σ & β)",
        "card_hln_3": "L · Clustering (KMeans)",
        "card_hln_4": "M · Forecasting (H-W)",
        "card_hln_5": "N · Multi-indicator scoring",
        "card_hln_sub": "Econ Health · Performance<br>\n          Competitiveness",
        "data_scope": "Data Scope",
        "dataset_title": "Dataset",
        "ds_source_lbl": "Source",
        "ds_source_val": "World Bank — Gold layer",
        "ds_period_lbl": "Period",
        "ds_period_val": "2015–2024 (10 years)",
        "ds_countries_lbl": "Countries",
        "ds_countries_val": "~180 (COUNTRY entities only)",
        "ds_indicators_lbl": "Indicators",
        "ds_indicators_val": "16 macroeconomic",
        "ds_coverage_lbl": "Coverage threshold",
        "ds_coverage_val": "70% min non-null per indicator",
        "ds_pipeline_lbl": "Pipeline",
        "ds_pipeline_val": "WB API → ETL Bronze/Silver/Gold",
        "fam_title": "Indicator Families",
        "fam_1_lbl": "Activity & Growth",
        "fam_1_val": "GDP · GDP per capita",
        "fam_2_lbl": "Prices & Monetary",
        "fam_2_val": "Inflation · Real interest rate",
        "fam_3_lbl": "Labor market",
        "fam_3_val": "Unemployment",
        "fam_4_lbl": "Public finances",
        "fam_4_val": "Expense (% GDP)",
        "fam_5_lbl": "External sector",
        "fam_5_val": "Current account · Trade · Exports · Imports",
        "snapshot_title": "E · Global Snapshot — Indicator Ranking",
        "conc_title": "Concentration Ranking — Mean Gini (higher = more concentrated)",
        "conc_xaxis": "Mean Gini",
        "conc_annot_high": "High (>0.8)",
        "stab_title": "Stability Ranking — Global CV% (lower = more stable)",
        "stab_xaxis": "Coefficient of Variation (%)",
        "key_findings": "Key findings",
        "finding_most_concentrated": "<strong>Most concentrated indicator</strong>: {name} "
                                      "— a handful of countries dominates absolute levels",
        "finding_most_stable": "<strong>Most stable indicator</strong>: {name} "
                                "— low cross-country variation over the decade",
        "finding_most_volatile": "<strong>Most volatile indicator</strong>: {name} "
                                  "— high sensitivity to macro shocks (COVID, commodity cycles)",
        "design_choices": "Design Choices",
        "finding_parametric": "<strong>Parametric design</strong>: all analyses run on a single "
                               "<code>indicator</code> parameter — changing it re-runs the full section "
                               "on any indicator without code changes",
        "finding_country_only": "<strong>COUNTRY entities only</strong>: World and regional aggregates "
                                 "are excluded from cross-country analyses (correlation, PCA, clustering, "
                                 "convergence) to preserve statistical validity",
        "finding_double_reading": "<strong>Double reading in scoring (Section N)</strong>: each score is "
                                   "computed twice — on indicator <em>values</em> (current position) and on "
                                   "<em>growth scores</em> (momentum). Same indicators, same weights, different "
                                   "data — this distinguishes structural position from directional trajectory",
        "finding_median": "<strong>Per-country median for full-period aggregation</strong>: "
                           "robust to the 2020 COVID outlier — consistent with the Trade notebook strategy",
    },
    "fr": {
        "eyebrow": "Analyse des indicateurs · Page 1 sur 4",
        "title": "Aperçu & méthodologie",
        "caption": "16 indicateurs macroéconomiques · ~180 pays · 2015–2024 · World Bank",
        "kpi_countries": "Pays",
        "kpi_indicators": "Indicateurs",
        "kpi_period": "Période",
        "kpi_coverage": "Couverture",
        "exec_summary": "Résumé exécutif",
        "lbl_global": "Mondial",
        "txt_global": "Les indicateurs économiques sont **fortement concentrés** — un petit nombre de pays "
                       "domine les niveaux absolus. La stabilité varie fortement entre indicateurs de niveau "
                       "et de flux.",
        "lbl_world_profile": "Profil mondial",
        "txt_world_profile": "L'agrégat Monde **masque l'hétérogénéité** — les tendances globales sont "
                              "dictées par une poignée de grandes économies dont les cycles dominent la moyenne.",
        "lbl_country_profile": "Profil pays",
        "txt_country_profile": "Les profils par pays révèlent des **divergences structurelles invisibles** "
                                "au niveau global — même indicateur, trajectoires très différentes selon la "
                                "taille et le niveau de développement.",
        "lbl_comparisons": "Comparaisons",
        "txt_comparisons": "Les comparaisons inter-pays en **indice-100 et z-score** révèlent des schémas de "
                            "convergence et de divergence que les valeurs brutes ne peuvent pas capturer.",
        "analytical_framework": "Cadre analytique",
        "card_e_title": "E · Vision globale",
        "card_e_1": "① Tendance — moyenne mondiale",
        "card_e_2": "② Distribution inter-pays",
        "card_e_3": "③ Stabilité (CV + glissant)",
        "card_e_4": "④ Concentration (Gini)",
        "card_e_5": "⑤ Dynamique de croissance",
        "card_e_6": "⑥ Meilleurs & pires pays",
        "card_fg_title": "F–I · Analyse approfondie",
        "card_fg_1": "F · Profil agrégé mondial",
        "card_fg_2": "G · Profil par pays",
        "card_fg_3": "H · Comparaison inter-pays",
        "card_fg_4": "I · Dynamique de croissance",
        "card_fg_sub": "CAGR · volatilité · chocs<br>\n          indice-100 · z-score",
        "card_hln_title": "J–N · Analyse avancée",
        "card_hln_1": "J · Corrélation & ACP",
        "card_hln_2": "K · Convergence (σ & β)",
        "card_hln_3": "L · Clustering (KMeans)",
        "card_hln_4": "M · Prévision (H-W)",
        "card_hln_5": "N · Scoring multi-indicateurs",
        "card_hln_sub": "Santé économique · Performance<br>\n          Compétitivité",
        "data_scope": "Périmètre des données",
        "dataset_title": "Jeu de données",
        "ds_source_lbl": "Source",
        "ds_source_val": "World Bank — couche Gold",
        "ds_period_lbl": "Période",
        "ds_period_val": "2015–2024 (10 ans)",
        "ds_countries_lbl": "Pays",
        "ds_countries_val": "~180 (entités COUNTRY uniquement)",
        "ds_indicators_lbl": "Indicateurs",
        "ds_indicators_val": "16 macroéconomiques",
        "ds_coverage_lbl": "Seuil de couverture",
        "ds_coverage_val": "70% min de valeurs non nulles par indicateur",
        "ds_pipeline_lbl": "Pipeline",
        "ds_pipeline_val": "API WB → ETL Bronze/Silver/Gold",
        "fam_title": "Familles d'indicateurs",
        "fam_1_lbl": "Activité & croissance",
        "fam_1_val": "PIB · PIB par habitant",
        "fam_2_lbl": "Prix & monétaire",
        "fam_2_val": "Inflation · Taux d'intérêt réel",
        "fam_3_lbl": "Marché du travail",
        "fam_3_val": "Chômage",
        "fam_4_lbl": "Finances publiques",
        "fam_4_val": "Dépenses (% PIB)",
        "fam_5_lbl": "Secteur extérieur",
        "fam_5_val": "Compte courant · Commerce · Exportations · Importations",
        "snapshot_title": "E · Aperçu global — Classement des indicateurs",
        "conc_title": "Classement par concentration — Gini moyen (plus élevé = plus concentré)",
        "conc_xaxis": "Gini moyen",
        "conc_annot_high": "Élevé (>0.8)",
        "stab_title": "Classement par stabilité — CV global % (plus bas = plus stable)",
        "stab_xaxis": "Coefficient de variation (%)",
        "key_findings": "Principaux constats",
        "finding_most_concentrated": "<strong>Indicateur le plus concentré</strong> : {name} "
                                      "— une poignée de pays domine les niveaux absolus",
        "finding_most_stable": "<strong>Indicateur le plus stable</strong> : {name} "
                                "— faible variation inter-pays sur la décennie",
        "finding_most_volatile": "<strong>Indicateur le plus volatil</strong> : {name} "
                                  "— forte sensibilité aux chocs macro (COVID, cycles des matières premières)",
        "design_choices": "Choix de conception",
        "finding_parametric": "<strong>Conception paramétrique</strong> : toutes les analyses s'exécutent sur "
                               "un seul paramètre <code>indicator</code> — le changer relance toute la section "
                               "sur n'importe quel indicateur, sans modifier le code",
        "finding_country_only": "<strong>Entités COUNTRY uniquement</strong> : le Monde et les agrégats "
                                 "régionaux sont exclus des analyses inter-pays (corrélation, ACP, clustering, "
                                 "convergence) pour préserver la validité statistique",
        "finding_double_reading": "<strong>Double lecture du scoring (Section N)</strong> : chaque score est "
                                   "calculé deux fois — sur les <em>valeurs</em> de l'indicateur (position "
                                   "actuelle) et sur les <em>scores de croissance</em> (momentum). Mêmes "
                                   "indicateurs, mêmes pondérations, données différentes — ceci distingue la "
                                   "position structurelle de la trajectoire directionnelle",
        "finding_median": "<strong>Médiane par pays pour l'agrégation sur la période complète</strong> : "
                           "robuste à la valeur aberrante COVID de 2020 — cohérent avec la stratégie du "
                           "notebook Trade",
    },
}

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_conc  = pd.read_csv(IND_FILES / "concentration.csv")
    df_stab  = pd.read_csv(IND_FILES / "stability.csv")
    df_ind   = pd.read_csv(IND_FILES / "df_indicators.csv")
    df_cv    = pd.read_csv(IND_FILES / "df_cv.csv")
    return df_conc, df_stab, df_ind, df_cv

df_conc, df_stab, df_ind, df_cv = load_data()

INDICATORS = sorted(df_ind["indicator_name"].dropna().unique().tolist())
N_COUNTRIES = df_ind[df_ind["entity_type"] == "COUNTRY"]["country_name"].nunique()
N_YEARS     = df_ind["year"].nunique()

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

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

_yr_min  = int(df_ind["year"].min())
_yr_max  = int(df_ind["year"].max())
_n_obs   = int(df_ind[df_ind["entity_type"] == "COUNTRY"]["value"].notna().sum())
_cov_pct = _n_obs / (N_COUNTRIES * len(INDICATORS) * N_YEARS) * 100

cols = st.columns(4)
for col, (label, value, color) in zip(cols, [
    (t(STRINGS, "kpi_countries"),  str(N_COUNTRIES),            "teal"),
    (t(STRINGS, "kpi_indicators"), str(len(INDICATORS)),         "blue"),
    (t(STRINGS, "kpi_period"),     f"{_yr_min}–{_yr_max}",       "amber"),
    (t(STRINGS, "kpi_coverage"),   f"{_cov_pct:.0f}%",           ""),
]):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# B. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "exec_summary"))

results = [
    (t(STRINGS, "lbl_global"), t(STRINGS, "txt_global")),
    (t(STRINGS, "lbl_world_profile"), t(STRINGS, "txt_world_profile")),
    (t(STRINGS, "lbl_country_profile"), t(STRINGS, "txt_country_profile")),
    (t(STRINGS, "lbl_comparisons"), t(STRINGS, "txt_comparisons")),
]

for label, text in results:
    col1, col2 = st.columns([1, 3])
    col1.markdown(
        f"<div style='font-size:13px;font-weight:600;color:#FCD34D;"
        f"padding:.4rem 0;'>{label}</div>",
        unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size:14px;color:#C4D5E8;padding:.4rem 0;'>{text}</div>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #304870;margin:.1rem 0;'>",
                unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# C. ANALYTICAL FRAMEWORK
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "analytical_framework"))

col_e, col_fg, col_hln = st.columns(3)

with col_e:
    st.markdown(f"""
    <div style="background:#231800;border:1.5px solid #D97706;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#FCD34D;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">{t(STRINGS, "card_e_title")}</div>
      <div style="font-size:13px;color:#FBBf24;line-height:1.9;opacity:.9;">
        <div>{t(STRINGS, "card_e_1")}</div>
        <div>{t(STRINGS, "card_e_2")}</div>
        <div>{t(STRINGS, "card_e_3")}</div>
        <div>{t(STRINGS, "card_e_4")}</div>
        <div>{t(STRINGS, "card_e_5")}</div>
        <div>{t(STRINGS, "card_e_6")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_fg:
    st.markdown(f"""
    <div style="background:#0F2240;border:1.5px solid #3B82F6;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#93C5FD;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">{t(STRINGS, "card_fg_title")}</div>
      <div style="font-size:13px;color:#BAD7F8;line-height:1.9;opacity:.9;">
        <div>{t(STRINGS, "card_fg_1")}</div>
        <div>{t(STRINGS, "card_fg_2")}</div>
        <div>{t(STRINGS, "card_fg_3")}</div>
        <div>{t(STRINGS, "card_fg_4")}</div>
        <div style="margin-top:.5rem;font-size:12px;color:#1E3A8A;">
          {t(STRINGS, "card_fg_sub")}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_hln:
    st.markdown(f"""
    <div style="background:#082E18;border:1.5px solid #10B981;border-radius:14px;
                padding:1rem;min-height:260px;">
      <div style="font-size:11px;font-weight:700;color:#34D399;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:.85rem;">{t(STRINGS, "card_hln_title")}</div>
      <div style="font-size:13px;color:#6EE7B7;line-height:1.9;opacity:.9;">
        <div>{t(STRINGS, "card_hln_1")}</div>
        <div>{t(STRINGS, "card_hln_2")}</div>
        <div>{t(STRINGS, "card_hln_3")}</div>
        <div>{t(STRINGS, "card_hln_4")}</div>
        <div>{t(STRINGS, "card_hln_5")}</div>
        <div style="margin-top:.5rem;font-size:12px;color:#065F46;">
          {t(STRINGS, "card_hln_sub")}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# DATA SCOPE
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "data_scope"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;min-height:260px;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">{t(STRINGS, "dataset_title")}</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:2.1;">
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_source_lbl")}</span>
             {t(STRINGS, "ds_source_val")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_period_lbl")}</span>
             {t(STRINGS, "ds_period_val")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_countries_lbl")}</span>
             {t(STRINGS, "ds_countries_val")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_indicators_lbl")}</span>
             {t(STRINGS, "ds_indicators_val")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_coverage_lbl")}</span>
             {t(STRINGS, "ds_coverage_val")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:130px;
                          display:inline-block;">{t(STRINGS, "ds_pipeline_lbl")}</span>
             {t(STRINGS, "ds_pipeline_val")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Indicator families
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;min-height:260px;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">{t(STRINGS, "fam_title")}</div>
      <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <div><span style="color:#10B981;font-weight:600;">{t(STRINGS, "fam_1_lbl")}</span>
             &nbsp;— {t(STRINGS, "fam_1_val")}</div>
        <div><span style="color:#60A5FA;font-weight:600;">{t(STRINGS, "fam_2_lbl")}</span>
             &nbsp;— {t(STRINGS, "fam_2_val")}</div>
        <div><span style="color:#F59E0B;font-weight:600;">{t(STRINGS, "fam_3_lbl")}</span>
             &nbsp;— {t(STRINGS, "fam_3_val")}</div>
        <div><span style="color:#A78BFA;font-weight:600;">{t(STRINGS, "fam_4_lbl")}</span>
             &nbsp;— {t(STRINGS, "fam_4_val")}</div>
        <div><span style="color:#F472B6;font-weight:600;">{t(STRINGS, "fam_5_lbl")}</span>
             &nbsp;— {t(STRINGS, "fam_5_val")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# E. GLOBAL SNAPSHOT — Concentration & Stability ranking
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "snapshot_title"))

col1, col2 = st.columns(2)

with col1:
    # Concentration ranking (Gini)
    df_c_sorted = df_conc.sort_values("mean_gini", ascending=True).copy()
    # Short names
    df_c_sorted["short_name"] = df_c_sorted["indicator_name"].apply(
        lambda x: x[:40] + "…" if len(x) > 40 else x)

    colors_c = ["#EF4444" if v > 0.8
                else "#F59E0B" if v > 0.6
                else "#60A5FA" if v > 0.4
                else "#10B981"
                for v in df_c_sorted["mean_gini"]]

    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        x=df_c_sorted["mean_gini"],
        y=df_c_sorted["short_name"],
        orientation="h",
        marker_color=colors_c,
        text=df_c_sorted["mean_gini"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        customdata=df_c_sorted[["trend_gini","last_gini"]],
        hovertemplate="<b>%{y}</b><br>Mean Gini: %{x:.3f}<br>"
                      "Last Gini: %{customdata[1]:.3f}<br>"
                      "Trend: %{customdata[0]:+.3f}<extra></extra>",
        opacity=0.85,
    ))
    fig_conc.add_vline(x=0.8, line_dash="dash", line_color="#EF4444",
                       annotation_text=t(STRINGS, "conc_annot_high"),
                       annotation_font_color="#EF4444")
    fig_conc.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "conc_title"),
        xaxis_title=t(STRINGS, "conc_xaxis"),
        height=520,
    )
    st.plotly_chart(fig_conc, width='stretch')

with col2:
    # Stability ranking (CV — lower = more stable)
    df_s_sorted = df_stab.sort_values("global_cv", ascending=True).copy()
    df_s_sorted["short_name"] = df_s_sorted["indicator_name"].apply(
        lambda x: x[:40] + "…" if len(x) > 40 else x)

    colors_s = ["#10B981" if v < 10
                else "#60A5FA" if v < 20
                else "#F59E0B" if v < 40
                else "#EF4444"
                for v in df_s_sorted["global_cv"]]

    fig_stab = go.Figure()
    fig_stab.add_trace(go.Bar(
        x=df_s_sorted["global_cv"],
        y=df_s_sorted["short_name"],
        orientation="h",
        marker_color=colors_s,
        text=df_s_sorted["global_cv"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Global CV: %{x:.1f}%<extra></extra>",
    ))
    fig_stab.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "stab_title"),
        xaxis_title=t(STRINGS, "stab_xaxis"),
        height=520,
    )
    st.plotly_chart(fig_stab, width='stretch')

with st.expander(t(STRINGS, "key_findings")):
    most_conc  = df_conc.nlargest(1,"mean_gini")["indicator_name"].iloc[0]
    most_stab  = df_stab.nsmallest(1,"global_cv")["indicator_name"].iloc[0]
    most_volat = df_stab.nlargest(1,"global_cv")["indicator_name"].iloc[0]
    finding(t(STRINGS, "finding_most_concentrated").format(name=most_conc))
    finding(t(STRINGS, "finding_most_stable").format(name=most_stab))
    finding(t(STRINGS, "finding_most_volatile").format(name=most_volat))

thin_divider()

# ════════════════════════════════════════════════════════════════
# DESIGN CHOICES
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "design_choices"))

finding(t(STRINGS, "finding_parametric"))
finding(t(STRINGS, "finding_country_only"))
finding(t(STRINGS, "finding_double_reading"))
finding(t(STRINGS, "finding_median"))