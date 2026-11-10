"""
Trade Analysis — T1 · Overview & Methodology
Covers: B. Executive Summary + C. Data & Methodology
Static page — no selectors needed
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "App"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project paths ─────────────────────────────────────────────
FIGURES_DIR  = ROOT / "Analysis" / "Figures"
TRADE_FIGS   = FIGURES_DIR / "Trade"
SCORE_GUIDE  = FIGURES_DIR / "score_reference_guide.png"
GOLD_DIR     = ROOT / "data" / "gold"

import pandas as pd
import streamlit as st
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 1 of 6",
        "title": "Overview & Methodology",
        "caption_header": "{n} countries · 10 years · 3 analytical dimensions · UN Comtrade · 2015–2024",
        "kpi_countries": "Countries",
        "kpi_years": "Years",
        "kpi_dimensions": "Dimensions",
        "kpi_gold_rows": "Gold rows",
        "executive_summary": "Executive Summary",
        "label_global_trade": "Global Trade",
        "label_covid_impact": "COVID Impact",
        "label_trade_power": "Trade Power",
        "label_bilateral_risk": "Bilateral Risk",
        "label_combined_score": "Combined Score",
        "finding_global_trade": "Top 10 countries control **{min}–{max}% of global exports** — highly concentrated and structurally stable over the decade",
        "finding_covid_impact": "2020 did not reshape bilateral architectures — rank correlations of country positions **remained above 0.995** throughout 2015–2024, with maximum structural shift in **{year}**",
        "finding_trade_power": "**{n} countries** hold a Global Leader profile in trade structure — led by China, USA, Germany, Japan",
        "finding_bilateral_risk": "**{pct}% of countries** ({n}) face **Critical Dependency** bilaterally — predominantly small commodity-dependent economies",
        "finding_combined_score": "**{n} Global Champions** combine top trade power with strong bilateral resilience",
        "main_conclusion_label": "Main Conclusion",
        "main_conclusion_body": (
            "Trade strength and trade resilience are complementary but independent. "
            "The most strategically valuable countries are those that combine both — "
            "a rare profile held by <strong style=\"color:#34D399;\">fewer than {pct}%</strong> "
            "of the world's trading nations."
        ),
        "data_scope": "Data Scope",
        "dataset_label": "Dataset",
        "field_source": "Source",
        "field_period": "Period",
        "field_countries": "Countries",
        "field_flows": "Flows",
        "field_granularity": "Granularity",
        "field_pipeline": "Pipeline",
        "val_source": "UN Comtrade — annual trade flows",
        "val_period": "2015–2024 (10 years)",
        "val_countries": "{n_total} (aggregate) / {n_bilateral} (bilateral 2024)",
        "val_flows": "Exports (X), Imports (M), Total Trade",
        "val_granularity": "Annual · aggregate (all products)",
        "val_pipeline": "Comtrade API → ETL Bronze/Silver/Gold",
        "key_limitation_label": "Key limitation",
        "key_limitation_body": (
            "Bilateral data for <strong style=\"color:#EEF4FB;\">2024 covers 133 countries</strong> "
            "due to UN Comtrade reporting lags. "
            "Full-period bilateral analysis uses all available years per country (2015–2023/2024)."
        ),
        "gold_layer_tables": "Gold layer tables",
        "gold_trade_data_desc": "544K rows · 188 reporters",
        "gold_entities_desc": "312 entities (262 countries + 50 aggregates)",
        "analytical_framework": "Analytical Framework",
        "f_title": "F · Global Trade Structure",
        "f_item1_h": "① Global system",
        "f_item1_b": "Concentration · evolution · volatility",
        "f_item2_h": "② Country profile",
        "f_item2_b": "Export share · coverage · growth",
        "f_item3_h": "③ Scoring",
        "f_item4_h": "④ Quadrants",
        "g_title": "G · Bilateral Trade Analysis",
        "g_item1_h": "① Partner map",
        "g_item1_b": "Sankey · top partners · HHI",
        "g_item2_h": "② Dependency",
        "g_item2_b": "BDI · critical share · asymmetry",
        "g_item3_h": "③ Scoring",
        "h_title": "H · Combined Scoring",
        "h_formula_h": "Formula",
        "h_formula_b": "0.70 × F rank + 0.30 × G rank",
        "h_profiles_h": "4 profiles",
        "h_quadrants_h": "4 quadrants",
        "design_choices": "Design Choices",
        "finding_median": (
            "<strong>Median over mean</strong> for full-period aggregation — "
            "robust to the 2020 COVID outlier across all indicators"
        ),
        "finding_rank_norm": (
            "<strong>Rank normalization</strong> in H — ensures equal contribution "
            "from F and G despite their different score distributions"
        ),
        "finding_export_centric": (
            "<strong>Export-centric scoring</strong> in F — trade power is measured by "
            "competitive export capacity, not import volume"
        ),
        "finding_aggregate_granularity": (
            "<strong>Aggregate-level granularity</strong> — product disaggregation requires one API call "
            "per reporter × year combination (~{n} countries × 10 years per product code). "
            "Aggregate flows provide complete, consistent {n}-country coverage "
            "that product-level data cannot match at this scale"
        ),
        "score_reading_guide": "Score Reading Guide",
        "score_guide_not_found": "Score reference guide image not found",
        "score_guide_expected": "Expected: Analysis/Figures/score_reference_guide.png",
        "continue_to": "Continue to →",
        "nav_global_dynamics": "Global Trade Dynamics",
        "nav_country_profile": "Country Trade Profile",
        "nav_cross_country": "Cross-Country Comparison",
        "nav_bilateral_risk": "Bilateral Risk",
        "nav_typology_scoring": "Typology & Scoring",
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 1 sur 6",
        "title": "Aperçu & méthodologie",
        "caption_header": "{n} pays · 10 ans · 3 dimensions analytiques · UN Comtrade · 2015–2024",
        "kpi_countries": "Pays",
        "kpi_years": "Années",
        "kpi_dimensions": "Dimensions",
        "kpi_gold_rows": "Lignes Gold",
        "executive_summary": "Synthèse exécutive",
        "label_global_trade": "Commerce mondial",
        "label_covid_impact": "Impact de la COVID",
        "label_trade_power": "Puissance commerciale",
        "label_bilateral_risk": "Risque bilatéral",
        "label_combined_score": "Score combiné",
        "finding_global_trade": "Les 10 premiers pays contrôlent **{min}–{max}% des exportations mondiales** — une concentration élevée et structurellement stable sur la décennie",
        "finding_covid_impact": "2020 n'a pas rebattu les architectures bilatérales — les corrélations de rang des positions des pays sont **restées au-dessus de 0,995** sur toute la période 2015–2024, avec le changement structurel maximal en **{year}**",
        "finding_trade_power": "**{n} pays** affichent un profil Global Leader dans la structure commerciale — menés par la Chine, les États-Unis, l'Allemagne et le Japon",
        "finding_bilateral_risk": "**{pct}% des pays** ({n}) sont en situation de **Critical Dependency** bilatérale — des économies majoritairement petites et dépendantes des matières premières",
        "finding_combined_score": "**{n} Global Champions** combinent une puissance commerciale de premier plan et une forte résilience bilatérale",
        "main_conclusion_label": "Conclusion principale",
        "main_conclusion_body": (
            "La puissance commerciale et la résilience commerciale sont complémentaires mais indépendantes. "
            "Les pays les plus stratégiquement précieux sont ceux qui combinent les deux — "
            "un profil rare détenu par <strong style=\"color:#34D399;\">moins de {pct}%</strong> "
            "des nations commerçantes dans le monde."
        ),
        "data_scope": "Périmètre des données",
        "dataset_label": "Jeu de données",
        "field_source": "Source",
        "field_period": "Période",
        "field_countries": "Pays",
        "field_flows": "Flux",
        "field_granularity": "Granularité",
        "field_pipeline": "Pipeline",
        "val_source": "UN Comtrade — flux commerciaux annuels",
        "val_period": "2015–2024 (10 ans)",
        "val_countries": "{n_total} (agrégé) / {n_bilateral} (bilatéral 2024)",
        "val_flows": "Exportations (X), Importations (M), Commerce total",
        "val_granularity": "Annuel · agrégé (tous produits)",
        "val_pipeline": "API Comtrade → ETL Bronze/Silver/Gold",
        "key_limitation_label": "Limite principale",
        "key_limitation_body": (
            "Les données bilatérales pour <strong style=\"color:#EEF4FB;\">2024 couvrent 133 pays</strong> "
            "en raison des délais de déclaration d'UN Comtrade. "
            "L'analyse bilatérale sur la période complète utilise toutes les années disponibles par pays (2015–2023/2024)."
        ),
        "gold_layer_tables": "Tables de la couche Gold",
        "gold_trade_data_desc": "544K lignes · 188 déclarants",
        "gold_entities_desc": "312 entités (262 pays + 50 agrégats)",
        "analytical_framework": "Cadre analytique",
        "f_title": "F · Structure du commerce mondial",
        "f_item1_h": "① Système mondial",
        "f_item1_b": "Concentration · évolution · volatilité",
        "f_item2_h": "② Profil pays",
        "f_item2_b": "Part des exportations · couverture · croissance",
        "f_item3_h": "③ Scoring",
        "f_item4_h": "④ Quadrants",
        "g_title": "G · Analyse bilatérale du commerce",
        "g_item1_h": "① Carte des partenaires",
        "g_item1_b": "Sankey · principaux partenaires · HHI",
        "g_item2_h": "② Dépendance",
        "g_item2_b": "BDI · part critique · asymétrie",
        "g_item3_h": "③ Scoring",
        "h_title": "H · Scoring combiné",
        "h_formula_h": "Formule",
        "h_formula_b": "0,70 × rang F + 0,30 × rang G",
        "h_profiles_h": "4 profils",
        "h_quadrants_h": "4 quadrants",
        "design_choices": "Choix de conception",
        "finding_median": (
            "<strong>Médiane plutôt que moyenne</strong> pour l'agrégation sur la période complète — "
            "robuste à la valeur aberrante du COVID de 2020 sur l'ensemble des indicateurs"
        ),
        "finding_rank_norm": (
            "<strong>Normalisation par rang</strong> dans H — garantit une contribution égale "
            "de F et G malgré leurs distributions de score différentes"
        ),
        "finding_export_centric": (
            "<strong>Scoring centré sur les exportations</strong> dans F — la puissance commerciale est mesurée "
            "par la capacité exportatrice compétitive, et non le volume des importations"
        ),
        "finding_aggregate_granularity": (
            "<strong>Granularité au niveau agrégé</strong> — la désagrégation par produit nécessite un appel API "
            "par combinaison déclarant × année (~{n} pays × 10 ans par code produit). "
            "Les flux agrégés offrent une couverture complète et cohérente de {n} pays "
            "qu'une donnée au niveau produit ne peut égaler à cette échelle"
        ),
        "score_reading_guide": "Guide de lecture des scores",
        "score_guide_not_found": "Image du guide de référence des scores introuvable",
        "score_guide_expected": "Attendu : Analysis/Figures/score_reference_guide.png",
        "continue_to": "Continuer vers →",
        "nav_global_dynamics": "Dynamique commerciale mondiale",
        "nav_country_profile": "Profil commercial par pays",
        "nav_cross_country": "Comparaison entre pays",
        "nav_bilateral_risk": "Risque bilatéral",
        "nav_typology_scoring": "Typologie & scoring",
    },
}

TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"

@st.cache_data
def load_overview_stats():
    df_c  = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_fp = pd.read_csv(TRADE_FILES / "df_full_periode.csv")
    df_st = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_cn = pd.read_csv(TRADE_FILES / "df_concentration.csv")
    df_fin         = pd.read_csv(TRADE_FILES / "df_final.csv")
    n_total        = len(df_c)
    n_champion     = int((df_c["combined_profile"] == "Global Champion").sum())
    pct_champion   = n_champion / n_total * 100
    n_critical     = int((df_fp["profile_full_periode"] == "Critical Dependency").sum())
    pct_critical   = n_critical / n_total * 100
    n_gl_leader    = int((df_st["meta_profile"] == "Global Leader").sum())
    top10_min      = df_cn["top10_share"].min()
    top10_max      = df_cn["top10_share"].max()
    min_corr_year  = int(df_cn.dropna(subset=["spearman_correlation"])
                           .loc[df_cn["spearman_correlation"].idxmin(), "year"])
    n_bilateral_2024 = int(df_fin[df_fin["year"] == 2024]["country_name"].nunique())
    return dict(
        n_total=n_total, n_champion=n_champion, pct_champion=pct_champion,
        n_critical=n_critical, pct_critical=pct_critical,
        n_gl_leader=n_gl_leader,
        top10_min=top10_min, top10_max=top10_max,
        min_corr_year=min_corr_year,
        n_bilateral_2024=n_bilateral_2024,
    )

_s = load_overview_stats()


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption_header").format(n=_s['n_total']))
thin_divider()


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

cols = st.columns(4)
kpis = [
    (t(STRINGS, "kpi_countries"),  str(_s["n_total"]), "teal"),
    (t(STRINGS, "kpi_years"),      "10",               "blue"),
    (t(STRINGS, "kpi_dimensions"), "3",                "amber"),
    (t(STRINGS, "kpi_gold_rows"),  "544K",             ""),
]
for col, (label, value, color) in zip(cols, kpis):
    col.markdown(kpi(label, value, color), unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# B. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "executive_summary"))

# Key findings — native Streamlit layout
results = [
    (t(STRINGS, "label_global_trade"),
     t(STRINGS, "finding_global_trade").format(min=f"{_s['top10_min']:.0f}", max=f"{_s['top10_max']:.0f}")),
    (t(STRINGS, "label_covid_impact"),
     t(STRINGS, "finding_covid_impact").format(year=_s['min_corr_year'])),
    (t(STRINGS, "label_trade_power"),
     t(STRINGS, "finding_trade_power").format(n=_s['n_gl_leader'])),
    (t(STRINGS, "label_bilateral_risk"),
     t(STRINGS, "finding_bilateral_risk").format(pct=f"{_s['pct_critical']:.0f}", n=_s['n_critical'])),
    (t(STRINGS, "label_combined_score"),
     t(STRINGS, "finding_combined_score").format(n=_s['n_champion'])),
]

for label, text in results:
    col1, col2 = st.columns([1, 3])
    col1.markdown(f"<div style='font-size:13px;font-weight:600;color:#FCD34D;padding:.4rem 0;'>{label}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='font-size:14px;color:#C4D5E8;padding:.4rem 0;'>{text}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #304870;margin:.1rem 0;'>", unsafe_allow_html=True)

# Main conclusion
st.markdown(f"""
<div style="background:#1A2C48;border:1px solid #304870;
            border-left:4px solid #10B981;border-radius:0 12px 12px 0;
            padding:1rem 1.25rem;margin-bottom:.5rem;">
  <div style="font-size:13px;font-weight:600;color:#10B981;
              text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;">
    {t(STRINGS, "main_conclusion_label")}
  </div>
  <div style="font-size:15px;color:#C4D5E8;line-height:1.8;font-style:italic;">
    {t(STRINGS, "main_conclusion_body").format(pct=f"{_s['pct_champion']:.0f}")}
  </div>
</div>
""", unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# C. DATA & METHODOLOGY
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "data_scope"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">{t(STRINGS, "dataset_label")}</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:2;">
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_source")}</span>
             {t(STRINGS, "val_source")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_period")}</span>
             {t(STRINGS, "val_period")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_countries")}</span>
             {t(STRINGS, "val_countries").format(n_total=_s['n_total'], n_bilateral=_s['n_bilateral_2024'])}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_flows")}</span>
             {t(STRINGS, "val_flows")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_granularity")}</span>
             {t(STRINGS, "val_granularity")}</div>
        <div><span style="color:#FCD34D;font-weight:600;min-width:120px;display:inline-block;">{t(STRINGS, "field_pipeline")}</span>
             {t(STRINGS, "val_pipeline")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem 1.1rem;height:100%;">
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.8rem;">{t(STRINGS, "key_limitation_label")}</div>
      <div style="font-size:14px;color:#A8C0D8;line-height:1.8;margin-bottom:.8rem;">
        {t(STRINGS, "key_limitation_body")}
      </div>
      <div style="font-size:12px;font-weight:600;color:#6890B0;text-transform:uppercase;
                  letter-spacing:.07em;margin:.8rem 0 .5rem;">{t(STRINGS, "gold_layer_tables")}</div>
      <div style="font-size:13px;color:#A8C0D8;line-height:1.9;
                  font-family:'IBM Plex Mono',monospace;">
        <span style="color:#34D399;">trade_data</span>
        &nbsp;·&nbsp;{t(STRINGS, "gold_trade_data_desc")}<br>
        <span style="color:#34D399;">entities</span>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp;{t(STRINGS, "gold_entities_desc")}<br>
        <span style="color:#34D399;">trade_flow</span>
        &nbsp;&nbsp;&nbsp;·&nbsp;M / X / RM / RX
      </div>
    </div>
    """, unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# ANALYTICAL FRAMEWORK — 3 sections
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "analytical_framework"))

st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;
            gap:1rem;align-items:stretch;">

  <div style="background:#231800;border:1.5px solid #D97706;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#FCD34D;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      {t(STRINGS, "f_title")}</div>
    <div style="font-size:13px;color:#FBBf24;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">{t(STRINGS, "f_item1_h")}</span><br>
        <span style="opacity:.8;">{t(STRINGS, "f_item1_b")}</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">{t(STRINGS, "f_item2_h")}</span><br>
        <span style="opacity:.8;">{t(STRINGS, "f_item2_b")}</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#FCD34D;font-weight:600;">{t(STRINGS, "f_item3_h")}</span><br>
        <span style="opacity:.8;">Strong · Good · Moderate · Weak</span>
      </div>
      <div>
        <span style="color:#FCD34D;font-weight:600;">{t(STRINGS, "f_item4_h")}</span><br>
        <span style="opacity:.8;">Static · Dynamic · Strategic</span>
      </div>
    </div>
  </div>

  <div style="background:#0F2240;border:1.5px solid #3B82F6;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#93C5FD;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      {t(STRINGS, "g_title")}</div>
    <div style="font-size:13px;color:#BAD7F8;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#93C5FD;font-weight:600;">{t(STRINGS, "g_item1_h")}</span><br>
        <span style="opacity:.8;">{t(STRINGS, "g_item1_b")}</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#93C5FD;font-weight:600;">{t(STRINGS, "g_item2_h")}</span><br>
        <span style="opacity:.8;">{t(STRINGS, "g_item2_b")}</span>
      </div>
      <div>
        <span style="color:#93C5FD;font-weight:600;">{t(STRINGS, "g_item3_h")}</span><br>
        <span style="opacity:.8;">Robust · Stable but Concentrated<br>Vulnerable · Critical Dependency</span>
      </div>
    </div>
  </div>

  <div style="background:#082E18;border:1.5px solid #10B981;border-radius:14px;
              padding:1rem;display:flex;flex-direction:column;">
    <div style="font-size:12px;font-weight:700;color:#34D399;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:.85rem;">
      {t(STRINGS, "h_title")}</div>
    <div style="font-size:13px;color:#6EE7B7;line-height:1.9;opacity:.9;flex:1;">
      <div style="margin-bottom:.5rem;">
        <span style="color:#34D399;font-weight:600;">{t(STRINGS, "h_formula_h")}</span><br>
        <span style="opacity:.8;">{t(STRINGS, "h_formula_b")}</span>
      </div>
      <div style="margin-bottom:.5rem;">
        <span style="color:#34D399;font-weight:600;">{t(STRINGS, "h_profiles_h")}</span><br>
        <span style="opacity:.8;">Global Champion · Competitive<br>Vulnerable · Fragile</span>
      </div>
      <div>
        <span style="color:#34D399;font-weight:600;">{t(STRINGS, "h_quadrants_h")}</span><br>
        <span style="opacity:.8;">Strong & Resilient · Strong but Exposed<br>Diversified Minor · Weak & Vulnerable</span>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# DESIGN CHOICES
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "design_choices"))
finding(t(STRINGS, "finding_median"))
finding(t(STRINGS, "finding_rank_norm"))
finding(t(STRINGS, "finding_export_centric"))
finding(t(STRINGS, "finding_aggregate_granularity").format(n=_s['n_total']))

thin_divider()


# ════════════════════════════════════════════════════════════════
# SCORE READING GUIDE
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "score_reading_guide"))
for candidate in [
    SCORE_GUIDE,
    ROOT / "Analysis" / "Figures" / "score_reference_guide.png",
]:
    if candidate.exists():
        st.image(str(candidate), width='stretch')
        break
else:
    st.markdown(f"""
    <div style="background:#1F3050;border:1px solid #304870;border-radius:12px;
                padding:1rem;text-align:center;color:#6890B0;font-size:13px;">
      {t(STRINGS, "score_guide_not_found")}<br>
      <span style="font-size:11px;font-family:'IBM Plex Mono',monospace;">
        {t(STRINGS, "score_guide_expected")}
      </span>
    </div>
    """, unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# NAVIGATION HINT
# ════════════════════════════════════════════════════════════════

st.markdown(
    f"<div style='font-size:13px;color:#6890B0;margin-bottom:.4rem;'>{t(STRINGS, 'continue_to')}</div>",
    unsafe_allow_html=True,
)
_nav_cols = st.columns(5)
_nav_pages = [
    ("pages/_04_Global_Trade_Dynamic.py",     t(STRINGS, "nav_global_dynamics")),
    ("pages/_05_Country_Trade_Profile.py",     t(STRINGS, "nav_country_profile")),
    ("pages/_06_Cross_Country_Comparison.py",  t(STRINGS, "nav_cross_country")),
    ("pages/_07_Bilateral_Risck_Analysis.py",  t(STRINGS, "nav_bilateral_risk")),
    ("pages/_08_Typology_Scoring.py",          t(STRINGS, "nav_typology_scoring")),
]
for _col, (_path, _label) in zip(_nav_cols, _nav_pages):
    _col.page_link(_path, label=_label)