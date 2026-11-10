"""
Indicators Analysis — I2 · By Indicator Deep Dive
Covers: E + I — Global vision + growth dynamics
Interactive: selectbox indicator → all charts regenerate
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
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding
from utils.i18n import t

inject_styles()

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════
# STRINGS — i18n
# ════════════════════════════════════════════════════════════════

STRINGS = {
    "en": {
        "eyebrow": "Indicators Analysis · Page 2 of 4",
        "title": "By Indicator — Deep Dive",
        "subtitle": "E + I — Global trend · distribution · stability · concentration · country rankings · world perspective",
        "select_indicator": "Select indicator",
        "kpi_countries": "Countries",
        "kpi_median_cagr": "Median CAGR",
        "kpi_median_volat": "Median Volat.",
        "kpi_avg_growth_score": "Avg Growth Score",
        "tab_trend": "📈 Trend & Distribution",
        "tab_stability": "📊 Stability & Concentration",
        "tab_rankings": "🔍 Country Rankings",
        "tab_world": "🌍 World Perspective",
        "sec_global_avg_trend": "E.1 · Global Average Trend",
        "metric_label": "Metric",
        "break_settings_expander": "⚙ Structural Break Settings",
        "break_threshold_slider": "Break threshold — |avg YoY| > threshold",
        "break_caption": "A structural break is identified when the **cross-country average** YoY growth rate exceeds this threshold in absolute value.",
        "label_global_mean": "Global Mean",
        "label_global_median": "Global Median",
        "trace_structural_break": "Structural Break",
        "annot_avg_annual_growth": "Avg annual growth : {v}",
        "annot_period_growth": "Period growth ({y0}–{y1}) : {v}",
        "trace_yoy_pct": "YoY Growth %",
        "thr_pos": "+{v}% threshold",
        "thr_neg": "−{v}% threshold",
        "title_trend_chart": "{ind} — Cross-Country Average",
        "sec_cross_country_dist": "E.2 · Cross-Country Distribution",
        "year_label": "Year",
        "trace_distribution": "Distribution",
        "annot_median": "Median: {v}",
        "annot_mean": "Mean: {v}",
        "title_dist_chart": "Distribution across countries — {yr}",
        "count_label": "Count",
        "sec_dist_evolution": "E.2 · Distribution Evolution — Box Plot",
        "clip_checkbox": "Clip extreme outliers from view (IQR-based)",
        "title_box_evolution": "Distribution Evolution — {ind}",
        "sec_data_coverage": "D.4 · Data Coverage",
        "metric_avg_coverage": "Avg Coverage",
        "metric_min_coverage": "Min Coverage ({yr})",
        "metric_total_countries": "Total Countries",
        "trace_coverage_pct": "Coverage %",
        "annot_70_threshold": "70% threshold",
        "title_data_coverage": "Data Coverage — {ind}",
        "yaxis_pct_countries_data": "% Countries with Data",
        "sec_stability_cv": "E.3 · Stability — Coefficient of Variation",
        "trace_cv_pct": "CV (%)",
        "trace_rolling_cv": "Rolling CV",
        "title_cv_chart": "Coefficient of Variation (%) — Cross-Country",
        "yaxis_cv_pct": "CV (%)",
        "info_no_cv": "No CV data for this indicator.",
        "sec_concentration_gini": "E.4 · Concentration — Gini Coefficient",
        "trace_gini": "Gini",
        "annot_high": "High (0.7)",
        "annot_moderate": "Moderate (0.5)",
        "annot_low": "Low (0.3)",
        "title_gini_chart": "Gini Coefficient — Cross-Country Concentration",
        "yaxis_gini": "Gini",
        "info_no_gini": "No Gini data for this indicator.",
        "level_low": "Low",
        "level_moderate": "Moderate",
        "level_high": "High",
        "level_very_high": "Very High",
        "kpi_cv_year": "CV ({yr})",
        "kpi_dispersion": "Dispersion",
        "kpi_gini_year": "Gini ({yr})",
        "kpi_concentration": "Concentration",
        "expander_methodology": "📖 How to read these charts — Methodology & Interpretation",
        "methodology_md": """
**Coefficient of Variation (CV) — cross-country dispersion**

The CV measures how spread out country values are relative to the global mean.
A high CV means some countries are very far from the average — the indicator is **divergent**.

| CV range | Reading |
|----------|---------|
| < 20 % | Low dispersion — countries converge around the mean |
| 20 – 50 % | Moderate dispersion — noticeable between-country differences |
| 50 – 100 % | High dispersion — strong inequalities across countries |
| > 100 % | Very high dispersion — a few outliers dominate the distribution |

A **rising CV** over time signals increasing divergence (countries are pulling apart).
A **falling CV** signals convergence (gap is closing).

---

**Gini Coefficient — cross-country concentration**

The Gini measures how unequally the indicator is distributed across countries (0 = perfect equality, 1 = maximum concentration).

| Gini range | Reading |
|-----------|---------|
| 0.0 – 0.3 | Low concentration — relatively equal distribution |
| 0.3 – 0.5 | Moderate concentration — some countries dominate |
| 0.5 – 0.7 | High concentration — a minority of countries hold most of the value |
| 0.7 – 1.0 | Very high concentration — extreme inequality across countries |

A **Gini above 0.6** typically signals that a handful of large economies (USA, China, EU) dominate the global total.

---

**Index-100 Evolution (2015 = 100)**

Each country's value is rebased to 100 in 2015, allowing **relative growth comparison** regardless of absolute scale.

- A country **above 100** has grown more than its 2015 level.
- A country **below 100** has contracted relative to 2015.
- **Parallel lines** = similar growth trajectories (structural alignment).
- **Diverging lines** = countries following different economic paths.
- **A line crossing 100** from below = the country recovered to its 2015 level.
""",
        "sec_index100": "E.3 · Index-100 Evolution — Top Countries",
        "highlight_countries": "Highlight countries",
        "annot_base2015": "Base 2015 = 100",
        "title_index100": "Index-100 Evolution (2015 = 100)",
        "yaxis_index": "Index",
        "sec_top_worst": "E.6 · Top & Worst Countries",
        "topbottom_n": "Top / Bottom N",
        "filter_countries": "Filter to specific countries (optional — leave empty to rank all countries)",
        "selected_countries": "Selected countries",
        "selected_countries_desc": "Selected countries (desc)",
        "top_n": "Top {n}",
        "bottom_n": "Bottom {n}",
        "heatmap_top_years": "{m} Heatmap — Top Countries × Years",
        "heatmap_top_n_countries": "{m} Heatmap — Top {n} Countries ({ind})",
        "expander_heatmap_howto": "📖 How to read this heatmap",
        "heatmap_md_template": (
            "**What is shown?** Each cell = **{m}** for one country (row) in one year (column).\n"
            "Countries are ordered by their most recent {m} (top = highest).\n\n"
            "**Color scale: {cs}**\n"
            "{color_note}\n\n"
            "**How to spot patterns**\n"
            "- **Full green row** → country consistently strong across all years.\n"
            "- **Full red row** → country structurally weak or declining.\n"
            "- **Vertical green column** → all top countries did well that year (global boom).\n"
            "- **Vertical red column** → systemic shock year (crisis, pandemic, etc.).\n"
            "- **Green-to-red gradient** along a row → a country that peaked and then declined.\n"
            "- **Blank / grey cell** → data not available for that year.\n\n"
            "{extra_note}"
        ),
        "color_note_zscore_yoy": "- 🟢 Green = positive / above average · 🔴 Red = negative / below average · 🟡 Yellow = neutral (near 0)",
        "color_note_other": "- 🟢 Dark green = highest values · ⬜ Light = lowest values",
        "extra_note_zscore": "**Z-score reminder** — a Z-score measures how many standard deviations a country is above (+) or below (−) the cross-country mean. |Z| > 2 signals an outlier.",
        "extra_note_yoy": "**YoY Growth %** — positive = growth vs prior year · negative = contraction.",
        "world_no_aggregate_msg": "⚠️ &nbsp;No <strong>World</strong> aggregate available for <em>{ind}</em> — F.1, F.2 (trend) and F.3 are hidden.",
        "world_no_aggregate_note": "The heatmap below covers all indicators available for World.",
        "sec_world_summary": "F.1 · World — Summary Statistics",
        "kpi_avg_annual_growth": "Avg Annual Growth",
        "kpi_cagr": "CAGR",
        "kpi_volatility": "Volatility (σ)",
        "kpi_pct_positive_years": "% Positive Years",
        "kpi_max_yoy": "Max YoY",
        "kpi_min_yoy": "Min YoY",
        "sec_world_trend": "F.2 · World — Value Trend & YoY Growth",
        "trace_world": "World",
        "annot_world_growth": "Avg growth : {avg}<br>CAGR : {cagr}",
        "title_world_value_trend": "{ind} — World Value Trend",
        "trace_world_yoy": "World YoY",
        "annot_mean_eq": "Mean = {v}",
        "title_world_yoy_growth": "{ind} — World YoY Growth",
        "sec_world_yoy_heatmap": "F.2 · YoY Growth Heatmap — World · All Indicators",
        "title_world_yoy_heatmap": "YoY Growth Heatmap — World | All Indicators",
        "colorbar_yoy_pct": "YoY %",
        "caption_extreme_values": "\\* {n} values outside ±50% range — clipped in color scale, actual value shown.",
        "sec_world_vs_countries": "F.3 · World vs Countries — Coherence & Growth Alignment",
        "caption_world_vs_countries": (
            "Compare the World Bank aggregate to country data. "
            "Use **Sum** for absolute/flow indicators (total GDP, exports…). "
            "Use **Mean** for per-capita, rates, or indices."
        ),
        "compare_world_to": "Compare World to",
        "sum_of_n_countries": "Sum of {n} countries",
        "mean_of_n_countries": "Mean of {n} countries",
        "info_no_overlap": "No overlapping data between World aggregate and country data for this indicator.",
        "gap_note_weighting": "<br><i style='font-size:10px'>WB: population-weighted<br>Mean: unweighted (each country = 1)</i>",
        "trace_world_wb": "World (WB)",
        "annot_avg_gap": "Avg gap = {pct}%{note}",
        "title_value_world_vs": "Value — World vs {ref}",
        "trace_divergence": "Divergence (World − Mean)",
        "trace_mean_country_yoy": "Mean country YoY",
        "title_growth_world_vs_mean": "Growth — World vs Mean of Countries",
        "yaxis_yoy_growth_pct": "YoY Growth (%)",
        "ratio_gap_note": (
            "💡 <strong>Expected gap for ratio / percentage indicators.</strong> "
            "The World Bank World value is computed as a global ratio "
            "(world total numerator ÷ world total denominator), "
            "not as the sum of individual country ratios. "
            "A gap around 100% in Sum mode is therefore methodologically consistent "
            "and does not signal a data issue."
        ),
    },
    "fr": {
        "eyebrow": "Analyse des indicateurs · Page 2 sur 4",
        "title": "Par indicateur — Analyse approfondie",
        "subtitle": "E + I — Tendance mondiale · distribution · stabilité · concentration · classements des pays · perspective mondiale",
        "select_indicator": "Sélectionner un indicateur",
        "kpi_countries": "Pays",
        "kpi_median_cagr": "CAGR médian",
        "kpi_median_volat": "Volat. médiane",
        "kpi_avg_growth_score": "Score de croissance moyen",
        "tab_trend": "📈 Tendance & distribution",
        "tab_stability": "📊 Stabilité & concentration",
        "tab_rankings": "🔍 Classements des pays",
        "tab_world": "🌍 Perspective mondiale",
        "sec_global_avg_trend": "E.1 · Tendance moyenne mondiale",
        "metric_label": "Indicateur",
        "break_settings_expander": "⚙ Paramètres de rupture structurelle",
        "break_threshold_slider": "Seuil de rupture — |variation annuelle moy.| > seuil",
        "break_caption": "Une rupture structurelle est identifiée lorsque le taux de croissance annuel **moyen entre pays** dépasse ce seuil en valeur absolue.",
        "label_global_mean": "Moyenne mondiale",
        "label_global_median": "Médiane mondiale",
        "trace_structural_break": "Rupture structurelle",
        "annot_avg_annual_growth": "Croissance annuelle moy. : {v}",
        "annot_period_growth": "Croissance sur la période ({y0}–{y1}) : {v}",
        "trace_yoy_pct": "Variation annuelle (%)",
        "thr_pos": "Seuil +{v}%",
        "thr_neg": "Seuil −{v}%",
        "title_trend_chart": "{ind} — Moyenne entre pays",
        "sec_cross_country_dist": "E.2 · Distribution entre pays",
        "year_label": "Année",
        "trace_distribution": "Distribution",
        "annot_median": "Médiane : {v}",
        "annot_mean": "Moyenne : {v}",
        "title_dist_chart": "Distribution entre pays — {yr}",
        "count_label": "Nombre",
        "sec_dist_evolution": "E.2 · Évolution de la distribution — Boîtes à moustaches",
        "clip_checkbox": "Rogner les valeurs extrêmes de la vue (méthode IQR)",
        "title_box_evolution": "Évolution de la distribution — {ind}",
        "sec_data_coverage": "D.4 · Couverture des données",
        "metric_avg_coverage": "Couverture moy.",
        "metric_min_coverage": "Couverture min. ({yr})",
        "metric_total_countries": "Total pays",
        "trace_coverage_pct": "Couverture (%)",
        "annot_70_threshold": "Seuil de 70 %",
        "title_data_coverage": "Couverture des données — {ind}",
        "yaxis_pct_countries_data": "% de pays avec données",
        "sec_stability_cv": "E.3 · Stabilité — Coefficient de variation",
        "trace_cv_pct": "CV (%)",
        "trace_rolling_cv": "CV glissant",
        "title_cv_chart": "Coefficient de variation (%) — Entre pays",
        "yaxis_cv_pct": "CV (%)",
        "info_no_cv": "Aucune donnée de CV pour cet indicateur.",
        "sec_concentration_gini": "E.4 · Concentration — Coefficient de Gini",
        "trace_gini": "Gini",
        "annot_high": "Élevé (0,7)",
        "annot_moderate": "Modéré (0,5)",
        "annot_low": "Faible (0,3)",
        "title_gini_chart": "Coefficient de Gini — Concentration entre pays",
        "yaxis_gini": "Gini",
        "info_no_gini": "Aucune donnée de Gini pour cet indicateur.",
        "level_low": "Faible",
        "level_moderate": "Modéré",
        "level_high": "Élevé",
        "level_very_high": "Très élevé",
        "kpi_cv_year": "CV ({yr})",
        "kpi_dispersion": "Dispersion",
        "kpi_gini_year": "Gini ({yr})",
        "kpi_concentration": "Concentration",
        "expander_methodology": "📖 Comment lire ces graphiques — Méthodologie & interprétation",
        "methodology_md": """
**Coefficient de variation (CV) — dispersion entre pays**

Le CV mesure à quel point les valeurs des pays sont dispersées par rapport à la moyenne mondiale.
Un CV élevé signifie que certains pays sont très loin de la moyenne — l'indicateur est **divergent**.

| Plage de CV | Lecture |
|----------|---------|
| < 20 % | Faible dispersion — les pays convergent autour de la moyenne |
| 20 – 50 % | Dispersion modérée — différences notables entre pays |
| 50 – 100 % | Forte dispersion — fortes inégalités entre pays |
| > 100 % | Dispersion très forte — quelques valeurs extrêmes dominent la distribution |

Un **CV en hausse** dans le temps signale une divergence croissante (les pays s'éloignent les uns des autres).
Un **CV en baisse** signale une convergence (l'écart se referme).

---

**Coefficient de Gini — concentration entre pays**

Le Gini mesure le degré d'inégalité de répartition de l'indicateur entre pays (0 = égalité parfaite, 1 = concentration maximale).

| Plage de Gini | Lecture |
|-----------|---------|
| 0,0 – 0,3 | Faible concentration — répartition relativement égale |
| 0,3 – 0,5 | Concentration modérée — certains pays dominent |
| 0,5 – 0,7 | Forte concentration — une minorité de pays détient la majeure partie de la valeur |
| 0,7 – 1,0 | Concentration très forte — inégalité extrême entre pays |

Un **Gini supérieur à 0,6** signale généralement qu'une poignée de grandes économies (États-Unis, Chine, UE) dominent le total mondial.

---

**Évolution base 100 (2015 = 100)**

La valeur de chaque pays est rebasée à 100 en 2015, permettant une **comparaison de croissance relative** indépendamment de l'échelle absolue.

- Un pays **au-dessus de 100** a progressé par rapport à son niveau de 2015.
- Un pays **en dessous de 100** s'est contracté par rapport à 2015.
- **Lignes parallèles** = trajectoires de croissance similaires (alignement structurel).
- **Lignes divergentes** = pays suivant des trajectoires économiques différentes.
- **Une ligne franchissant 100** par le bas = le pays a retrouvé son niveau de 2015.
""",
        "sec_index100": "E.3 · Évolution base 100 — Principaux pays",
        "highlight_countries": "Pays à mettre en évidence",
        "annot_base2015": "Base 2015 = 100",
        "title_index100": "Évolution base 100 (2015 = 100)",
        "yaxis_index": "Indice",
        "sec_top_worst": "E.6 · Meilleurs & pires pays",
        "topbottom_n": "Top / Bottom N",
        "filter_countries": "Filtrer sur des pays spécifiques (optionnel — laisser vide pour classer tous les pays)",
        "selected_countries": "Pays sélectionnés",
        "selected_countries_desc": "Pays sélectionnés (desc.)",
        "top_n": "Top {n}",
        "bottom_n": "Bottom {n}",
        "heatmap_top_years": "Heatmap {m} — Meilleurs pays × années",
        "heatmap_top_n_countries": "Heatmap {m} — Top {n} pays ({ind})",
        "expander_heatmap_howto": "📖 Comment lire cette heatmap",
        "heatmap_md_template": (
            "**Que montre ce graphique ?** Chaque cellule = **{m}** pour un pays (ligne) sur une année (colonne).\n"
            "Les pays sont classés selon leur {m} le plus récent (le haut = le plus élevé).\n\n"
            "**Échelle de couleur : {cs}**\n"
            "{color_note}\n\n"
            "**Comment repérer les tendances**\n"
            "- **Ligne entièrement verte** → pays constamment performant sur toutes les années.\n"
            "- **Ligne entièrement rouge** → pays structurellement faible ou en déclin.\n"
            "- **Colonne verte verticale** → tous les meilleurs pays ont bien performé cette année (essor mondial).\n"
            "- **Colonne rouge verticale** → année de choc systémique (crise, pandémie, etc.).\n"
            "- **Dégradé du vert vers le rouge** sur une ligne → un pays qui a culminé puis décliné.\n"
            "- **Cellule vide / grise** → données non disponibles pour cette année.\n\n"
            "{extra_note}"
        ),
        "color_note_zscore_yoy": "- 🟢 Vert = positif / au-dessus de la moyenne · 🔴 Rouge = négatif / en dessous de la moyenne · 🟡 Jaune = neutre (proche de 0)",
        "color_note_other": "- 🟢 Vert foncé = valeurs les plus élevées · ⬜ Clair = valeurs les plus faibles",
        "extra_note_zscore": "**Rappel sur le Z-score** — un Z-score mesure le nombre d'écarts-types par lequel un pays se situe au-dessus (+) ou en dessous (−) de la moyenne entre pays. |Z| > 2 signale une valeur aberrante.",
        "extra_note_yoy": "**Variation annuelle (%)** — positif = croissance par rapport à l'année précédente · négatif = contraction.",
        "world_no_aggregate_msg": "⚠️ &nbsp;Aucun agrégat <strong>World</strong> disponible pour <em>{ind}</em> — F.1, F.2 (tendance) et F.3 sont masquées.",
        "world_no_aggregate_note": "Le heatmap ci-dessous couvre tous les indicateurs disponibles pour World.",
        "sec_world_summary": "F.1 · World — Statistiques synthétiques",
        "kpi_avg_annual_growth": "Croissance annuelle moy.",
        "kpi_cagr": "CAGR",
        "kpi_volatility": "Volatilité (σ)",
        "kpi_pct_positive_years": "% d'années positives",
        "kpi_max_yoy": "Variation annuelle max.",
        "kpi_min_yoy": "Variation annuelle min.",
        "sec_world_trend": "F.2 · World — Tendance de valeur & croissance annuelle",
        "trace_world": "World",
        "annot_world_growth": "Croissance moy. : {avg}<br>CAGR : {cagr}",
        "title_world_value_trend": "{ind} — Tendance de valeur World",
        "trace_world_yoy": "Variation annuelle World",
        "annot_mean_eq": "Moyenne = {v}",
        "title_world_yoy_growth": "{ind} — Croissance annuelle World",
        "sec_world_yoy_heatmap": "F.2 · Heatmap de croissance annuelle — World · Tous indicateurs",
        "title_world_yoy_heatmap": "Heatmap de croissance annuelle — World | Tous indicateurs",
        "colorbar_yoy_pct": "Var. an. %",
        "caption_extreme_values": "\\* {n} valeurs hors de la plage ±50 % — tronquées dans l'échelle de couleur, valeur réelle affichée.",
        "sec_world_vs_countries": "F.3 · World vs pays — Cohérence & alignement de croissance",
        "caption_world_vs_countries": (
            "Comparer l'agrégat de la Banque mondiale aux données pays. "
            "Utiliser **Somme** pour les indicateurs absolus/de flux (PIB total, exportations…). "
            "Utiliser **Moyenne** pour les indicateurs par habitant, les taux ou les indices."
        ),
        "compare_world_to": "Comparer World à",
        "sum_of_n_countries": "Somme de {n} pays",
        "mean_of_n_countries": "Moyenne de {n} pays",
        "info_no_overlap": "Aucune donnée commune entre l'agrégat World et les données pays pour cet indicateur.",
        "gap_note_weighting": "<br><i style='font-size:10px'>WB : pondéré par population<br>Moyenne : non pondérée (chaque pays = 1)</i>",
        "trace_world_wb": "World (WB)",
        "annot_avg_gap": "Écart moy. = {pct}%{note}",
        "title_value_world_vs": "Valeur — World vs {ref}",
        "trace_divergence": "Divergence (World − Moyenne)",
        "trace_mean_country_yoy": "Variation annuelle moy. pays",
        "title_growth_world_vs_mean": "Croissance — World vs moyenne des pays",
        "yaxis_yoy_growth_pct": "Croissance annuelle (%)",
        "ratio_gap_note": (
            "💡 <strong>Écart attendu pour les indicateurs de ratio / pourcentage.</strong> "
            "La valeur World de la Banque mondiale est calculée comme un ratio mondial "
            "(numérateur total mondial ÷ dénominateur total mondial), "
            "et non comme la somme des ratios individuels par pays. "
            "Un écart proche de 100 % en mode Somme est donc méthodologiquement cohérent "
            "et ne signale pas un problème de données."
        ),
    },
}

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_ind     = pd.read_csv(IND_FILES / "df_indicators.csv")
    df_cv      = pd.read_csv(IND_FILES / "df_cv.csv")
    df_gini    = pd.read_csv(IND_FILES / "df_gini.csv")
    df_growth  = pd.read_csv(IND_FILES / "growth_df.csv")
    df_gdp_cl  = pd.read_csv(IND_FILES / "GDP_per_capita_USD_cluster_df.csv")
    # Countries only — also drop rows where country_name is NaN (unresolved join artifacts)
    df_entity = df_ind[df_ind["entity_type"] == "ENTITY"].copy()
    df_ind = df_ind[(df_ind["entity_type"] == "COUNTRY") &
                    df_ind["country_name"].notna()].copy()
    return df_ind, df_cv, df_gini, df_growth, df_gdp_cl, df_entity

df_ind, df_cv, df_gini, df_growth, df_gdp_cl, df_entity = load_data()

INDICATORS = sorted(df_ind["indicator_name"].dropna().unique().tolist())
YEARS      = sorted(df_ind["year"].unique())
YEAR_MAX   = int(YEARS[-1])

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
PLOTLY_LAYOUT_NM = {k: v for k, v in PLOTLY_LAYOUT.items()
                    if k not in ("margin","yaxis","xaxis","legend",
                                 "paper_bgcolor","plot_bgcolor")}


def _fmt_val(v, is_pct: bool = False) -> str:
    """Format a numeric value with K/M/B/T suffix for readability."""
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if is_pct:
        return f"{v:+.2f}%"
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1e12:
        return f"{v / 1e12:.2f} T"
    if av >= 1e9:
        return f"{v / 1e9:.2f} B"
    if av >= 1e6:
        return f"{v / 1e6:.2f} M"
    if av >= 1e3:
        return f"{v / 1e3:.1f} K"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"

CLUSTER_COLORS = {
    0: "#10B981", 1: "#60A5FA", 2: "#F59E0B",
    3: "#EF4444", 4: "#A78BFA", 5: "#F472B6",
}

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "subtitle"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# INDICATOR SELECTOR
# ════════════════════════════════════════════════════════════════

col_sel, col_info = st.columns([2, 2])
with col_sel:
    default_idx = next((i for i, n in enumerate(INDICATORS)
                        if "GDP per capita" in n and "current" in n), 0)
    indicator = st.selectbox(t(STRINGS, "select_indicator"), INDICATORS,
                             index=default_idx, key="i2_indicator")

# Filter data for indicator
df_i   = df_ind[df_ind["indicator_name"] == indicator].copy()
df_c_i = df_cv[df_cv["indicator_name"] == indicator].sort_values("year")
df_g_i = df_gini[df_gini["indicator_name"] == indicator].sort_values("year")
df_gr_i = df_growth[df_growth["indicator_name"] == indicator].copy()

# Unit
unit = df_i["indicator_unit"].iloc[0] if not df_i.empty else ""

with col_info:
    if not df_gr_i.empty:
        median_cagr  = df_gr_i["cagr"].median()
        median_vol   = df_gr_i["volatility"].median()
        n_countries  = df_i["country_name"].nunique()
        mean_gs      = df_gr_i["growth_score"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi(t(STRINGS, "kpi_countries"),    str(n_countries),         "teal"),  unsafe_allow_html=True)
        k2.markdown(kpi(t(STRINGS, "kpi_median_cagr"),  f"{median_cagr*100:.1f}%","amber"), unsafe_allow_html=True)
        k3.markdown(kpi(t(STRINGS, "kpi_median_volat"), f"{median_vol:.2f}",      "blue"),  unsafe_allow_html=True)
        k4.markdown(kpi(t(STRINGS, "kpi_avg_growth_score"), f"{mean_gs:.3f}",      ""),      unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    t(STRINGS, "tab_trend"),
    t(STRINGS, "tab_stability"),
    t(STRINGS, "tab_rankings"),
    t(STRINGS, "tab_world"),
])


# ── TAB 1 : TREND & DISTRIBUTION ─────────────────────────────
# ── TAB 1 : TREND & DISTRIBUTION ─────────────────────────────
with tab1:
    section_label(t(STRINGS, "sec_global_avg_trend"))

    view_metric = st.radio(
        t(STRINGS, "metric_label"),
        ["Mean value", "Median value", "YoY Growth %"],
        horizontal=True, key="trend_metric",
    )

    df_trend = (df_i.groupby("year")
                .agg(mean_val=("value","mean"),
                     median_val=("value","median"),
                     mean_yoy=("yoy_growth","mean"))
                .reset_index())

    # Structural break threshold (user-configurable)
    with st.expander(t(STRINGS, "break_settings_expander"), expanded=False):
        _BREAK_THR = st.slider(
            t(STRINGS, "break_threshold_slider"),
            min_value=0.03, max_value=0.20, value=0.07, step=0.01,
            format="%.2f", key="break_thr",
        )
        st.caption(t(STRINGS, "break_caption"))
    _break_years  = df_trend[df_trend["mean_yoy"].abs() > _BREAK_THR]["year"].tolist()
    _avg_growth   = df_trend["mean_yoy"].mean()

    fig_trend = go.Figure()

    if view_metric in ("Mean value", "Median value"):
        _y_col      = "mean_val"   if view_metric == "Mean value" else "median_val"
        _line_color = "#10B981"    if view_metric == "Mean value" else "#60A5FA"
        _fill_color = ("rgba(16,185,129,0.07)" if view_metric == "Mean value"
                       else "rgba(96,165,250,0.07)")
        _label      = t(STRINGS, "label_global_mean") if view_metric == "Mean value" else t(STRINGS, "label_global_median")
        fig_trend.add_trace(go.Scatter(
            x=df_trend["year"], y=df_trend[_y_col],
            mode="lines+markers", name=_label,
            line=dict(color=_line_color, width=2.5),
            marker=dict(size=7, color=_line_color),
            fill="tozeroy", fillcolor=_fill_color,
        ))
        # Red circle markers ON break data points + YoY% label above each
        _df_brk = df_trend[df_trend["year"].isin(_break_years)]
        if not _df_brk.empty:
            _brk_texts = [f"{v*100:+.1f}%" for v in _df_brk["mean_yoy"]]
            fig_trend.add_trace(go.Scatter(
                x=_df_brk["year"], y=_df_brk[_y_col],
                mode="markers+text", name=t(STRINGS, "trace_structural_break"),
                marker=dict(color="#EF4444", size=13, symbol="circle",
                            line=dict(color="#EF4444", width=2)),
                text=_brk_texts, textposition="top center",
                textfont=dict(size=10, color="#EF4444"),
            ))
        # Avg annual growth — bottom-right corner annotation
        fig_trend.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text=t(STRINGS, "annot_avg_annual_growth").format(v=f"{_avg_growth*100:+.2f}%"),
            showarrow=False, align="right",
            font=dict(size=11, color="#F59E0B"),
            bgcolor="#1F3050", bordercolor="#304870",
            borderwidth=1, borderpad=4,
        )
        # Period growth (E.5) — top-left corner annotation
        _first = df_trend[_y_col].iloc[0]
        _last  = df_trend[_y_col].iloc[-1]
        if _first != 0 and not np.isnan(_first):
            _period_gr = (_last - _first) / abs(_first) * 100
            _yr_first  = int(df_trend["year"].iloc[0])
            _yr_last   = int(df_trend["year"].iloc[-1])
            fig_trend.add_annotation(
                x=0.02, y=0.95, xref="paper", yref="paper",
                text=t(STRINGS, "annot_period_growth").format(
                    y0=_yr_first, y1=_yr_last, v=f"{_period_gr:+.1f}%"),
                showarrow=False, align="left",
                font=dict(size=11, color="#A78BFA"),
                bgcolor="#1F3050", bordercolor="#304870",
                borderwidth=1, borderpad=4,
            )
    else:
        # YoY Growth %: break years in amber, others green/red
        _bar_colors = []
        _bar_widths = []
        for _yr, _v in zip(df_trend["year"], df_trend["mean_yoy"]):
            if _yr in _break_years:
                _bar_colors.append("#F59E0B")
                _bar_widths.append(2)
            else:
                _bar_colors.append("#10B981" if _v >= 0 else "#EF4444")
                _bar_widths.append(0)
        fig_trend.add_trace(go.Bar(
            x=df_trend["year"], y=df_trend["mean_yoy"] * 100,
            name=t(STRINGS, "trace_yoy_pct"),
            marker=dict(color=_bar_colors,
                        line=dict(color="#F59E0B", width=_bar_widths)),
            opacity=0.85,
        ))
        fig_trend.add_hline(y=0, line_color="#EEF4FB",
                            line_width=1, line_dash="dash")
        fig_trend.add_hline(y=_BREAK_THR * 100, line_dash="dot",
                            line_color="#F59E0B", line_width=1,
                            annotation_text=t(STRINGS, "thr_pos").format(v=f"{_BREAK_THR*100:.0f}"),
                            annotation_font_color="#F59E0B",
                            annotation_position="top right")
        fig_trend.add_hline(y=-_BREAK_THR * 100, line_dash="dot",
                            line_color="#F59E0B", line_width=1,
                            annotation_text=t(STRINGS, "thr_neg").format(v=f"{_BREAK_THR*100:.0f}"),
                            annotation_font_color="#F59E0B",
                            annotation_position="bottom right")

    fig_trend.update_layout(**{**PLOTLY_LAYOUT,
        "title": t(STRINGS, "title_trend_chart").format(ind=indicator[:50]),
        "yaxis_title": unit if view_metric != "YoY Growth %" else "%",
    })
    st.plotly_chart(fig_trend, width='stretch')

    section_label(t(STRINGS, "sec_cross_country_dist"))

    year_dist = st.selectbox(
        t(STRINGS, "year_label"), YEARS[::-1], index=0, key="dist_year")

    df_dist = df_i[df_i["year"] == year_dist]["value"].dropna()

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_dist,
        nbinsx=30,
        marker_color="#60A5FA",
        opacity=0.75,
        name=t(STRINGS, "trace_distribution"),
    ))
    fig_dist.add_vline(
        x=df_dist.median(), line_dash="dash",
        line_color="#10B981",
        annotation_text=t(STRINGS, "annot_median").format(v=_fmt_val(df_dist.median())),
        annotation_font_color="#10B981",
        annotation_position="top left",
    )
    fig_dist.add_vline(
        x=df_dist.mean(), line_dash="dash",
        line_color="#F59E0B",
        annotation_text=t(STRINGS, "annot_mean").format(v=_fmt_val(df_dist.mean())),
        annotation_font_color="#F59E0B",
        annotation_position="bottom right",
    )
    # Anchor x-axis at actual data minimum to avoid phantom negative range
    _x_lo = df_dist.min()
    _x_hi = df_dist.max()
    _x_pad = (_x_hi - _x_lo) * 0.05
    fig_dist.update_layout(**{**PLOTLY_LAYOUT,
        "title": t(STRINGS, "title_dist_chart").format(yr=year_dist),
        "xaxis": dict(range=[_x_lo - _x_pad, _x_hi + _x_pad],
                      title=unit,
                      gridcolor="#304870", linecolor="#304870"),
        "yaxis_title": t(STRINGS, "count_label"),
    })
    st.plotly_chart(fig_dist, width='stretch')

    # Box plot evolution
    section_label(t(STRINGS, "sec_dist_evolution"))
    _clip_box = st.checkbox(t(STRINGS, "clip_checkbox"),
                            value=True, key="box_clip")
    df_box = df_i[["year","value"]].dropna()

    # IQR-based clipping: keeps boxes clearly visible while hiding extreme tails
    _q1  = df_box["value"].quantile(0.25)
    _q3  = df_box["value"].quantile(0.75)
    _iqr = _q3 - _q1
    _y_min = max(df_box["value"].min(), _q1 - 1.5 * _iqr)
    _y_max = _q3 + 3.0 * _iqr
    _y_range = [_y_min, _y_max] if _clip_box else None

    fig_box = go.Figure()
    for yr in YEARS:
        df_yr = df_box[df_box["year"] == yr]["value"]
        fig_box.add_trace(go.Box(
            y=df_yr,
            name=str(yr),
            line_color="#60A5FA",
            fillcolor="rgba(96,165,250,0.15)",
            boxmean=True,
            showlegend=False,
            boxpoints="outliers",
            marker=dict(color="#EF4444", size=4, opacity=0.5),
        ))
    _box_yaxis = dict(gridcolor="#304870", linecolor="#304870")
    if _y_range:
        _box_yaxis["range"] = _y_range
    _box_layout = {**PLOTLY_LAYOUT,
                   "title": t(STRINGS, "title_box_evolution").format(ind=indicator[:50]),
                   "xaxis_title": t(STRINGS, "year_label"),
                   "yaxis_title": unit,
                   "yaxis": _box_yaxis,
                   "height": 520}
    fig_box.update_layout(**_box_layout)
    st.plotly_chart(fig_box, width='stretch')

    section_label(t(STRINGS, "sec_data_coverage"))
    _n_total = df_ind["country_name"].nunique()
    _cov = (df_i.groupby("year")["value"]
            .apply(lambda x: x.notna().sum() / _n_total * 100)
            .reset_index(name="coverage_pct"))

    _avg_cov    = _cov["coverage_pct"].mean()
    _min_cov    = _cov["coverage_pct"].min()
    _min_cov_yr = int(_cov.loc[_cov["coverage_pct"].idxmin(), "year"])

    _dc1, _dc2, _dc3 = st.columns(3)
    _dc1.metric(t(STRINGS, "metric_avg_coverage"), f"{_avg_cov:.0f}%")
    _dc2.metric(t(STRINGS, "metric_min_coverage").format(yr=_min_cov_yr), f"{_min_cov:.0f}%")
    _dc3.metric(t(STRINGS, "metric_total_countries"), str(_n_total))

    _bar_clrs = ["#EF4444" if c < 70 else "#10B981"
                 for c in _cov["coverage_pct"]]
    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(
        x=_cov["year"], y=_cov["coverage_pct"],
        marker_color=_bar_clrs, name=t(STRINGS, "trace_coverage_pct"),
        text=[f"{v:.0f}%" for v in _cov["coverage_pct"]],
        textposition="outside",
    ))
    fig_cov.add_hline(y=70, line_dash="dash", line_color="#F59E0B",
                      line_width=1.5, annotation_text=t(STRINGS, "annot_70_threshold"),
                      annotation_font_color="#F59E0B",
                      annotation_position="bottom right")
    fig_cov.update_layout(**{**PLOTLY_LAYOUT,
        "title": t(STRINGS, "title_data_coverage").format(ind=indicator[:50]),
        "yaxis": dict(range=[0, 110], gridcolor="#304870", linecolor="#304870"),
        "xaxis_title": t(STRINGS, "year_label"),
        "yaxis_title": t(STRINGS, "yaxis_pct_countries_data"),
        "height": 300,
    })
    st.plotly_chart(fig_cov, width='stretch')



# ── TAB 2 : STABILITY & CONCENTRATION ────────────────────────
# ── TAB 2 : STABILITY & CONCENTRATION ────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        section_label(t(STRINGS, "sec_stability_cv"))
        if not df_c_i.empty:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=df_c_i["year"], y=df_c_i["cv"] * 100,
                mode="lines+markers", name=t(STRINGS, "trace_cv_pct"),
                line=dict(color="#F59E0B", width=2.5),
                marker=dict(size=7),
            ))
            if "rolling_cv" in df_c_i.columns:
                fig_cv.add_trace(go.Scatter(
                    x=df_c_i["year"], y=df_c_i["rolling_cv"] * 100,
                    mode="lines", name=t(STRINGS, "trace_rolling_cv"),
                    line=dict(color="#A78BFA", width=2, dash="dash"),
                ))
            fig_cv.update_layout(**PLOTLY_LAYOUT,
                title=t(STRINGS, "title_cv_chart"),
            )
            fig_cv.update_yaxes(title=t(STRINGS, "yaxis_cv_pct"), ticksuffix="%", tickformat=".1f")
            st.plotly_chart(fig_cv, width='stretch')
        else:
            st.info(t(STRINGS, "info_no_cv"))

    with col2:
        section_label(t(STRINGS, "sec_concentration_gini"))
        if not df_g_i.empty:
            fig_gini = go.Figure()
            fig_gini.add_trace(go.Scatter(
                x=df_g_i["year"], y=df_g_i["gini"],
                mode="lines+markers", name=t(STRINGS, "trace_gini"),
                line=dict(color="#EF4444", width=2.5),
                marker=dict(size=7),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.07)",
            ))
            fig_gini.add_hline(y=0.7, line_dash="dash",
                               line_color="#EF4444", line_width=1,
                               annotation_text=t(STRINGS, "annot_high"),
                               annotation_font_color="#EF4444",
                               annotation_position="top right")
            fig_gini.add_hline(y=0.5, line_dash="dash",
                               line_color="#F59E0B", line_width=1,
                               annotation_text=t(STRINGS, "annot_moderate"),
                               annotation_font_color="#F59E0B",
                               annotation_position="top right")
            fig_gini.add_hline(y=0.3, line_dash="dot",
                               line_color="#10B981", line_width=1,
                               annotation_text=t(STRINGS, "annot_low"),
                               annotation_font_color="#10B981",
                               annotation_position="top right")
            fig_gini.update_layout(**PLOTLY_LAYOUT,
                title=t(STRINGS, "title_gini_chart"),
                yaxis_title=t(STRINGS, "yaxis_gini"),
            )
            st.plotly_chart(fig_gini, width='stretch')
        else:
            st.info(t(STRINGS, "info_no_gini"))

    # ── Current-year summary (always visible, not inside expander) ─────────
    _tab2_kpis = st.columns(4)
    if not df_c_i.empty:
        _cv_last  = df_c_i.iloc[-1]["cv"] * 100          # stored as 0-1, display as %
        _cv_yr    = int(df_c_i.iloc[-1]["year"])
        _cv_trend = (df_c_i["cv"].iloc[-1] - df_c_i["cv"].iloc[0]) * 100 if len(df_c_i) > 1 else 0
        _cv_label = (t(STRINGS, "level_low") if _cv_last < 20 else t(STRINGS, "level_moderate") if _cv_last < 50
                     else t(STRINGS, "level_high") if _cv_last < 100 else t(STRINGS, "level_very_high"))
        _cv_dir   = "↗" if _cv_trend > 2 else ("↘" if _cv_trend < -2 else "→")
        _tab2_kpis[0].markdown(kpi(t(STRINGS, "kpi_cv_year").format(yr=_cv_yr), f"{_cv_last:.1f}%", "amber"),
                                unsafe_allow_html=True)
        _tab2_kpis[1].markdown(kpi(t(STRINGS, "kpi_dispersion"), f"{_cv_label} {_cv_dir}", ""),
                                unsafe_allow_html=True)
    if not df_g_i.empty:
        _gini_last  = df_g_i.iloc[-1]["gini"]
        _gini_yr    = int(df_g_i.iloc[-1]["year"])
        _gini_label = (t(STRINGS, "level_low") if _gini_last < 0.3 else t(STRINGS, "level_moderate") if _gini_last < 0.5
                       else t(STRINGS, "level_high") if _gini_last < 0.7 else t(STRINGS, "level_very_high"))
        _tab2_kpis[2].markdown(kpi(t(STRINGS, "kpi_gini_year").format(yr=_gini_yr), f"{_gini_last:.3f}", "blue"),
                                unsafe_allow_html=True)
        _tab2_kpis[3].markdown(kpi(t(STRINGS, "kpi_concentration"), _gini_label, ""),
                                unsafe_allow_html=True)
    thin_divider()

    with st.expander(t(STRINGS, "expander_methodology"),
                      expanded=False):
        st.markdown(t(STRINGS, "methodology_md"))


    # Index 100 comparison
    section_label(t(STRINGS, "sec_index100"))
    highlight_i = st.multiselect(
        t(STRINGS, "highlight_countries"),
        options=sorted(df_i["country_name"].dropna().unique().tolist()),
        default=[c for c in ["United States","France","Japan","Canada"]
                         if c in df_i["country_name"].values],
        max_selections=8,
        key="idx100_countries",
    )
    if highlight_i:
        df_idx = df_i[df_i["country_name"].isin(highlight_i)].copy()
        PALETTE = ["#F59E0B","#60A5FA","#10B981","#EF4444",
                   "#A78BFA","#F472B6","#34D399","#FB923C"]
        cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(highlight_i)}

        fig_idx = go.Figure()
        for c in highlight_i:
            df_cc = df_idx[df_idx["country_name"] == c].sort_values("year")
            fig_idx.add_trace(go.Scatter(
                x=df_cc["year"], y=df_cc["index_100"],
                mode="lines+markers", name=c,
                line=dict(color=cmap[c], width=2),
                marker=dict(size=6),
            ))
        fig_idx.add_hline(y=100, line_dash="dash",
                          line_color="#EEF4FB", line_width=1,
                          annotation_text=t(STRINGS, "annot_base2015"))
        fig_idx.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_index100"),
            yaxis_title=t(STRINGS, "yaxis_index"),
        )
        st.plotly_chart(fig_idx, width='stretch')





# ── TAB 3 : COUNTRY RANKINGS ─────────────────────────────────
# ── TAB 4 : COUNTRY RANKINGS ─────────────────────────────────
with tab3:
    section_label(t(STRINGS, "sec_top_worst"))

    _rk1, _rk2, _rk3 = st.columns([2, 1, 2])
    with _rk1:
        year_rank = st.selectbox(t(STRINGS, "year_label"), YEARS[::-1], index=0, key="rank_year")
    with _rk2:
        n_rank = st.selectbox(t(STRINGS, "topbottom_n"), [5, 10, 15, 20], index=1, key="rank_n")
    with _rk3:
        rank_metric = st.radio(
            t(STRINGS, "metric_label"), ["Value", "YoY Growth %", "Z-score"],
            horizontal=True, key="rank_metric",
        )

    _country_filter = st.multiselect(
        t(STRINGS, "filter_countries"),
        options=sorted(df_i["country_name"].dropna().unique().tolist()),
        default=[],
        key="rank_country_filter",
    )

    metric_col = {"Value": "value",
                  "YoY Growth %": "yoy_growth",
                  "Z-score": "z_score"}[rank_metric]

    df_rank_yr = df_i[df_i["year"] == year_rank].dropna(subset=[metric_col])
    if _country_filter:
        df_rank_yr = df_rank_yr[df_rank_yr["country_name"].isin(_country_filter)]

    col1, col2 = st.columns(2)

    with col1:
        df_top = (df_rank_yr.sort_values(metric_col) if _country_filter
                  else df_rank_yr.nlargest(n_rank, metric_col).sort_values(metric_col))
        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            x=df_top[metric_col] * (100 if rank_metric == "YoY Growth %" else 1),
            y=df_top["country_name"],
            orientation="h",
            marker_color="#10B981",
            text=(df_top[metric_col]
                  * (100 if rank_metric == "YoY Growth %" else 1)
                  ).apply(lambda v: f"{v:+.2f}%" if rank_metric == "YoY Growth %"
                                    else (f"{v:+.3f}" if rank_metric == "Z-score"
                                          else _fmt_val(v))),
            textposition="auto",
            opacity=0.85,
        ))
        _top_label = t(STRINGS, "selected_countries") if _country_filter else t(STRINGS, "top_n").format(n=n_rank)
        fig_top.update_layout(**PLOTLY_LAYOUT,
            title=f"{_top_label} — {rank_metric} ({year_rank})",
            xaxis_title=rank_metric,
        )
        st.plotly_chart(fig_top, width='stretch')

    with col2:
        df_bot = (df_rank_yr.sort_values(metric_col, ascending=False) if _country_filter
                  else df_rank_yr.nsmallest(n_rank, metric_col).sort_values(metric_col, ascending=False))
        fig_bot = go.Figure()
        fig_bot.add_trace(go.Bar(
            x=df_bot[metric_col] * (100 if rank_metric == "YoY Growth %" else 1),
            y=df_bot["country_name"],
            orientation="h",
            marker_color="#EF4444",
            text=(df_bot[metric_col]
                  * (100 if rank_metric == "YoY Growth %" else 1)
                  ).apply(lambda v: f"{v:+.2f}%" if rank_metric == "YoY Growth %"
                                    else (f"{v:+.3f}" if rank_metric == "Z-score"
                                          else _fmt_val(v))),
            textposition="auto",
            opacity=0.85,
        ))
        _bot_label = t(STRINGS, "selected_countries_desc") if _country_filter else t(STRINGS, "bottom_n").format(n=n_rank)
        fig_bot.update_layout(**PLOTLY_LAYOUT,
            title=f"{_bot_label} — {rank_metric} ({year_rank})",
            xaxis_title=rank_metric,
        )
        st.plotly_chart(fig_bot, width='stretch')

    # Heatmap — aligned with the selected metric (same country set as ranking)
    section_label(t(STRINGS, "heatmap_top_years").format(m=rank_metric))
    _HM_N = 15
    top_c = (df_rank_yr["country_name"].tolist() if _country_filter
             else (df_rank_yr.nlargest(_HM_N, metric_col)["country_name"].tolist()
                   if metric_col in df_rank_yr.columns else []))
    if top_c:
        _hm_scale = 100 if rank_metric == "YoY Growth %" else 1
        df_hm = df_i[df_i["country_name"].isin(top_c)].pivot_table(
            index="country_name", columns="year", values=metric_col
        ).reindex(top_c)

        _hm_vals = df_hm.values * _hm_scale
        _colorscale = "RdYlGn" if rank_metric in ("Z-score", "YoY Growth %") else "Greens"
        _zmid      = 0 if rank_metric in ("Z-score", "YoY Growth %") else None

        fig_hm = go.Figure(go.Heatmap(
            z=_hm_vals,
            x=[str(y) for y in df_hm.columns],
            y=df_hm.index.tolist(),
            colorscale=_colorscale,
            zmid=_zmid,
            text=np.round(_hm_vals, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            showscale=True,
            hoverongaps=False,
        ))
        fig_hm.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "heatmap_top_n_countries").format(m=rank_metric, n=_HM_N, ind=indicator[:40]),
            xaxis_title=t(STRINGS, "year_label"),
            height=max(350, len(top_c) * 35 + 80),
        )
        st.plotly_chart(fig_hm, width='stretch')

        with st.expander(t(STRINGS, "expander_heatmap_howto"), expanded=False):
            _cs_name = "RdYlGn" if rank_metric in ("Z-score", "YoY Growth %") else "Greens"
            _color_note = (t(STRINGS, "color_note_zscore_yoy")
                           if rank_metric in ("Z-score", "YoY Growth %")
                           else t(STRINGS, "color_note_other"))
            _extra_note = (t(STRINGS, "extra_note_zscore") if rank_metric == "Z-score"
                           else t(STRINGS, "extra_note_yoy") if rank_metric == "YoY Growth %"
                           else "")
            st.markdown(t(STRINGS, "heatmap_md_template").format(
                m=rank_metric, cs=_cs_name, color_note=_color_note, extra_note=_extra_note))


# ── TAB 4 : WORLD PERSPECTIVE ────────────────────────────────
# ── TAB 5 : WORLD PERSPECTIVE (F) ────────────────────────────
with tab4:
    df_w = (df_entity[(df_entity["country_name"] == "World") &
                      (df_entity["indicator_name"] == indicator)]
            .sort_values("year").copy())

    _w_has_data = not df_w.empty and df_w["value"].notna().any()

    if not _w_has_data:
        st.markdown(
            f"<div style='background:#2D1B00;border:1.5px solid #D97706;border-radius:10px;"
            f"padding:.8rem 1rem;color:#FCD34D;font-size:14px;margin:.5rem 0 1rem;'>"
            f"{t(STRINGS, 'world_no_aggregate_msg').format(ind=indicator)}<br>"
            f"<span style='font-size:12px;opacity:.8;'>{t(STRINGS, 'world_no_aggregate_note')}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        # ── F.1 · Summary KPIs ───────────────────────────────
        section_label(t(STRINGS, "sec_world_summary"))
        _w_yoy   = df_w["yoy_growth"].dropna()
        _w_vals  = df_w["value"].dropna()
        _w_avg   = _w_yoy.mean()
        _w_vol   = _w_yoy.std()
        _w_ppos  = (_w_yoy > 0).mean() * 100
        _w_n     = len(_w_vals)
        try:
            _w_ratio = _w_vals.iloc[-1] / _w_vals.iloc[0]
            _w_cagr  = (_w_ratio ** (1 / (_w_n - 1)) - 1
                        if _w_n >= 2 and _w_vals.iloc[0] != 0 and _w_ratio > 0
                        else float("nan"))
        except Exception:
            _w_cagr = float("nan")
        _w_max_p = _w_yoy.max()
        _w_max_n = _w_yoy.min()

        _wk1, _wk2, _wk3, _wk4, _wk5, _wk6 = st.columns(6)
        _wk1.markdown(kpi(t(STRINGS, "kpi_avg_annual_growth"), f"{_w_avg*100:+.2f}%", "teal"),  unsafe_allow_html=True)
        _wk2.markdown(kpi(t(STRINGS, "kpi_cagr"),              f"{_w_cagr*100:+.2f}%" if not np.isnan(_w_cagr) else "N/A", "amber"), unsafe_allow_html=True)
        _wk3.markdown(kpi(t(STRINGS, "kpi_volatility"),    f"{_w_vol:.3f}",                 "blue"),  unsafe_allow_html=True)
        _wk4.markdown(kpi(t(STRINGS, "kpi_pct_positive_years"),  f"{_w_ppos:.0f}%",               ""),      unsafe_allow_html=True)
        _wk5.markdown(kpi(t(STRINGS, "kpi_max_yoy"),           f"{_w_max_p*100:+.1f}%",         "teal"),  unsafe_allow_html=True)
        _wk6.markdown(kpi(t(STRINGS, "kpi_min_yoy"),           f"{_w_max_n*100:+.1f}%",         ""),      unsafe_allow_html=True)

        thin_divider()

        # ── F.2 · Value Trend + YoY Barchart ─────────────────
        section_label(t(STRINGS, "sec_world_trend"))
        _fc1, _fc2 = st.columns(2)

        _f_break_thr = st.session_state.get("break_thr", 0.07)
        _w_brk_yrs   = df_w[df_w["yoy_growth"].abs() > _f_break_thr]["year"].tolist()

        with _fc1:
            fig_fw = go.Figure()
            fig_fw.add_trace(go.Scatter(
                x=df_w["year"], y=df_w["value"],
                mode="lines+markers", name=t(STRINGS, "trace_world"),
                line=dict(color="#60A5FA", width=2.5),
                marker=dict(size=7, color="#60A5FA"),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
            ))
            _df_wbrk = df_w[df_w["year"].isin(_w_brk_yrs)]
            if not _df_wbrk.empty:
                fig_fw.add_trace(go.Scatter(
                    x=_df_wbrk["year"], y=_df_wbrk["value"],
                    mode="markers+text", name=t(STRINGS, "trace_structural_break"),
                    marker=dict(color="#EF4444", size=13, symbol="circle",
                                line=dict(color="#EF4444", width=2)),
                    text=[f"{v*100:+.1f}%" for v in _df_wbrk["yoy_growth"]],
                    textposition="top center",
                    textfont=dict(size=10, color="#EF4444"),
                ))
            _w_first_v = df_w["value"].dropna().iloc[0] if not df_w["value"].dropna().empty else float("nan")
            _w_last_v  = df_w["value"].dropna().iloc[-1] if not df_w["value"].dropna().empty else float("nan")
            _w_period  = ((_w_last_v - _w_first_v) / abs(_w_first_v) * 100
                          if _w_first_v and not np.isnan(_w_first_v) and _w_first_v != 0
                          else float("nan"))
            fig_fw.add_annotation(
                x=0.98, y=0.05, xref="paper", yref="paper",
                text=t(STRINGS, "annot_world_growth").format(
                    avg=f"{_w_avg*100:+.2f}%", cagr=f"{_w_cagr*100:+.2f}%"),
                showarrow=False, align="right",
                font=dict(size=11, color="#F59E0B"),
                bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=4,
            )
            if not np.isnan(_w_period):
                _yr0 = int(df_w["year"].iloc[0])
                _yrN = int(df_w["year"].iloc[-1])
                fig_fw.add_annotation(
                    x=0.02, y=0.95, xref="paper", yref="paper",
                    text=t(STRINGS, "annot_period_growth").format(
                        y0=_yr0, y1=_yrN, v=f"{_w_period:+.1f}%"),
                    showarrow=False, align="left",
                    font=dict(size=11, color="#A78BFA"),
                    bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=4,
                )
            fig_fw.update_layout(**{**PLOTLY_LAYOUT,
                "title": t(STRINGS, "title_world_value_trend").format(ind=indicator[:50]),
                "yaxis_title": unit,
            })
            st.plotly_chart(fig_fw, width='stretch')

        with _fc2:
            _yoy_vals  = df_w["yoy_growth"].dropna()
            _yoy_years = df_w[df_w["yoy_growth"].notna()]["year"]
            _bar_clrs  = []
            for _yr, _v in zip(_yoy_years, _yoy_vals):
                if _yr in _w_brk_yrs:
                    _bar_clrs.append("#F59E0B")
                else:
                    _bar_clrs.append("#10B981" if _v >= 0 else "#EF4444")
            fig_fyoy = go.Figure()
            fig_fyoy.add_trace(go.Bar(
                x=_yoy_years, y=_yoy_vals * 100,
                marker_color=_bar_clrs, name=t(STRINGS, "trace_world_yoy"), opacity=0.85,
                text=[f"{v*100:+.1f}%" for v in _yoy_vals],
                textposition="outside",
                textfont=dict(color="#EEF4FB", size=11),
            ))
            fig_fyoy.add_hline(y=_w_avg * 100, line_dash="dash",
                               line_color="#F59E0B", line_width=1.5)
            fig_fyoy.add_annotation(
                x=0.99, y=0.97, xref="paper", yref="paper",
                text=t(STRINGS, "annot_mean_eq").format(v=f"{_w_avg*100:+.2f}%"),
                showarrow=False, align="right",
                font=dict(size=11, color="#F59E0B"),
                bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=3,
                xanchor="right", yanchor="top",
            )
            fig_fyoy.add_hline(y=0, line_color="#EEF4FB", line_width=1, line_dash="dash")
            fig_fyoy.update_layout(**{**PLOTLY_LAYOUT,
                "title": t(STRINGS, "title_world_yoy_growth").format(ind=indicator[:50]),
                "yaxis_title": "%",
            })
            st.plotly_chart(fig_fyoy, width='stretch')

        thin_divider()

    # ── F.2 · YoY Heatmap — All Indicators × Years ───────
    section_label(t(STRINGS, "sec_world_yoy_heatmap"))
    _df_world_all = (df_entity[df_entity["country_name"] == "World"]
                     .dropna(subset=["yoy_growth"]))
    if not _df_world_all.empty:
        _hm_world = _df_world_all.pivot_table(
            index="indicator_name", columns="year", values="yoy_growth"
        ) * 100
        _ind_order = (_hm_world.mean(axis=1).sort_values(ascending=False).index.tolist())
        _hm_world  = _hm_world.reindex(_ind_order)

        _hw_vals = np.round(_hm_world.values, 1)
        _hw_text = np.where(
            np.abs(_hw_vals) > 50,
            np.vectorize(lambda v: f"{v:+.0f}%*")(np.nan_to_num(_hw_vals)),
            np.vectorize(lambda v: f"{v:+.1f}%" if not np.isnan(v) else "")(
                _hm_world.values),
        )

        fig_fhm = go.Figure(go.Heatmap(
            z=_hw_vals,
            x=[str(y) for y in _hm_world.columns],
            y=[n[:45] for n in _hm_world.index],
            colorscale="RdYlGn",
            zmid=0,
            zmin=-50, zmax=50,
            text=_hw_text,
            texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=True,
            hoverongaps=False,
            colorbar=dict(title=t(STRINGS, "colorbar_yoy_pct")),
        ))
        fig_fhm.update_layout(**{**PLOTLY_LAYOUT,
            "title": t(STRINGS, "title_world_yoy_heatmap"),
            "xaxis_title": t(STRINGS, "year_label"),
            "height": max(400, len(_ind_order) * 40 + 100),
            "margin": dict(l=260, r=40, t=50, b=40),
        })
        _n_extreme = int((np.abs(_hw_vals) > 50).sum())
        st.plotly_chart(fig_fhm, width='stretch')
        if _n_extreme > 0:
            st.caption(t(STRINGS, "caption_extreme_values").format(n=_n_extreme))

    thin_divider()


    if _w_has_data:
        # ── F.3 · World vs Countries ──────────────────────────
        section_label(t(STRINGS, "sec_world_vs_countries"))
        st.caption(t(STRINGS, "caption_world_vs_countries"))
        _f3_default = (1 if any(kw in indicator.lower() for kw in
                          ["per capita", "per person", "per worker",
                           "per head", " rate", "ratio", "index",
                           "share", "% of", "growth rate"])
                       else 0)
        _f3_mode = st.radio(
            t(STRINGS, "compare_world_to"),
            ["Sum of countries", "Mean of countries"],
            horizontal=True, key=f"f3_mode_{indicator}",
            index=_f3_default,
        )

        _n_c = df_i["country_name"].nunique()
        _mean_yoy_c = (df_i.groupby("year")["yoy_growth"].mean()
                       .reset_index().rename(columns={"yoy_growth": "mean_yoy_c"}))

        if _f3_mode == "Sum of countries":
            _ref_c = (df_i.groupby("year")["value"].sum()
                      .reset_index().rename(columns={"value": "ref_countries"}))
            _ref_label = t(STRINGS, "sum_of_n_countries").format(n=_n_c)
        else:
            _ref_c = (df_i.groupby("year")["value"].mean()
                      .reset_index().rename(columns={"value": "ref_countries"}))
            _ref_label = t(STRINGS, "mean_of_n_countries").format(n=_n_c)

        _merged_v = df_w[["year","value"]].merge(_ref_c, on="year", how="inner")

        if _merged_v.empty:
            st.info(t(STRINGS, "info_no_overlap"))
        else:
            _fc3, _fc4 = st.columns(2)

            with _fc3:
                _gap_pct = ((_merged_v["value"] - _merged_v["ref_countries"]) /
                             _merged_v["ref_countries"].abs()).mean() * 100
                _gap_note = (t(STRINGS, "gap_note_weighting")
                             if _f3_mode != "Sum of countries" and abs(_gap_pct) > 15 else "")
                fig_f3l = go.Figure()
                fig_f3l.add_trace(go.Scatter(
                    x=_merged_v["year"], y=_merged_v["value"],
                    mode="lines+markers", name=t(STRINGS, "trace_world_wb"),
                    line=dict(color="#60A5FA", width=2.5),
                    marker=dict(size=7),
                ))
                fig_f3l.add_trace(go.Scatter(
                    x=_merged_v["year"], y=_merged_v["ref_countries"],
                    mode="lines+markers", name=_ref_label,
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                    marker=dict(size=7),
                ))
                fig_f3l.add_annotation(
                    x=0.98, y=0.5, xref="paper", yref="paper",
                    text=t(STRINGS, "annot_avg_gap").format(pct=f"{_gap_pct:+.1f}", note=_gap_note),
                    showarrow=False, align="right",
                    font=dict(size=11, color="#EEF4FB"),
                    bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=4,
                    xanchor="right", yanchor="middle",
                )
                fig_f3l.update_layout(**{**PLOTLY_LAYOUT,
                    "title": t(STRINGS, "title_value_world_vs").format(ref=_ref_label),
                    "yaxis_title": unit,
                    "legend": dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
                })
                st.plotly_chart(fig_f3l, width='stretch')

            with _fc4:
                _merged_g = (df_w[["year","yoy_growth"]].dropna()
                             .merge(_mean_yoy_c, on="year", how="inner"))
                _div = (_merged_g["yoy_growth"] - _merged_g["mean_yoy_c"]) * 100

                fig_f3r = go.Figure()
                fig_f3r.add_trace(go.Bar(
                    x=_merged_g["year"], y=_div,
                    name=t(STRINGS, "trace_divergence"),
                    marker_color=["rgba(16,185,129,0.35)" if v >= 0 else "rgba(239,68,68,0.35)"
                                  for v in _div],
                    showlegend=True,
                ))
                fig_f3r.add_trace(go.Scatter(
                    x=_merged_g["year"], y=_merged_g["yoy_growth"] * 100,
                    mode="lines+markers", name=t(STRINGS, "trace_world_yoy"),
                    line=dict(color="#60A5FA", width=2.5),
                    marker=dict(size=7),
                ))
                fig_f3r.add_trace(go.Scatter(
                    x=_merged_g["year"], y=_merged_g["mean_yoy_c"] * 100,
                    mode="lines+markers", name=t(STRINGS, "trace_mean_country_yoy"),
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                    marker=dict(size=7),
                ))
                fig_f3r.add_hline(y=0, line_color="#EEF4FB", line_width=1, line_dash="dash")
                fig_f3r.update_layout(**{**PLOTLY_LAYOUT,
                    "title": t(STRINGS, "title_growth_world_vs_mean"),
                    "yaxis_title": t(STRINGS, "yaxis_yoy_growth_pct"),
                    "barmode": "overlay",
                    "legend": dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
                })
                st.plotly_chart(fig_f3r, width='stretch')

            # Note contextuelle : écart ~100% en mode Somme pour indicateurs ratio
            _is_ratio_ind = any(kw in indicator.lower() for kw in
                                ["% of", "ratio", "share", " rate", " index",
                                 "per capita", "per person", "per head"])
            if _f3_mode == "Sum of countries" and abs(_gap_pct) > 50 and _is_ratio_ind:
                st.markdown(
                    "<div style='background:#0F2240;border-left:3px solid #60A5FA;"
                    "border-radius:6px;padding:.6rem .9rem;margin-top:.5rem;"
                    "font-size:12px;color:#93C5FD;'>"
                    + t(STRINGS, "ratio_gap_note") +
                    "</div>",
                    unsafe_allow_html=True,
                )
