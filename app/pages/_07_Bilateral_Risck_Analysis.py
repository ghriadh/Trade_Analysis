"""
Trade Analysis — T5 · Bilateral Risk Analysis
Tab 1 — Sankey & Top Partners
Tab 2 — HHI Concentration (shocks, scatter, ranking)
Tab 3 — BDIS Analysis (evolution, heatmap)
Tab 4 — BDI / Dependency (cartography, heatmap partner×year)
Tab 5 — Critical Dependencies (heatmap, summary table)
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
from plotly.subplots import make_subplots
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, format_value
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 5 of 6",
        "title": "Bilateral Risk Analysis",
        "caption": "G — Sankey · HHI concentration · BDIS · BDI dependency · critical dependencies",
        "select_country": "Select country",
        "select_partner": "Select partner",
        "year_label": "Year",
        "kpi_hhi_median": "HHI (Median)",
        "kpi_bdis_median": "BDIS (Median)",
        "kpi_critical_share": "Critical Share",
        "kpi_n_critical": "N Critical",
        "kpi_score_full_period": "Score Full Period",
        "kpi_profile_full_period": "Profile Full Period",
        "tab1_label": "🌊 Sankey & Partners",
        "tab2_label": "📊 HHI Concentration",
        "tab3_label": "📈 BDIS Analysis",
        "tab4_label": "🔗 BDI / Dependency",
        "tab5_label": "🏅 Scoring & Profile",
        "no_bilateral_data": "No bilateral data for {country}.",
        "top_n_partners_label": "Top N partners",
        "show_label": "Show",
        "flow_label": "Flow",
        "focus_label": "Focus",
        "opt_both": "Both",
        "opt_imports_only": "Imports only",
        "opt_exports_only": "Exports only",
        "sankey_title": "{country} — Top {n} Trade Partners · {year}  (🔵 Imports  🟡 Exports)",
        "top_partners_bar_section": "Top Partners — Bar Chart",
        "opt_exports_x": "Exports (X)",
        "opt_imports_m": "Imports (M)",
        "bar_title": "{country} — Top {n} {flow} Partners · {year}",
        "axis_usd_billion": "USD Billion",
        "section_hhi_evolution": "G.5 · HHI Evolution",
        "expander_what_is_hhi": "📖 What is HHI?",
        "hhi_md": """
**Herfindahl-Hirschman Index (HHI)** measures trade partner concentration.

**Formula**: HHI = Σ (share_i)² for all partners i

**Interpretation**:
- HHI → **0**: trade diversified across many partners
- HHI → **1**: trade concentrated on a single partner

| Level | Range |
|---|---|
| Low | HHI < 0.08 |
| Moderate | 0.08 – 0.15 |
| High | 0.15 – 0.25 |
| Extreme | > 0.25 |
        """,
        "shock_threshold_label": "Break detection threshold (|ΔHHI| ≥)",
        "trace_import_hhi": "Import HHI",
        "trace_export_hhi": "Export HHI",
        "hline_extreme": "Extreme (0.25)",
        "hline_high": "High (0.15)",
        "hline_moderate": "Moderate (0.08)",
        "annot_level_trend": "Level: <b>{lv}</b>  ·  Trend: <b>{tr}</b>",
        "hhi_evolution_title": "{country} — HHI Evolution  (▲ concentration break · ▼ diversification break)",
        "yaxis_hhi": "HHI",
        "section_hhi_scatter": "G.5.5 · HHI Import vs Export — All Countries",
        "caption_hhi_scatter": "Default view zoomed to 95th percentile — use Autoscale to reveal outliers.",
        "highlight_countries_label": "Highlight countries",
        "fig_hhi_sc_title": "HHI Import vs Export — {year}",
        "axis_hhi_import": "HHI Import",
        "axis_hhi_export": "HHI Export",
        "trace_all_countries": "All countries",
        "trace_import_eq_export": "Import = Export",
        "section_hhi_ranking": "G.5 · HHI Ranking",
        "opt_x_export": "X (Export)",
        "opt_m_import": "M (Import)",
        "opt_most_concentrated": "Most concentrated",
        "opt_most_diversified": "Most diversified",
        "fig_rank_hhi_title": "HHI {flow_rank} — {rank_dir} — {year}",
        "expander_break_detection": "📖 Break detection",
        "break_detection_md": """
**Break detection** (▲▼): a year is flagged as a break when |ΔHHI| ≥ threshold.
- **ΔHHI ≥ +{thr}** ▲ (red) — concentration break (fewer, larger partners)
- **ΔHHI ≤ −{thr}** ▼ (green) — diversification break (more balanced partners)
- Current threshold: **{thr}** (adjustable via the slider above)
        """,
        "section_bdis_evolution": "G.7 · BDIS Evolution",
        "expander_what_is_bdis": "📖 What is BDIS?",
        "bdis_md": """
**BDIS — Bilateral Dependency Index Structural** measures the structural imbalance
between export and import trade structures across all partners.

**Formula**: BDIS = Σ |X_share_i − M_share_i| / 2  (over all partners i)

A high BDIS means export and import partner structures diverge significantly
— sign of structural asymmetry and potential vulnerability.

| Level | Range |
|---|---|
| Interdependence | BDIS < 0.20 |
| Moderate Dependance | 0.20 – 0.50 |
| High Dependance | 0.50 – 0.80 |
| Critical Dependance | > 0.80 |

**Trend** (derived from ΔBDIS): Increasing · Stable · Decreasing

**Break detection** (▲▼): a year is flagged when |ΔBDIS| ≥ threshold.
- **ΔBDIS ≥ +threshold** ▲ (red) — structural dependency worsening
- **ΔBDIS ≤ −threshold** ▼ (green) — structural dependency improving
        """,
        "bdis_threshold_label": "Break detection threshold (|ΔBDIS| ≥)",
        "hline_critical": "Critical (0.80)",
        "hline_high2": "High (0.50)",
        "hline_moderate2": "Moderate (0.20)",
        "bdis_evolution_title": "{country} — BDIS Evolution  (▲ dependency break · ▼ improvement break)",
        "yaxis_bdis_score": "BDIS Score",
        "section_bdis_heatmap": "BDIS Heatmap — Countries × Years",
        "countries_label": "Countries",
        "expander_bdi_definitions": "📖 BDI — Bilateral Dependency Index — Definitions",
        "bdi_def_md": """
**BDI (Bilateral Dependency Index)** measures the structural asymmetry of dependency between two trading partners.

**Formulas** :
- `dependency(A→B)` = (total bilateral trade A↔B from A's perspective) / (A's total trade) &nbsp;*(X + M)*
- `dependency_reverse(A→B)` = (total bilateral trade B↔A from B's perspective) / (B's total trade)
- `BDI(A,B) = |dependency(A→B) − dependency_reverse(A→B)|` → [0, 1]

BDI measures the **absolute gap** between the relative weights each country assigns to the bilateral relationship in its trade portfolio.
A high BDI means one country is structurally much more dependent on the other.

| Level | Threshold | Meaning |
|---|---|---|
| **Interdependence** | BDI < 0.20 | Low asymmetry — balanced relationship |
| **Moderate Dependance** | 0.20 ≤ BDI < 0.50 | Moderate asymmetry |
| **High Dependance** | 0.50 ≤ BDI < 0.80 | Strong structural asymmetry |
| **Critical Dependance** | BDI ≥ 0.80 | Critical dependency |
        """,
        "section_dependency_map": "G.6.3 · Dependency Map — Top Partners ({year})",
        "no_bdi_data": "No BDI data for {country} in {year}.",
        "hover_we_depend": "We depend on them:",
        "hover_bdi_level": "BDI Level:",
        "hover_bdi_norm": "BDI norm:",
        "hover_they_depend": "They depend on us:",
        "hover_relationship": "Relationship:",
        "fig_mirror_title": "{country} — Dependency Map — Top {n} Partners ({year})",
        "direction_line": "← Partner depends on {country} &nbsp;|&nbsp; {country} depends on partner →",
        "expander_how_to_read": "How to read this chart",
        "how_to_read_md": """
**Right bars** — share of **{country}'s** total trade with each partner.
Colored by BDI level: green = Interdependence · blue = Moderate · orange = High · red = Critical.

**Left bars** — share of the **partner's** total trade with {country}.
Always shown in semi-transparent blue.

**BDI (right column)** — `|dep − dep_reverse|`: absolute asymmetry between the two relative weights.
Close to 0 = balanced relationship · Close to 1 = strong structural dependency.

**Threshold lines**: 20% (orange) and 50% (red) on the right side mark increasing risk zones.
        """,
        "expander_previous_chart": "Previous chart G.6.3 — for comparison",
        "opt_most_dependent_on": "Most dependent on",
        "opt_most_depends_on_us": "Most depends on us",
        "xlabel_dependency": "Dependency (we depend on them)",
        "xlabel_reverse_dependency": "Reverse Dependency (they depend on us)",
        "fig_cmp_title": "{country} — {flow} — Top {n} Partners ({year})",
        "section_pair_evolution": "G.6.2 · Pair Evolution — Deep Dive",
        "no_data_pair": "No data for {country} ↔ {partner}.",
        "hline_moderate3": "Moderate (0.20)",
        "hline_high3": "High (0.50)",
        "fig_pair_title": "{country} ↔ {partner} — Bilateral Dependency Evolution",
        "yaxis_dependency_pct": "Dependency (%)",
        "annot_level_bdi_dominant": "Level: <b>{lv}</b>  ·  BDI: <b>{bdi}</b>  ·  Dominant: <b>{dom}</b>",
        "section_bdi_heatmap_partner": "BDI Heatmap — Partner × Year",
        "select_partners_label": "Select partners",
        "fig_bdi_hm_title": "{country} — BDI by Partner × Year",
        "section_dep_heatmap": "G.8 · Dependency Heatmap — Countries × Years",
        "fig_g8_title": "G.8 · Top Partner Dependency — Partner | Dep% | BDI | N Critical",
        "expander_how_to_read_heatmap": "📖 How to read this heatmap",
        "how_to_read_heatmap_md": """
**Color** — bilateral trade share (X+M) with the top partner (green = low, red = high)

**Each cell (3 lines):**
- Line 1 — top trade partner name
- Line 2 — `dep%` : bilateral trade share · `BDI` with that partner
- Line 3 — `crit` : number of critical partners (trade share ≥ 15%)
        """,
        "section_score_components": "Score Components — {country} ({year})",
        "card_hhi_avg": "HHI (avg)",
        "card_top_dependency": "Top Dependency",
        "card_critical_share": "Critical Share",
        "card_n_critical": "N Critical",
        "card_score_global": "Score Global",
        "pf_profile_logic": "Profile · Logic",
        "pf_profile_score": "Profile · Score",
        "badge_divergence": "⚠ Divergence",
        "expander_definitions_formulas": "📖 Definitions & Formulas",
        "kpi_definitions_md": """
<style>
.kpi-exp h4 { font-size:12px !important; font-weight:700; text-transform:uppercase;
              letter-spacing:.06em; color:#8BAAC8; margin:10px 0 3px; }
.kpi-exp p, .kpi-exp li, .kpi-exp td, .kpi-exp th { font-size:12px !important; }
.kpi-exp code { font-size:11px !important; }
.kpi-exp hr { margin:6px 0; }
</style>
<div class="kpi-exp">

#### Profile Logic *(per flow — Import & Export separately)*
Based on **HHI level + trend** and **Top Dependency level + trend** :

| HHI Level | Dep Top Level | HHI Trend | Dep Trend | → Profile |
|---|---|---|---|---|
| Extreme | Critical | — | — | **Critical Dependency** |
| High or Extreme | High or Critical | Increasing | Increasing | **Critical Dependency** |
| Extreme | — | — | — | **Vulnerable** |
| — | Critical | — | — | **Vulnerable** |
| High | High | — | — | **Vulnerable** |
| High | Moderate | Increasing | — | **Vulnerable** |
| Moderate | High | — | Increasing | **Vulnerable** |
| Moderate | Moderate | Increasing | Increasing | **Vulnerable** |
| Moderate or High | — | — | — | **Stable but Concentrated** |
| Low | Low | — | — | **Robust** |

#### Profile Logic Global *(Import + Export combined)*

| Import Profile | Export Profile | → Global Profile |
|---|---|---|
| Critical | Critical | **Critical Dependency** |
| Critical | any | **Vulnerable** |
| any | Critical | **Vulnerable** |
| Vulnerable | any | **Vulnerable** |
| Robust | Robust | **Robust** |
| — | — | **Stable but Concentrated** |

---

#### Score Global
`score = 0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`

*(HHI_norm = average of import & export HHI norms)*

#### Profile Score *(from score global)*

| Profile | Score |
|---|---|
| **Robust** | < 0.25 |
| **Stable but Concentrated** | 0.25 – 0.35 |
| **Vulnerable** | 0.35 – 0.45 |
| **Critical Dependency** | ≥ 0.45 |

---

#### Thresholds reference

| Indicator | Low | Moderate | High | Extreme/Critical |
|---|---|---|---|---|
| **HHI** | < 0.08 | 0.08 – 0.15 | 0.15 – 0.25 | ≥ 0.25 |
| **Top Dependency** | < 0.15 | 0.15 – 0.25 | 0.25 – 0.45 | ≥ 0.45 |
| **BDIS** | < 0.20 (Interdep.) | 0.20 – 0.50 | 0.50 – 0.80 | ≥ 0.80 |

**Trend** : Stable (|Δ| ≤ 0.025) · Increasing (Δ > 0.025) · Decreasing (Δ < −0.025)

**Critical Share** : share of total trade (X+M) with partners where bilateral dependency ≥ 15%

**N Critical** : number of partners with bilateral dependency ≥ 15%

</div>
        """,
        "section_risk_space": "G.9.1 · Annual Risk Space — Animated (Import × Export)",
        "countries_to_display_label": "Countries to display",
        "lbl_import_risk_score": "Import Risk Score",
        "lbl_export_risk_score": "Export Risk Score",
        "lbl_profile": "Profile",
        "lbl_total_trade_b": "Total Trade (B$)",
        "fig_anim_title": "Bilateral Risk Space — Annual Evolution",
        "axis_import_risk": "Import Risk Score (higher = riskier) →",
        "axis_export_risk": "Export Risk Score (higher = riskier) →",
        "caption_anim": "▲ = first year · ● = last year · dotted lines show country trajectory",
        "anim_frame_annot": "<b>{country} · {year}</b><br>Score: <b>{score}</b>",
        "section_profile_heatmap": "G.9.2 · Profile Heatmap — Country × Year",
        "fig_phm_title": "G.9.2 · Annual Bilateral Profile (logic) — Country × Year",
        "section_divergence": "G.9.3 · Logic vs Score Divergence — {country}",
        "trace_logic_profile": "Logic Profile",
        "trace_score_profile": "Score Profile",
        "hover_logic_prefix": "Logic",
        "hover_profile": "Profile:",
        "hover_convergent": "Convergent:",
        "fig_div_title": "{country} — Logic Profile vs Score Profile by Year",
        "divergence_summary": "Logic ≠ Score in <b>{n}</b> out of {total} years ({pct:.0f}%) — {status}",
        "status_frequent": "⚠ Frequent divergence",
        "status_convergent": "✓ Mostly convergent",
        "no_annual_data": "No annual data for {country}.",
        "section_drift_stability": "G.10 · Full Period — Drift & Stability",
        "pills_add_columns_label": "Add columns",
        "col_country": "Country",
        "col_profile_full_period": "Profile Full Period",
        "col_dominant_annual_score": "Dominant Annual (Score)",
        "col_dominant_annual_logic": "Dominant Annual (Logic)",
        "col_divergence_full_vs_annual": "Divergence Full vs Annual",
        "col_divergence_category": "Divergence Category",
        "col_convergence_score_logic": "Convergence Score-Logic",
        "col_profile_stability": "Profile Stability",
        "col_stability_score": "Stability Score",
        "col_profile_drift": "Profile Drift",
        "col_drift_magnitude": "Drift Magnitude",
        "col_hhi_median": "HHI (Median)",
        "col_top_partner_dependency": "Top Partner Dependency",
        "col_critical_partners_share": "Critical Partners Share",
        "col_n_critical_partners": "N Critical Partners",
        "col_bdis_median": "BDIS (Median)",
        "col_score_full_period": "Score Full Period",
        "opt_drift_stability_conv": "Drift, Stability & Conv.",
        "opt_indicators": "Indicators",
        "expander_column_definitions": "📖 Column definitions & formulas",
        "fp_definitions_md": """
<style>
.fp-exp h4 { font-size:12px !important; font-weight:700; text-transform:uppercase;
             letter-spacing:.06em; color:#8BAAC8; margin:10px 0 3px; }
.fp-exp p, .fp-exp li, .fp-exp td, .fp-exp th { font-size:12px !important; }
.fp-exp code { font-size:11px !important; }
.fp-exp hr { margin:6px 0; }
</style>
<div class="fp-exp">

#### Always visible

**Profile Full Period** — risk classification computed from full-period median indicator values.
Thresholds : < 0.25 Robust · 0.25–0.35 Stable but Concentrated · 0.35–0.45 Vulnerable · ≥ 0.45 Critical Dependency

**Score Full Period formula** :
`score = 0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`
*(all components are cross-country normalized full-period median values)*

**Dominant Annual (Score)** — the risk profile that appeared most frequently across years (score-based method)

**Dominant Annual (Logic)** — the risk profile that appeared most frequently across years (logic-based method)
If the two dominants differ → the two methods disagreed structurally over the period.

**Divergence Full vs Annual** — share of years where the annual risk profile differed from *Profile Full Period*.
High value = the country changed profile frequently; the full-period classification hides significant volatility.

**Profile Stability** — summary label : ≥ 80% Very Stable · ≥ 60% Stable · ≥ 40% Moderate · < 40% Unstable

**Profile Drift** — compares the profile of the last 3 years to the full-period profile : *Improving · Stable · Worsening*

---

#### Drift, Stability & Conv. (toggle)

**Divergence Category** — categorical reading of *Divergence Full vs Annual* :
Never (0%) · Rare (≤ 20%) · Frequent (≤ 50%) · Systematic (> 50%)

**Convergence Score-Logic** — share of years where the score method and the logic method gave the same profile.
High value = the two approaches are consistent; low value = the country is in methodological grey zone.

**Stability Score** — raw proportion behind *Profile Stability* (0 to 1)

**Drift Magnitude** — number of profile levels shifted between recent period and full period (0 = no shift, 3 = maximum)

---

#### Indicators (toggle) — all values are full-period medians

**HHI (Median)** — trade concentration index, average of import & export HHI

**Top Partner Dependency** — share of total trade held by the single most important partner

**Critical Partners Share** — share of total trade with partners where bilateral dependency ≥ 15%

**N Critical Partners** — number of partners with bilateral dependency ≥ 15%

**BDIS (Median)** — bilateral dependency imbalance score, averaged over all partners

**Score Full Period** — composite risk score computed from full-period median values (same formula as above)

</div>
        """,
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 5 sur 6",
        "title": "Analyse du risque bilatéral",
        "caption": "G — Sankey · Concentration HHI · BDIS · Dépendance BDI · dépendances critiques",
        "select_country": "Sélectionner un pays",
        "select_partner": "Sélectionner un partenaire",
        "year_label": "Année",
        "kpi_hhi_median": "HHI (médiane)",
        "kpi_bdis_median": "BDIS (médiane)",
        "kpi_critical_share": "Part critique",
        "kpi_n_critical": "N critique",
        "kpi_score_full_period": "Score période complète",
        "kpi_profile_full_period": "Profil période complète",
        "tab1_label": "🌊 Sankey & partenaires",
        "tab2_label": "📊 Concentration HHI",
        "tab3_label": "📈 Analyse BDIS",
        "tab4_label": "🔗 BDI / Dépendance",
        "tab5_label": "🏅 Scoring & profil",
        "no_bilateral_data": "Aucune donnée bilatérale pour {country}.",
        "top_n_partners_label": "Top N partenaires",
        "show_label": "Afficher",
        "flow_label": "Flux",
        "focus_label": "Focus",
        "opt_both": "Les deux",
        "opt_imports_only": "Importations uniquement",
        "opt_exports_only": "Exportations uniquement",
        "sankey_title": "{country} — Top {n} partenaires commerciaux · {year}  (🔵 Importations  🟡 Exportations)",
        "top_partners_bar_section": "Top partenaires — Graphique en barres",
        "opt_exports_x": "Exportations (X)",
        "opt_imports_m": "Importations (M)",
        "bar_title": "{country} — Top {n} partenaires {flow} · {year}",
        "axis_usd_billion": "Milliards USD",
        "section_hhi_evolution": "G.5 · Évolution du HHI",
        "expander_what_is_hhi": "📖 Qu'est-ce que le HHI ?",
        "hhi_md": """
L'**indice de Herfindahl-Hirschman (HHI)** mesure la concentration des partenaires commerciaux.

**Formule** : HHI = Σ (part_i)² pour tous les partenaires i

**Interprétation** :
- HHI → **0** : commerce diversifié sur de nombreux partenaires
- HHI → **1** : commerce concentré sur un seul partenaire

| Niveau | Plage |
|---|---|
| Low | HHI < 0.08 |
| Moderate | 0.08 – 0.15 |
| High | 0.15 – 0.25 |
| Extreme | > 0.25 |
        """,
        "shock_threshold_label": "Seuil de détection des ruptures (|ΔHHI| ≥)",
        "trace_import_hhi": "HHI Import",
        "trace_export_hhi": "HHI Export",
        "hline_extreme": "Extrême (0.25)",
        "hline_high": "Élevé (0.15)",
        "hline_moderate": "Modéré (0.08)",
        "annot_level_trend": "Niveau : <b>{lv}</b>  ·  Tendance : <b>{tr}</b>",
        "hhi_evolution_title": "{country} — Évolution du HHI  (▲ rupture de concentration · ▼ rupture de diversification)",
        "yaxis_hhi": "HHI",
        "section_hhi_scatter": "G.5.5 · HHI import vs export — Tous les pays",
        "caption_hhi_scatter": "Vue par défaut zoomée sur le 95e percentile — utilisez Autoscale pour révéler les valeurs extrêmes.",
        "highlight_countries_label": "Pays à mettre en évidence",
        "fig_hhi_sc_title": "HHI import vs export — {year}",
        "axis_hhi_import": "HHI import",
        "axis_hhi_export": "HHI export",
        "trace_all_countries": "Tous les pays",
        "trace_import_eq_export": "Importations = Exportations",
        "section_hhi_ranking": "G.5 · Classement HHI",
        "opt_x_export": "X (Exportation)",
        "opt_m_import": "M (Importation)",
        "opt_most_concentrated": "Plus concentré",
        "opt_most_diversified": "Plus diversifié",
        "fig_rank_hhi_title": "HHI {flow_rank} — {rank_dir} — {year}",
        "expander_break_detection": "📖 Détection des ruptures",
        "break_detection_md": """
**Détection des ruptures** (▲▼) : une année est signalée comme rupture lorsque |ΔHHI| ≥ seuil.
- **ΔHHI ≥ +{thr}** ▲ (rouge) — rupture de concentration (partenaires moins nombreux et plus importants)
- **ΔHHI ≤ −{thr}** ▼ (vert) — rupture de diversification (partenaires plus équilibrés)
- Seuil actuel : **{thr}** (ajustable via le curseur ci-dessus)
        """,
        "section_bdis_evolution": "G.7 · Évolution du BDIS",
        "expander_what_is_bdis": "📖 Qu'est-ce que le BDIS ?",
        "bdis_md": """
Le **BDIS — Bilateral Dependency Index Structural** mesure le déséquilibre structurel
entre les structures commerciales d'exportation et d'importation, sur l'ensemble des partenaires.

**Formule** : BDIS = Σ |part_X_i − part_M_i| / 2  (sur tous les partenaires i)

Un BDIS élevé signifie que les structures de partenaires à l'export et à l'import
divergent fortement — signe d'asymétrie structurelle et de vulnérabilité potentielle.

| Niveau | Plage |
|---|---|
| Interdependence | BDIS < 0.20 |
| Moderate Dependance | 0.20 – 0.50 |
| High Dependance | 0.50 – 0.80 |
| Critical Dependance | > 0.80 |

**Tendance** (dérivée de ΔBDIS) : Increasing · Stable · Decreasing

**Détection des ruptures** (▲▼) : une année est signalée lorsque |ΔBDIS| ≥ seuil.
- **ΔBDIS ≥ +seuil** ▲ (rouge) — dégradation de la dépendance structurelle
- **ΔBDIS ≤ −seuil** ▼ (vert) — amélioration de la dépendance structurelle
        """,
        "bdis_threshold_label": "Seuil de détection des ruptures (|ΔBDIS| ≥)",
        "hline_critical": "Critique (0.80)",
        "hline_high2": "Élevé (0.50)",
        "hline_moderate2": "Modéré (0.20)",
        "bdis_evolution_title": "{country} — Évolution du BDIS  (▲ rupture de dépendance · ▼ rupture d'amélioration)",
        "yaxis_bdis_score": "Score BDIS",
        "section_bdis_heatmap": "Carte de chaleur BDIS — Pays × années",
        "countries_label": "Pays",
        "expander_bdi_definitions": "📖 BDI — Bilateral Dependency Index — Définitions",
        "bdi_def_md": """
Le **BDI (Bilateral Dependency Index)** mesure l'asymétrie structurelle de dépendance entre deux partenaires commerciaux.

**Formules** :
- `dependency(A→B)` = (commerce bilatéral total A↔B du point de vue de A) / (commerce total de A) &nbsp;*(X + M)*
- `dependency_reverse(A→B)` = (commerce bilatéral total B↔A du point de vue de B) / (commerce total de B)
- `BDI(A,B) = |dependency(A→B) − dependency_reverse(A→B)|` → [0, 1]

Le BDI mesure l'**écart absolu** entre les poids relatifs que chaque pays accorde à la relation bilatérale dans son portefeuille commercial.
Un BDI élevé signifie qu'un pays est structurellement beaucoup plus dépendant de l'autre.

| Niveau | Seuil | Signification |
|---|---|---|
| **Interdependence** | BDI < 0.20 | Faible asymétrie — relation équilibrée |
| **Moderate Dependance** | 0.20 ≤ BDI < 0.50 | Asymétrie modérée |
| **High Dependance** | 0.50 ≤ BDI < 0.80 | Forte asymétrie structurelle |
| **Critical Dependance** | BDI ≥ 0.80 | Dépendance critique |
        """,
        "section_dependency_map": "G.6.3 · Carte de dépendance — Top partenaires ({year})",
        "no_bdi_data": "Aucune donnée BDI pour {country} en {year}.",
        "hover_we_depend": "Nous dépendons d'eux :",
        "hover_bdi_level": "Niveau BDI :",
        "hover_bdi_norm": "Norme BDI :",
        "hover_they_depend": "Ils dépendent de nous :",
        "hover_relationship": "Relation :",
        "fig_mirror_title": "{country} — Carte de dépendance — Top {n} partenaires ({year})",
        "direction_line": "← Le partenaire dépend de {country} &nbsp;|&nbsp; {country} dépend du partenaire →",
        "expander_how_to_read": "Comment lire ce graphique",
        "how_to_read_md": """
**Barres de droite** — part du commerce total de **{country}** avec chaque partenaire.
Colorée par niveau BDI : vert = Interdependence · bleu = Moderate · orange = High · rouge = Critical.

**Barres de gauche** — part du commerce total du **partenaire** avec {country}.
Toujours affichée en bleu semi-transparent.

**BDI (colonne de droite)** — `|dep − dep_reverse|` : asymétrie absolue entre les deux poids relatifs.
Proche de 0 = relation équilibrée · Proche de 1 = forte dépendance structurelle.

**Lignes de seuil** : 20% (orange) et 50% (rouge) sur le côté droit marquent des zones de risque croissant.
        """,
        "expander_previous_chart": "Graphique précédent G.6.3 — pour comparaison",
        "opt_most_dependent_on": "Le plus dépendant de",
        "opt_most_depends_on_us": "Dépend le plus de nous",
        "xlabel_dependency": "Dépendance (nous dépendons d'eux)",
        "xlabel_reverse_dependency": "Dépendance inverse (ils dépendent de nous)",
        "fig_cmp_title": "{country} — {flow} — Top {n} partenaires ({year})",
        "section_pair_evolution": "G.6.2 · Évolution de la paire — Analyse approfondie",
        "no_data_pair": "Aucune donnée pour {country} ↔ {partner}.",
        "hline_moderate3": "Modéré (0.20)",
        "hline_high3": "Élevé (0.50)",
        "fig_pair_title": "{country} ↔ {partner} — Évolution de la dépendance bilatérale",
        "yaxis_dependency_pct": "Dépendance (%)",
        "annot_level_bdi_dominant": "Niveau : <b>{lv}</b>  ·  BDI : <b>{bdi}</b>  ·  Dominant : <b>{dom}</b>",
        "section_bdi_heatmap_partner": "Carte de chaleur BDI — Partenaire × année",
        "select_partners_label": "Sélectionner des partenaires",
        "fig_bdi_hm_title": "{country} — BDI par partenaire × année",
        "section_dep_heatmap": "G.8 · Carte de chaleur de la dépendance — Pays × années",
        "fig_g8_title": "G.8 · Dépendance au partenaire principal — Partenaire | Dép% | BDI | N critique",
        "expander_how_to_read_heatmap": "📖 Comment lire cette carte de chaleur",
        "how_to_read_heatmap_md": """
**Couleur** — part du commerce bilatéral (X+M) avec le partenaire principal (vert = faible, rouge = élevé)

**Chaque cellule (3 lignes) :**
- Ligne 1 — nom du principal partenaire commercial
- Ligne 2 — `dep%` : part du commerce bilatéral · `BDI` avec ce partenaire
- Ligne 3 — `crit` : nombre de partenaires critiques (part commerciale ≥ 15%)
        """,
        "section_score_components": "Composantes du score — {country} ({year})",
        "card_hhi_avg": "HHI (moy.)",
        "card_top_dependency": "Dépendance principale",
        "card_critical_share": "Part critique",
        "card_n_critical": "N critique",
        "card_score_global": "Score global",
        "pf_profile_logic": "Profil · Logique",
        "pf_profile_score": "Profil · Score",
        "badge_divergence": "⚠ Divergence",
        "expander_definitions_formulas": "📖 Définitions & formules",
        "kpi_definitions_md": """
<style>
.kpi-exp h4 { font-size:12px !important; font-weight:700; text-transform:uppercase;
              letter-spacing:.06em; color:#8BAAC8; margin:10px 0 3px; }
.kpi-exp p, .kpi-exp li, .kpi-exp td, .kpi-exp th { font-size:12px !important; }
.kpi-exp code { font-size:11px !important; }
.kpi-exp hr { margin:6px 0; }
</style>
<div class="kpi-exp">

#### Logique de profil *(par flux — import & export séparément)*
Basée sur le **niveau + tendance du HHI** et le **niveau + tendance de la dépendance principale** :

| Niveau HHI | Niveau Dép. principale | Tendance HHI | Tendance Dép. | → Profil |
|---|---|---|---|---|
| Extreme | Critical | — | — | **Critical Dependency** |
| High ou Extreme | High ou Critical | Increasing | Increasing | **Critical Dependency** |
| Extreme | — | — | — | **Vulnerable** |
| — | Critical | — | — | **Vulnerable** |
| High | High | — | — | **Vulnerable** |
| High | Moderate | Increasing | — | **Vulnerable** |
| Moderate | High | — | Increasing | **Vulnerable** |
| Moderate | Moderate | Increasing | Increasing | **Vulnerable** |
| Moderate ou High | — | — | — | **Stable but Concentrated** |
| Low | Low | — | — | **Robust** |

#### Logique de profil globale *(import + export combinés)*

| Profil Import | Profil Export | → Profil global |
|---|---|---|
| Critical | Critical | **Critical Dependency** |
| Critical | quelconque | **Vulnerable** |
| quelconque | Critical | **Vulnerable** |
| Vulnerable | quelconque | **Vulnerable** |
| Robust | Robust | **Robust** |
| — | — | **Stable but Concentrated** |

---

#### Score global
`score = 0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`

*(HHI_norm = moyenne des normes HHI import & export)*

#### Profil par score *(à partir du score global)*

| Profil | Score |
|---|---|
| **Robust** | < 0.25 |
| **Stable but Concentrated** | 0.25 – 0.35 |
| **Vulnerable** | 0.35 – 0.45 |
| **Critical Dependency** | ≥ 0.45 |

---

#### Référence des seuils

| Indicateur | Low | Moderate | High | Extreme/Critical |
|---|---|---|---|---|
| **HHI** | < 0.08 | 0.08 – 0.15 | 0.15 – 0.25 | ≥ 0.25 |
| **Dépendance principale** | < 0.15 | 0.15 – 0.25 | 0.25 – 0.45 | ≥ 0.45 |
| **BDIS** | < 0.20 (Interdep.) | 0.20 – 0.50 | 0.50 – 0.80 | ≥ 0.80 |

**Tendance** : Stable (|Δ| ≤ 0.025) · Increasing (Δ > 0.025) · Decreasing (Δ < −0.025)

**Part critique** : part du commerce total (X+M) avec les partenaires dont la dépendance bilatérale ≥ 15%

**N critique** : nombre de partenaires avec une dépendance bilatérale ≥ 15%

</div>
        """,
        "section_risk_space": "G.9.1 · Espace de risque annuel — Animé (Import × Export)",
        "countries_to_display_label": "Pays à afficher",
        "lbl_import_risk_score": "Score de risque import",
        "lbl_export_risk_score": "Score de risque export",
        "lbl_profile": "Profil",
        "lbl_total_trade_b": "Commerce total (Md$)",
        "fig_anim_title": "Espace de risque bilatéral — Évolution annuelle",
        "axis_import_risk": "Score de risque import (plus élevé = plus risqué) →",
        "axis_export_risk": "Score de risque export (plus élevé = plus risqué) →",
        "caption_anim": "▲ = première année · ● = dernière année · les lignes pointillées montrent la trajectoire du pays",
        "anim_frame_annot": "<b>{country} · {year}</b><br>Score : <b>{score}</b>",
        "section_profile_heatmap": "G.9.2 · Carte de chaleur des profils — Pays × année",
        "fig_phm_title": "G.9.2 · Profil bilatéral annuel (logique) — Pays × année",
        "section_divergence": "G.9.3 · Divergence logique vs score — {country}",
        "trace_logic_profile": "Profil logique",
        "trace_score_profile": "Profil score",
        "hover_logic_prefix": "Logique",
        "hover_profile": "Profil :",
        "hover_convergent": "Convergent :",
        "fig_div_title": "{country} — Profil logique vs profil score par année",
        "divergence_summary": "Logique ≠ Score sur <b>{n}</b> des {total} années ({pct:.0f}%) — {status}",
        "status_frequent": "⚠ Divergence fréquente",
        "status_convergent": "✓ Majoritairement convergent",
        "no_annual_data": "Aucune donnée annuelle pour {country}.",
        "section_drift_stability": "G.10 · Période complète — Dérive & stabilité",
        "pills_add_columns_label": "Ajouter des colonnes",
        "col_country": "Pays",
        "col_profile_full_period": "Profil période complète",
        "col_dominant_annual_score": "Dominant annuel (score)",
        "col_dominant_annual_logic": "Dominant annuel (logique)",
        "col_divergence_full_vs_annual": "Divergence complète vs annuelle",
        "col_divergence_category": "Catégorie de divergence",
        "col_convergence_score_logic": "Convergence score-logique",
        "col_profile_stability": "Stabilité du profil",
        "col_stability_score": "Score de stabilité",
        "col_profile_drift": "Dérive du profil",
        "col_drift_magnitude": "Ampleur de la dérive",
        "col_hhi_median": "HHI (médiane)",
        "col_top_partner_dependency": "Dépendance au partenaire principal",
        "col_critical_partners_share": "Part des partenaires critiques",
        "col_n_critical_partners": "N partenaires critiques",
        "col_bdis_median": "BDIS (médiane)",
        "col_score_full_period": "Score période complète",
        "opt_drift_stability_conv": "Dérive, stabilité & conv.",
        "opt_indicators": "Indicateurs",
        "expander_column_definitions": "📖 Définitions des colonnes & formules",
        "fp_definitions_md": """
<style>
.fp-exp h4 { font-size:12px !important; font-weight:700; text-transform:uppercase;
             letter-spacing:.06em; color:#8BAAC8; margin:10px 0 3px; }
.fp-exp p, .fp-exp li, .fp-exp td, .fp-exp th { font-size:12px !important; }
.fp-exp code { font-size:11px !important; }
.fp-exp hr { margin:6px 0; }
</style>
<div class="fp-exp">

#### Toujours visible

**Profil période complète** — classification de risque calculée à partir des valeurs médianes des indicateurs sur la période complète.
Seuils : < 0.25 Robust · 0.25–0.35 Stable but Concentrated · 0.35–0.45 Vulnerable · ≥ 0.45 Critical Dependency

**Formule du score période complète** :
`score = 0.35 × critical_share_norm + 0.25 × HHI_norm + 0.20 × dep_top_norm + 0.10 × n_critical_norm + 0.10 × BDIS_norm`
*(toutes les composantes sont des valeurs médianes période complète normalisées inter-pays)*

**Dominant annuel (score)** — le profil de risque apparu le plus fréquemment sur les années (méthode basée sur le score)

**Dominant annuel (logique)** — le profil de risque apparu le plus fréquemment sur les années (méthode basée sur la logique)
Si les deux dominants diffèrent → les deux méthodes ont structurellement divergé sur la période.

**Divergence complète vs annuelle** — part des années où le profil de risque annuel diffère du *Profil période complète*.
Valeur élevée = le pays a changé de profil fréquemment ; la classification période complète masque une volatilité significative.

**Stabilité du profil** — étiquette résumée : ≥ 80% Very Stable · ≥ 60% Stable · ≥ 40% Moderate · < 40% Unstable

**Dérive du profil** — compare le profil des 3 dernières années au profil période complète : *Improving · Stable · Worsening*

---

#### Dérive, stabilité & conv. (bascule)

**Catégorie de divergence** — lecture catégorielle de *Divergence complète vs annuelle* :
Never (0%) · Rare (≤ 20%) · Frequent (≤ 50%) · Systematic (> 50%)

**Convergence score-logique** — part des années où la méthode score et la méthode logique ont donné le même profil.
Valeur élevée = les deux approches sont cohérentes ; valeur basse = le pays est dans une zone grise méthodologique.

**Score de stabilité** — proportion brute derrière *Stabilité du profil* (0 à 1)

**Ampleur de la dérive** — nombre de niveaux de profil décalés entre la période récente et la période complète (0 = aucun décalage, 3 = maximum)

---

#### Indicateurs (bascule) — toutes les valeurs sont des médianes période complète

**HHI (médiane)** — indice de concentration commerciale, moyenne du HHI import & export

**Dépendance au partenaire principal** — part du commerce total détenue par le partenaire le plus important

**Part des partenaires critiques** — part du commerce total avec les partenaires dont la dépendance bilatérale ≥ 15%

**N partenaires critiques** — nombre de partenaires avec une dépendance bilatérale ≥ 15%

**BDIS (médiane)** — score de déséquilibre de dépendance bilatérale, moyenné sur tous les partenaires

**Score période complète** — score de risque composite calculé à partir des valeurs médianes période complète (même formule que ci-dessus)

</div>
        """,
    },
}

def _fmt(x):
    return format_value(x)

TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
TRADE_FIGS  = ROOT / "Analysis" / "Figures" / "Trade"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_bi_trade      = pd.read_csv(TRADE_FILES / "df_bi_trade.csv")
    df_bi_hhi        = pd.read_csv(TRADE_FILES / "df_bi_hhi.csv")
    df_bdi           = pd.read_csv(TRADE_FILES / "df_bdi.csv")
    df_bdi_str       = pd.read_csv(TRADE_FILES / "df_bdi_structural.csv")
    df_combined      = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_dep_top       = pd.read_csv(TRADE_FILES / "df_dependency_top.csv")

    # Compute hhi_avg for KPIs
    df_combined["hhi_avg"] = (df_combined["hhi_import"] + df_combined["hhi_export"]) / 2

    # Compute critical_share per country/year from df_bdi
    df_crit = (df_bdi[df_bdi["bdi_level"].isin(["Critical","High Dependance"])]
               .groupby(["country_name","year"])
               .agg(n_critical=("partner_name","count"))
               .reset_index())
    df_bdi_str = df_bdi_str.merge(df_crit, on=["country_name","year"], how="left")
    df_bdi_str["n_critical"] = df_bdi_str["n_critical"].fillna(0).astype(int)

    # Top partner per country/year (import flow, highest share)
    df_top_partner = (df_bi_trade[df_bi_trade["flow_code"] == "M"]
                      .sort_values("share", ascending=False)
                      .groupby(["country_name","year"])
                      .first()
                      .reset_index()[["country_name","year","partner_name","share"]]
                      .rename(columns={"partner_name":"top_import_partner",
                                       "share":"top_share"}))
    df_bdi_str = df_bdi_str.merge(df_top_partner, on=["country_name","year"], how="left")

    df_final        = pd.read_csv(TRADE_FILES / "df_final.csv")
    df_full_periode = pd.read_csv(TRADE_FILES / "df_full_periode.csv")

    return df_bi_trade, df_bi_hhi, df_bdi, df_bdi_str, df_combined, df_dep_top, df_final, df_full_periode

df_bi_trade, df_bi_hhi, df_bdi, df_bdi_str, df_combined, df_dep_top, df_final, df_full_periode = load_data()

COUNTRIES = sorted(df_bi_trade["country_name"].dropna().unique().tolist())
YEARS     = sorted(df_bi_trade["year"].dropna().unique())
YEAR_MAX  = int(df_bi_hhi["year"].max())

PROFILE_COLORS = {
    "Robust":                  "#10B981",
    "Stable":                  "#60A5FA",
    "Stable but Concentrated": "#A78BFA",
    "Vulnerable":              "#F59E0B",
    "Critical":                "#EF4444",
}
BDIS_COLORS = {
    "Interdependence":     "#10B981",
    "Moderate Dependance": "#60A5FA",
    "High Dependance":     "#F59E0B",
    "Critical Dependance": "#EF4444",
}
HHI_COLORS = {
    "Low":      "#10B981",
    "Moderate": "#60A5FA",
    "High":     "#F59E0B",
    "Extreme":  "#C026D3",
}

# ════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1F3050", plot_bgcolor="#162035",
    font=dict(family="IBM Plex Sans", color="#C4D5E8", size=13),
    title_font=dict(color="#EEF4FB", size=15),
    legend=dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="#304870", linecolor="#304870"),
    yaxis=dict(gridcolor="#304870", linecolor="#304870"),
)
PLC = {k: v for k, v in PLOTLY_LAYOUT.items()
       if k not in ("margin","yaxis","xaxis","legend","paper_bgcolor","plot_bgcolor")}

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR + FULL PERIOD KPI CARDS
# ════════════════════════════════════════════════════════════════

_PROF_COLORS = {
    "Robust":                  "#10B981",
    "Stable but Concentrated": "#A78BFA",
    "Vulnerable":              "#F59E0B",
    "Critical Dependency":     "#EF4444",
}

col_sel, col_kpis = st.columns([1, 5])
with col_sel:
    default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox(t(STRINGS, "select_country"), COUNTRIES, index=default_idx, key="t5_country")

df_bdi_c  = df_bdi_str[df_bdi_str["country_name"] == country]
df_comb_c = df_combined[df_combined["country_name"] == country]
df_hhi_c  = df_bi_hhi[df_bi_hhi["country_name"] == country]
_fp_c     = df_full_periode[df_full_periode["country_name"] == country]

with col_kpis:
    if not _fp_c.empty:
        _fp = _fp_c.iloc[0]

        def _hhi_border(v):
            if v < 0.08: return "#10B981"
            if v < 0.15: return "#A78BFA"
            if v < 0.25: return "#F59E0B"
            return "#EF4444"

        def _bdis_border(v):
            if v < 0.20: return "#10B981"
            if v < 0.50: return "#A78BFA"
            if v < 0.80: return "#F59E0B"
            return "#EF4444"

        def _cs_border(v):
            if v < 0.15: return "#10B981"
            if v < 0.30: return "#A78BFA"
            if v < 0.50: return "#F59E0B"
            return "#EF4444"

        def _nc_border(v):
            if v == 0:  return "#10B981"
            if v <= 1:  return "#F59E0B"
            return "#EF4444"

        def _sc_border(v):
            if v < 0.25: return "#10B981"
            if v < 0.35: return "#A78BFA"
            if v < 0.45: return "#F59E0B"
            return "#EF4444"

        def _top_card(label, value, clr):
            return (
                f'<div style="background:#1F3050;border:1px solid #304870;'
                f'border-top:3px solid {clr};border-radius:10px;'
                f'padding:.5rem .7rem;text-align:center;">'
                f'<div style="font-size:10px;font-weight:600;color:#6890B0;'
                f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:18px;font-weight:700;color:{clr};line-height:1.1;">{value}</div>'
                f'</div>'
            )

        _hhi_v = float(_fp.get("hhi_avg", 0) or 0)
        _bds_v = float(_fp.get("bdis", 0) or 0)
        _cs_v  = float(_fp.get("critical_share", 0) or 0)
        _nc_v  = int(float(_fp.get("n_critical", 0) or 0))
        _sc_v  = float(_fp.get("score_full_periode", 0) or 0)
        _prof  = str(_fp.get("profile_full_periode", "—"))
        _pclr  = _PROF_COLORS.get(_prof, "#6B7280")

        _kc = st.columns(6)
        _kc[0].markdown(_top_card(t(STRINGS, "kpi_hhi_median"),        f"{_hhi_v:.3f}", _hhi_border(_hhi_v)),  unsafe_allow_html=True)
        _kc[1].markdown(_top_card(t(STRINGS, "kpi_bdis_median"),       f"{_bds_v:.3f}", _bdis_border(_bds_v)), unsafe_allow_html=True)
        _kc[2].markdown(_top_card(t(STRINGS, "kpi_critical_share"),    f"{_cs_v:.1%}",  _cs_border(_cs_v)),    unsafe_allow_html=True)
        _kc[3].markdown(_top_card(t(STRINGS, "kpi_n_critical"),        str(_nc_v),      _nc_border(_nc_v)),    unsafe_allow_html=True)
        _kc[4].markdown(_top_card(t(STRINGS, "kpi_score_full_period"), f"{_sc_v:.3f}",  _sc_border(_sc_v)),    unsafe_allow_html=True)
        _kc[5].markdown(_top_card(t(STRINGS, "kpi_profile_full_period"), _prof,         _pclr),                unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# YEAR SELECTOR (linked to tabs)
# ════════════════════════════════════════════════════════════════

_yr_col, _ = st.columns([1, 4])
with _yr_col:
    _yr_list    = YEARS[::-1]
    _yr_default = _yr_list.index(2024) if 2024 in _yr_list else 0
    sel_year    = st.selectbox(t(STRINGS, "year_label"), _yr_list, index=_yr_default, key="t5_year")

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t(STRINGS, "tab1_label"),
    t(STRINGS, "tab2_label"),
    t(STRINGS, "tab3_label"),
    t(STRINGS, "tab4_label"),
    t(STRINGS, "tab5_label"),
])


# ── TAB 1: SANKEY & TOP PARTNERS ─────────────────────────────
with tab1:
    if df_bi_trade[df_bi_trade["country_name"]==country].empty:
        st.warning(t(STRINGS, "no_bilateral_data").format(country=country))
        st.stop()

    # NOTE: the radio option values are built via t() and reused as the exact
    # comparison values below, so the underlying logic stays correct in both languages.
    _OPT_BOTH = t(STRINGS, "opt_both")
    _OPT_IMPORTS_ONLY = t(STRINGS, "opt_imports_only")
    _OPT_EXPORTS_ONLY = t(STRINGS, "opt_exports_only")

    col1, col2 = st.columns([1, 1])
    with col1:
        top_n = st.selectbox(t(STRINGS, "top_n_partners_label"), [5, 10, 15], index=0, key="sankey_n")
    with col2:
        sankey_flow = st.radio(t(STRINGS, "show_label"), [_OPT_BOTH, _OPT_IMPORTS_ONLY, _OPT_EXPORTS_ONLY], key="sankey_flow")

    year_s = sel_year

    df_s_year = df_bi_trade[
        (df_bi_trade["country_name"] == country) & (df_bi_trade["year"] == year_s)
    ].copy()

    if sankey_flow == _OPT_IMPORTS_ONLY:
        df_s_year = df_s_year[df_s_year["flow_code"] == "M"]
    elif sankey_flow == _OPT_EXPORTS_ONLY:
        df_s_year = df_s_year[df_s_year["flow_code"] == "X"]

    df_imp = (df_s_year[df_s_year["flow_code"]=="M"]
              .groupby("partner_name")["value"].sum()
              .nlargest(top_n).reset_index()
              if sankey_flow != _OPT_EXPORTS_ONLY else pd.DataFrame())
    df_exp = (df_s_year[df_s_year["flow_code"]=="X"]
              .groupby("partner_name")["value"].sum()
              .nlargest(top_n).reset_index()
              if sankey_flow != _OPT_IMPORTS_ONLY else pd.DataFrame())

    # Sankey
    if not df_imp.empty or not df_exp.empty:
        imp_partners = df_imp["partner_name"].tolist() if not df_imp.empty else []
        exp_partners = df_exp["partner_name"].tolist() if not df_exp.empty else []
        all_nodes    = list(dict.fromkeys(imp_partners + [country] + exp_partners))
        node_idx     = {n: i for i, n in enumerate(all_nodes)}
        reporter_idx = node_idx[country]

        sources, targets, values, link_colors = [], [], [], []
        for _, r in (df_imp.iterrows() if not df_imp.empty else iter([])):
            p = r["partner_name"]
            if p not in node_idx: continue
            sources.append(node_idx[p]); targets.append(reporter_idx)
            values.append(r["value"]/1e9); link_colors.append("rgba(96,165,250,0.35)")
        for _, r in (df_exp.iterrows() if not df_exp.empty else iter([])):
            p = r["partner_name"]
            if p not in node_idx: continue
            sources.append(reporter_idx); targets.append(node_idx[p])
            values.append(r["value"]/1e9); link_colors.append("rgba(245,158,11,0.35)")

        node_colors = ["#10B981" if n==country else "#60A5FA" if n in imp_partners else "#F59E0B"
                       for n in all_nodes]
        fig_sk = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(pad=15, thickness=20, label=all_nodes, color=node_colors,
                      line=dict(color="#304870", width=0.5),
                      hovertemplate="<b>%{label}</b><extra></extra>"),
            link=dict(source=sources, target=targets, value=values, color=link_colors,
                      hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                                    "%{value:.2f}B USD<extra></extra>"),
        ))
        fig_sk.update_layout(
            paper_bgcolor="#1F3050",
            font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
            title=dict(text=t(STRINGS, "sankey_title").format(country=country, n=top_n, year=year_s),
                       font=dict(color="#EEF4FB", size=15)),
            height=520, margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_sk, width='stretch')

    thin_divider()
    section_label(t(STRINGS, "top_partners_bar_section"))

    # NOTE: "M"/"X" flow-code letters stay inside the translated option text,
    # so the substring check below keeps working in both languages.
    flow_bar = st.radio(t(STRINGS, "flow_label"), [t(STRINGS, "opt_exports_x"), t(STRINGS, "opt_imports_m")], horizontal=True, key="bar_flow")
    fc = "M" if "M" in flow_bar else "X"
    color_bar = "#60A5FA" if fc == "M" else "#F59E0B"

    df_bar = (df_bi_trade[
        (df_bi_trade["country_name"]==country) &
        (df_bi_trade["year"]==year_s) &
        (df_bi_trade["flow_code"]==fc)
    ].nlargest(top_n, "value").sort_values("value").copy())

    if not df_bar.empty:
        df_bar["_fmt_v"] = df_bar["value"].apply(lambda v: f"${_fmt(v)}")
        df_bar["_share_pct"] = df_bar["share"] * 100

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_bar["value"]/1e9, y=df_bar["partner_name"],
            orientation="h",
            marker_color=color_bar, opacity=0.85,
            text=df_bar.apply(lambda r: f"{r['_fmt_v']} ({r['_share_pct']:.1f}%)", axis=1),
            textposition="auto",
            customdata=df_bar[["_fmt_v","_share_pct"]],
            hovertemplate="<b>%{y}</b><br>Value: %{customdata[0]}<br>"
                          "Share: %{customdata[1]:.1f}%<extra></extra>",
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "bar_title").format(country=country, n=top_n, flow=flow_bar, year=year_s),
            xaxis_title=t(STRINGS, "axis_usd_billion"),
            height=max(350, top_n * 38),
        )
        st.plotly_chart(fig_bar, width='stretch')


# ── TAB 2: HHI CONCENTRATION ─────────────────────────────────
with tab2:
    section_label(t(STRINGS, "section_hhi_evolution"))

    # 1. Expander at the top
    with st.expander(t(STRINGS, "expander_what_is_hhi")):
        st.markdown(t(STRINGS, "hhi_md"))

    shock_threshold = st.slider(
        t(STRINGS, "shock_threshold_label"), min_value=0.005, max_value=0.05,
        value=0.01, step=0.005, format="%.3f", key="hhi_shock_thr",
    )

    # 2. Level/Trend — computed for annotation inside the chart
    _lv, _tr, _lv_color = "—", "—", "#C4D5E8"
    if not df_hhi_c.empty:
        _df_exp_sorted = df_hhi_c[df_hhi_c["flow_code"]=="X"].sort_values("year")
        if not _df_exp_sorted.empty:
            _latest   = _df_exp_sorted.iloc[-1]
            _lv       = _latest.get("hhi_level", "—")
            _tr       = _latest.get("hhi_trend",  "—")
            _lv_color = HHI_COLORS.get(_lv, "#C4D5E8")

    # Evolution chart — full width
    fig_hhi = go.Figure()
    for flow, color, label in [("M","#60A5FA",t(STRINGS, "trace_import_hhi")), ("X","#F59E0B",t(STRINGS, "trace_export_hhi"))]:
        df_f = df_hhi_c[df_hhi_c["flow_code"]==flow].sort_values("year")
        if df_f.empty: continue
        fig_hhi.add_trace(go.Scatter(
            x=df_f["year"], y=df_f["hhi"],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5), marker=dict(size=7),
            customdata=df_f[["hhi_level","hhi_trend"]],
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>"
                          "HHI: %{y:.4f}<br>Level: %{customdata[0]}<br>"
                          "Trend: %{customdata[1]}<extra></extra>",
        ))
        # Breaks — fixed threshold (notebook G.5.7: |ΔHHI| >= threshold)
        df_shocks = df_f[df_f["delta_hhi"].abs() >= shock_threshold].dropna(subset=["delta_hhi"])
        if not df_shocks.empty:
            df_up   = df_shocks[df_shocks["delta_hhi"] > 0]
            df_down = df_shocks[df_shocks["delta_hhi"] < 0]
            for df_sh, sym, clr, lbl in [
                (df_up,   "triangle-up",   "#EF4444", f"↑ concentration break {label}"),
                (df_down, "triangle-down", "#10B981", f"↓ diversification break {label}"),
            ]:
                if df_sh.empty: continue
                _delta_fmt = df_sh["delta_hhi"].apply(lambda v: f"{v:+.3f}").tolist()
                fig_hhi.add_trace(go.Scatter(
                    x=df_sh["year"], y=df_sh["hhi"],
                    mode="markers", name=lbl, showlegend=False,
                    marker=dict(symbol=sym, size=14, color=clr,
                                line=dict(width=1.5, color="#EEF4FB")),
                    customdata=[[v] for v in _delta_fmt],
                    hovertemplate="<b>Break</b><br>Year: %{x}<br>"
                                  "Δ HHI: %{customdata[0]}<extra></extra>",
                ))

    for y_val, lbl, clr in [
        (0.25, t(STRINGS, "hline_extreme"), "#C026D3"),
        (0.15, t(STRINGS, "hline_high"),    "#F59E0B"),
        (0.08, t(STRINGS, "hline_moderate"),"#60A5FA"),
    ]:
        fig_hhi.add_hline(y=y_val, line_dash="dash", line_color=clr, line_width=1,
                          annotation_text=lbl, annotation_font_color=clr)

    # 2. Level/Trend as annotation inside the chart
    fig_hhi.add_annotation(
        text=t(STRINGS, "annot_level_trend").format(lv=_lv, tr=_tr),
        xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=11, color=_lv_color),
        bgcolor="#1A2840", bordercolor=_lv_color, borderwidth=1, borderpad=5,
    )

    fig_hhi.update_layout(**PLOTLY_LAYOUT,
        # 3. Title: concentration break / diversification break
        title=t(STRINGS, "hhi_evolution_title").format(country=country),
        yaxis_title=t(STRINGS, "yaxis_hhi"),
    )
    st.plotly_chart(fig_hhi, width='stretch')

    # 4. G.5.5 Scatter — below, full width
    thin_divider()
    section_label(t(STRINGS, "section_hhi_scatter"))

    df_hhi_latest_m = df_bi_hhi[(df_bi_hhi["year"]==sel_year) &
                                 (df_bi_hhi["flow_code"]=="M")][["country_name","hhi","hhi_level"]].rename(
                          columns={"hhi":"hhi_imp","hhi_level":"level_imp"})
    df_hhi_latest_x = df_bi_hhi[(df_bi_hhi["year"]==sel_year) &
                                 (df_bi_hhi["flow_code"]=="X")][["country_name","hhi"]].rename(
                          columns={"hhi":"hhi_exp"})
    df_scatter_hhi  = df_hhi_latest_m.merge(df_hhi_latest_x, on="country_name", how="inner")

    # Axis limits: 95th percentile, focuses on non-outlier zone by default
    _x_max = df_scatter_hhi["hhi_imp"].quantile(0.95) * 1.15
    _y_max = df_scatter_hhi["hhi_exp"].quantile(0.95) * 1.15
    # Always show the base country
    _row_base = df_scatter_hhi[df_scatter_hhi["country_name"]==country]
    if not _row_base.empty:
        _x_max = max(_x_max, float(_row_base["hhi_imp"].iloc[0]) * 1.3)
        _y_max = max(_y_max, float(_row_base["hhi_exp"].iloc[0]) * 1.3)

    _default_highlights = [c for c in
        ["USA","China","Germany","Japan","Canada","Brazil","Nigeria","Qatar","India","France"]
        if c in df_scatter_hhi["country_name"].values and c != country]
    highlights = st.multiselect(
        t(STRINGS, "highlight_countries_label"),
        [c for c in sorted(df_scatter_hhi["country_name"].unique()) if c != country],
        default=_default_highlights,
        key="hhi_sc_highlights",
    )

    _all_special = set([country] + highlights)

    fig_hhi_sc = go.Figure()
    df_sc_bg = df_scatter_hhi[~df_scatter_hhi["country_name"].isin(_all_special)]
    fig_hhi_sc.add_trace(go.Scatter(
        x=df_sc_bg["hhi_imp"], y=df_sc_bg["hhi_exp"],
        mode="markers", name=t(STRINGS, "trace_all_countries"),
        marker=dict(color=[HHI_COLORS.get(l,"#6B7280") for l in df_sc_bg["level_imp"]],
                    size=7, opacity=0.4),
        customdata=df_sc_bg[["country_name","level_imp"]],
        hovertemplate="<b>%{customdata[0]}</b><br>HHI Import: %{x:.4f}<br>"
                      "HHI Export: %{y:.4f}<br>Level: %{customdata[1]}<extra></extra>",
        showlegend=False,
    ))
    _hl_colors = ["#F97316","#A78BFA","#34D399","#F472B6","#FBBF24","#38BDF8",
                  "#E879F9","#4ADE80","#FB923C","#818CF8"]
    for i, hl in enumerate(highlights):
        row_hl = df_scatter_hhi[df_scatter_hhi["country_name"] == hl]
        if row_hl.empty: continue
        r = row_hl.iloc[0]
        clr = _hl_colors[i % len(_hl_colors)]
        fig_hhi_sc.add_trace(go.Scatter(
            x=[r["hhi_imp"]], y=[r["hhi_exp"]],
            mode="markers+text", name=hl,
            marker=dict(color=clr, size=12, symbol="circle",
                        line=dict(width=1.5, color="#EEF4FB")),
            text=[hl], textposition="top center",
            textfont=dict(size=10, color=clr),
            hovertemplate=f"<b>{hl}</b><br>HHI Import: {r['hhi_imp']:.4f}<br>"
                          f"HHI Export: {r['hhi_exp']:.4f}<extra></extra>",
        ))
    if not _row_base.empty:
        r = _row_base.iloc[0]
        fig_hhi_sc.add_trace(go.Scatter(
            x=[r["hhi_imp"]], y=[r["hhi_exp"]],
            mode="markers+text", name=country,
            marker=dict(color="#EEF4FB", size=16, symbol="star",
                        line=dict(width=1.5, color="#304870")),
            text=[country], textposition="top center",
            textfont=dict(size=11, color="#EEF4FB"),
            hovertemplate=f"<b>{country}</b><br>HHI Import: {r['hhi_imp']:.4f}<br>"
                          f"HHI Export: {r['hhi_exp']:.4f}<extra></extra>",
        ))
    _diag_max = df_scatter_hhi[["hhi_imp","hhi_exp"]].max().max() * 1.05
    fig_hhi_sc.add_trace(go.Scatter(
        x=[0, _diag_max], y=[0, _diag_max],
        mode="lines", name=t(STRINGS, "trace_import_eq_export"),
        line=dict(color="#304870", dash="dot", width=1.5),
    ))
    fig_hhi_sc.update_layout(**PLC,
        paper_bgcolor="#1F3050", plot_bgcolor="#162035",
        legend=dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
        title=t(STRINGS, "fig_hhi_sc_title").format(year=sel_year),
        xaxis=dict(gridcolor="#304870", linecolor="#304870", title=t(STRINGS, "axis_hhi_import"),
                   range=[0, _x_max]),
        yaxis=dict(gridcolor="#304870", linecolor="#304870", title=t(STRINGS, "axis_hhi_export"),
                   range=[0, _y_max]),
        height=480,
    )
    st.plotly_chart(fig_hhi_sc, width='stretch')
    st.caption(t(STRINGS, "caption_hhi_scatter"))

    # 5. Ranking — Export by default, Most diversified label
    thin_divider()
    section_label(t(STRINGS, "section_hhi_ranking"))
    col_r1, col_r2, col_r3 = st.columns([1, 1, 2])
    # NOTE: "X"/"M" flow codes stay inside the translated option text, so
    # flow_rank.split()[0] keeps returning the correct code in both languages.
    _OPT_CONCENTRATED = t(STRINGS, "opt_most_concentrated")
    _OPT_DIVERSIFIED  = t(STRINGS, "opt_most_diversified")
    with col_r1:
        flow_rank = st.radio(t(STRINGS, "flow_label"), [t(STRINGS, "opt_x_export"), t(STRINGS, "opt_m_import")], horizontal=True, key="hhi_flow_rank")
    with col_r2:
        rank_dir  = st.selectbox(t(STRINGS, "show_label"), [_OPT_CONCENTRATED, _OPT_DIVERSIFIED], key="hhi_dir")
    fc_rank = flow_rank.split()[0]
    df_hhi_rk = df_bi_hhi[(df_bi_hhi["year"]==sel_year) & (df_bi_hhi["flow_code"]==fc_rank)]
    df_rank = (df_hhi_rk.nlargest(10,"hhi") if rank_dir == _OPT_CONCENTRATED
               else df_hhi_rk.nsmallest(10,"hhi")).sort_values("hhi")

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=df_rank["hhi"], y=df_rank["country_name"],
        orientation="h",
        marker_color=[HHI_COLORS.get(l,"#6B7280") for l in df_rank["hhi_level"]],
        text=df_rank["hhi"].apply(lambda v: f"{v:.4f}"),
        textposition="auto", opacity=0.85,
        customdata=df_rank[["hhi_level","hhi_trend"]],
        hovertemplate="<b>%{y}</b><br>HHI: %{x:.4f}<br>"
                      "Level: %{customdata[0]}<br>Trend: %{customdata[1]}<extra></extra>",
    ))
    if country in df_rank["country_name"].values:
        idx = df_rank["country_name"].tolist().index(country)
        fig_rank.add_shape(type="rect", x0=0, x1=df_rank["hhi"].max()*1.1,
                           y0=idx-0.5, y1=idx+0.5, line=dict(color="#EEF4FB", width=1.5))
    fig_rank.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "fig_rank_hhi_title").format(flow_rank=flow_rank, rank_dir=rank_dir, year=sel_year),
        xaxis_title=t(STRINGS, "yaxis_hhi"), height=420,
    )
    st.plotly_chart(fig_rank, width='stretch')

    with st.expander(t(STRINGS, "expander_break_detection")):
        st.markdown(t(STRINGS, "break_detection_md").format(thr=f"{shock_threshold:.3f}"))


# ── TAB 3: BDIS ANALYSIS ─────────────────────────────────────
with tab3:
    section_label(t(STRINGS, "section_bdis_evolution"))

    # Expander at the top
    with st.expander(t(STRINGS, "expander_what_is_bdis")):
        st.markdown(t(STRINGS, "bdis_md"))

    bdis_threshold = st.slider(
        t(STRINGS, "bdis_threshold_label"), min_value=0.01, max_value=0.15,
        value=0.05, step=0.01, format="%.2f", key="bdis_break_thr",
    )

    # Level/Trend — computed from latest year
    _bdis_lv, _bdis_tr, _bdis_lv_color = "—", "—", "#C4D5E8"
    if not df_bdi_c.empty:
        _latest_bdis = df_bdi_c.sort_values("year").iloc[-1]
        _bdis_lv       = _latest_bdis.get("bdis_level", "—")
        _bdis_lv_color = BDIS_COLORS.get(_bdis_lv, "#C4D5E8")
        _delta         = _latest_bdis.get("delta_bdis", None)
        if _delta is None or (isinstance(_delta, float) and np.isnan(_delta)):
            _bdis_tr = "—"
        elif _delta > 0.01:
            _bdis_tr = "Increasing"
        elif _delta < -0.01:
            _bdis_tr = "Decreasing"
        else:
            _bdis_tr = "Stable"

    # BDIS Evolution chart — full width
    df_bdis_srt = df_bdi_c.sort_values("year")
    fig_bdis = go.Figure()
    fig_bdis.add_trace(go.Scatter(
        x=df_bdis_srt["year"], y=df_bdis_srt["bdis"],
        mode="lines+markers", name="BDIS",
        line=dict(color="#F59E0B", width=2.5), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
        customdata=df_bdis_srt[["bdis_level","n_partners"]].assign(
            delta_fmt=df_bdis_srt["delta_bdis"].apply(
                lambda v: f"{v:+.3f}" if not (isinstance(v, float) and np.isnan(v)) else "—"
            )
        ).apply(lambda r: [r["bdis_level"], r["n_partners"], r["delta_fmt"]], axis=1).tolist(),
        hovertemplate="<b>%{x}</b><br>BDIS: %{y:.3f}<br>"
                      "Level: %{customdata[0]}<br>"
                      "Partners: %{customdata[1]}<br>"
                      "Δ: %{customdata[2]}<extra></extra>",
    ))

    # Breaks — fixed threshold
    df_bdis_sh = df_bdis_srt[df_bdis_srt["delta_bdis"].abs() >= bdis_threshold].dropna(subset=["delta_bdis"])
    for df_sh, sym, clr in [
        (df_bdis_sh[df_bdis_sh["delta_bdis"] >= bdis_threshold],  "triangle-up",   "#EF4444"),
        (df_bdis_sh[df_bdis_sh["delta_bdis"] <= -bdis_threshold], "triangle-down", "#10B981"),
    ]:
        if df_sh.empty: continue
        _dfmt = df_sh["delta_bdis"].apply(lambda v: f"{v:+.3f}").tolist()
        fig_bdis.add_trace(go.Scatter(
            x=df_sh["year"], y=df_sh["bdis"],
            mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=14, color=clr,
                        line=dict(width=1.5, color="#EEF4FB")),
            customdata=[[v] for v in _dfmt],
            hovertemplate="<b>Break</b><br>Δ BDIS: %{customdata[0]}<extra></extra>",
        ))

    # Threshold lines
    for y_val, lbl, clr in [
        (0.80, t(STRINGS, "hline_critical"),  "#EF4444"),
        (0.50, t(STRINGS, "hline_high2"),      "#F59E0B"),
        (0.20, t(STRINGS, "hline_moderate2"),  "#60A5FA"),
    ]:
        fig_bdis.add_hline(y=y_val, line_dash="dash", line_color=clr, line_width=1,
                           annotation_text=lbl, annotation_font_color=clr)

    # Level/Trend annotation inside the chart
    fig_bdis.add_annotation(
        text=t(STRINGS, "annot_level_trend").format(lv=_bdis_lv, tr=_bdis_tr),
        xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=11, color=_bdis_lv_color),
        bgcolor="#1A2840", bordercolor=_bdis_lv_color, borderwidth=1, borderpad=5,
    )

    fig_bdis.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "bdis_evolution_title").format(country=country),
        yaxis_title=t(STRINGS, "yaxis_bdis_score"),
    )
    st.plotly_chart(fig_bdis, width='stretch')

    # Heatmap — below, full width
    thin_divider()
    section_label(t(STRINGS, "section_bdis_heatmap"))

    _def_heatmap = [cc for cc in
        ["USA","China","Germany","France","Japan","Canada","Brazil","Nigeria","Qatar","India"]
        if cc in df_bdi_str["country_name"].values]
    hm_countries = st.multiselect(
        t(STRINGS, "countries_label"), sorted(df_bdi_str["country_name"].unique().tolist()),
        default=_def_heatmap, max_selections=15, key="bdis_hm_countries",
    )

    if hm_countries:
        df_hm = df_bdi_str[df_bdi_str["country_name"].isin(hm_countries)].pivot_table(
            index="country_name", columns="year", values="bdis"
        ).reindex(hm_countries)

        fig_hm = go.Figure(go.Heatmap(
            z=df_hm.values,
            x=[str(int(y)) for y in df_hm.columns],
            y=df_hm.index.tolist(),
            colorscale=[[0,"#10B981"],[0.2,"#60A5FA"],[0.5,"#F59E0B"],[0.8,"#EF4444"],[1,"#7C3AED"]],
            zmin=0, zmax=1,
            text=np.round(df_hm.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#EEF4FB"),
            showscale=True,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> — %{x}<br>BDIS: %{z:.3f}<extra></extra>",
        ))
        if country in df_hm.index.tolist():
            idx = df_hm.index.tolist().index(country)
            fig_hm.add_shape(type="rect",
                x0=-0.5, x1=len(df_hm.columns)-0.5,
                y0=idx-0.5, y1=idx+0.5,
                line=dict(color="#EEF4FB", width=2))
        fig_hm.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title=t(STRINGS, "section_bdis_heatmap"),
            xaxis=dict(type="category", gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"),
            legend=dict(bgcolor="#1F3050"),
            margin=dict(l=120, r=20, t=50, b=40),
            height=max(350, len(hm_countries)*35+80),
        )
        st.plotly_chart(fig_hm, width='stretch')


# ── TAB 4: BDI / DEPENDENCY ──────────────────────────────────
with tab4:

    # ── Définitions (expander en haut) ───────────────────────────
    with st.expander(t(STRINGS, "expander_bdi_definitions")):
        st.markdown(t(STRINGS, "bdi_def_md"))

    BDI_LV_COLORS = {
        "Interdependence":    "#10B981",
        "Moderate Dependance":"#60A5FA",
        "High Dependance":    "#F59E0B",
        "Critical Dependance":"#EF4444",
    }

    df_bdi_c_yr = df_bdi[
        (df_bdi["country_name"] == country) & (df_bdi["year"] == sel_year)
    ].copy()

    # ── ② G.6.3 · Mirror Bar Chart — Dependency Map (NOUVEAU) ────
    section_label(t(STRINGS, "section_dependency_map").format(year=sel_year))
    n_bdi = st.selectbox(t(STRINGS, "top_n_partners_label"), [5, 10, 15, 20], index=1, key="bdi_n")

    if df_bdi_c_yr.empty:
        st.info(t(STRINGS, "no_bdi_data").format(country=country, year=sel_year))
    else:
        df_mirror = df_bdi_c_yr.nlargest(n_bdi, "dependency").sort_values("dependency", ascending=True)

        _lv_m  = df_mirror["bdi_level"].tolist()
        _dep_m = df_mirror["dependency"].apply(lambda v: f"{v:.4f}").tolist()
        _rev_m = df_mirror["dependency_reverse"].apply(lambda v: f"{v:.4f}").tolist()
        _bdi_m = df_mirror["bdi_norm"].apply(lambda v: f"{v:.3f}").tolist()
        _dt_m  = df_mirror["dependency_type"].tolist()

        # Axe X symétrique avec valeurs absolues (pas de signe négatif)
        _xmax = max(df_mirror["dependency"].max(), df_mirror["dependency_reverse"].max())
        _step = 0.05 if _xmax < 0.25 else 0.10
        _nt   = int(_xmax / _step) + 2
        _tv   = [round(i * _step, 3) for i in range(-_nt, _nt + 1)]
        _tt   = [f"{int(abs(v)*100)}%" for v in _tv]

        # Alpha baked into bar colors — text reste 100% opaque
        BDI_LV_RGBA = {
            "Interdependence":    "rgba(16,185,129,0.85)",
            "Moderate Dependance":"rgba(96,165,250,0.85)",
            "High Dependance":    "rgba(245,158,11,0.85)",
            "Critical Dependance":"rgba(239,68,68,0.85)",
        }

        fig_mirror = go.Figure()

        fig_mirror.add_trace(go.Bar(
            x=df_mirror["dependency"].tolist(),
            y=df_mirror["partner_name"].tolist(),
            orientation="h",
            showlegend=False,
            marker_color=[BDI_LV_RGBA.get(lv, "rgba(107,114,128,0.85)") for lv in _lv_m],
            text=[f"{v*100:.1f}%" for v in df_mirror["dependency"].tolist()],
            textposition="outside",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Mono"),
            customdata=[[_lv_m[i], _dep_m[i], _bdi_m[i]] for i in range(len(df_mirror))],
            hovertemplate="<b>%{y}</b><br>" + t(STRINGS, "hover_we_depend") + " %{customdata[1]}<br>"
                          + t(STRINGS, "hover_bdi_level") + " %{customdata[0]}<br>" + t(STRINGS, "hover_bdi_norm") + " %{customdata[2]}<extra></extra>",
        ))
        fig_mirror.add_trace(go.Bar(
            x=[-v for v in df_mirror["dependency_reverse"].tolist()],
            y=df_mirror["partner_name"].tolist(),
            orientation="h",
            showlegend=False,
            marker_color="rgba(59,130,246,0.50)",
            text=[f"{v*100:.1f}%" for v in df_mirror["dependency_reverse"].tolist()],
            textposition="outside",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Mono"),
            customdata=[[_rev_m[i], _dt_m[i]] for i in range(len(df_mirror))],
            hovertemplate="<b>%{y}</b><br>" + t(STRINGS, "hover_they_depend") + " %{customdata[0]}<br>"
                          + t(STRINGS, "hover_relationship") + " %{customdata[1]}<extra></extra>",
        ))

        # Centre + seuils ±20 % / ±50 %
        fig_mirror.add_vline(x=0, line_color="#EEF4FB", line_width=1.5)
        for thr, col, lbl in [(0.20, "#F59E0B", "20%"), (0.50, "#EF4444", "50%")]:
            fig_mirror.add_vline(x= thr, line_dash="dash", line_color=col,
                                 line_width=1, opacity=0.65,
                                 annotation_text=lbl, annotation_position="top",
                                 annotation_font_color=col, annotation_font_size=9)
            fig_mirror.add_vline(x=-thr, line_dash="dash", line_color="#3B82F6",
                                 line_width=1, opacity=0.40)

        # Annotations BDI colonne droite
        for _, row in df_mirror.iterrows():
            bdi_col = BDI_LV_COLORS.get(row["bdi_level"], "#6B7280")
            fig_mirror.add_annotation(
                x=1.02, xref="paper",
                y=row["partner_name"], yref="y",
                text=f"BDI: {row['bdi_norm']:.3f}",
                showarrow=False, xanchor="left",
                font=dict(size=9, color=bdi_col, family="IBM Plex Mono"),
                bgcolor="#1A2840", bordercolor=bdi_col,
                borderwidth=1, borderpad=3,
            )

        fig_mirror.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            barmode="overlay",
            bargap=0.25,
            showlegend=False,
            title=t(STRINGS, "fig_mirror_title").format(country=country, n=n_bdi, year=sel_year),
            xaxis=dict(
                title="",
                gridcolor="#304870",
                tickvals=_tv, ticktext=_tt,
            ),
            yaxis=dict(gridcolor="#304870"),
            margin=dict(l=140, r=120, t=50, b=40),
            height=max(420, n_bdi * 50),
        )
        st.plotly_chart(fig_mirror, width='stretch')

        # ── Légende BDI (4 niveaux, sans partner→country) ──────────
        _lv_entries = [
            ("rgba(16,185,129,0.85)",  "Interdependence (BDI < 0.20)"),
            ("rgba(96,165,250,0.85)",  "Moderate Dependance (0.20 – 0.50)"),
            ("rgba(245,158,11,0.85)",  "High Dependance (0.50 – 0.80)"),
            ("rgba(239,68,68,0.85)",   "Critical Dependance (≥ 0.80)"),
        ]
        _squares = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:18px;">'
            f'<span style="width:12px;height:12px;border-radius:2px;'
            f'background:{c};flex-shrink:0;"></span>'
            f'<span style="color:#C4D5E8;">{lbl}</span></span>'
            for c, lbl in _lv_entries
        )
        # ── Direction line (en haut) ─────────────────────────────────
        st.markdown(
            f'<div style="text-align:center;font-size:12px;color:#7090B0;'
            f'font-family:\'IBM Plex Mono\',monospace;padding:4px 0 2px 0;">'
            f'{t(STRINGS, "direction_line").format(country=country)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── BDI level legend (en bas) ────────────────────────────────
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;justify-content:center;'
            f'padding:2px 0 8px 0;font-size:12px;font-family:\'IBM Plex Mono\',monospace;">'
            f'{_squares}</div>',
            unsafe_allow_html=True,
        )

        with st.expander(t(STRINGS, "expander_how_to_read")):
            st.markdown(t(STRINGS, "how_to_read_md").format(country=country))

    # Previous bar chart kept for comparison
    with st.expander(t(STRINGS, "expander_previous_chart")):
        if not df_bdi_c_yr.empty:
            col1_cmp, col2_cmp = st.columns([1, 3])
            # NOTE: option values built via t() and reused in the comparison below,
            # so the logic keeps working regardless of the selected language.
            _OPT_DEP_ON = t(STRINGS, "opt_most_dependent_on")
            _OPT_DEP_US = t(STRINGS, "opt_most_depends_on_us")
            with col1_cmp:
                flow_bdi = st.radio(
                    t(STRINGS, "focus_label"), [_OPT_DEP_ON, _OPT_DEP_US], key="bdi_flow")
            if flow_bdi == _OPT_DEP_ON:
                df_bdi_top = df_bdi_c_yr.nlargest(n_bdi, "dependency").sort_values("dependency")
                x_col, x_label = "dependency", t(STRINGS, "xlabel_dependency")
            else:
                df_bdi_top = df_bdi_c_yr.nlargest(n_bdi, "dependency_reverse").sort_values("dependency_reverse")
                x_col, x_label = "dependency_reverse", t(STRINGS, "xlabel_reverse_dependency")
            _lv_o  = df_bdi_top["bdi_level"].tolist()
            _dt_o  = df_bdi_top["dependency_type"].tolist()
            _bn_o  = df_bdi_top["bdi_norm"].apply(lambda v: f"{v:.4f}").tolist()
            with col2_cmp:
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    x=df_bdi_top[x_col].tolist(),
                    y=df_bdi_top["partner_name"].tolist(),
                    orientation="h",
                    marker_color=[BDI_LV_COLORS.get(lv, "#6B7280") for lv in _lv_o],
                    text=df_bdi_top[x_col].apply(lambda v: f"{v:.4f}"),
                    textposition="auto", opacity=0.85,
                    customdata=[[_lv_o[i], _dt_o[i], _bn_o[i]] for i in range(len(df_bdi_top))],
                    hovertemplate="<b>%{y}</b><br>Value: %{x:.4f}<br>"
                                  "BDI Level: %{customdata[0]}<br>"
                                  "Type: %{customdata[1]}<br>"
                                  "BDI norm: %{customdata[2]}<extra></extra>",
                ))
                fig_cmp.update_layout(**PLC,
                    paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                    title=t(STRINGS, "fig_cmp_title").format(country=country, flow=flow_bdi, n=n_bdi, year=sel_year),
                    xaxis=dict(title=x_label, gridcolor="#304870"),
                    yaxis=dict(gridcolor="#304870"),
                    legend=dict(bgcolor="#1F3050"),
                    margin=dict(l=130, r=20, t=50, b=40),
                    height=max(350, n_bdi * 38),
                )
                st.plotly_chart(fig_cmp, width='stretch')

    thin_divider()

    # ── ③ G.6.2 · Pair Evolution — Deep Dive (NOUVEAU) ───────────
    section_label(t(STRINGS, "section_pair_evolution"))

    # Default = top partner by trade share (dependency)
    _bp_row   = (df_bdi[(df_bdi["country_name"] == country) & (df_bdi["year"] == sel_year)]
                 .nlargest(1, "dependency"))
    _def_part = _bp_row["partner_name"].values[0] if len(_bp_row) > 0 else None
    _all_part = sorted(df_bdi[df_bdi["country_name"] == country]["partner_name"].unique().tolist())
    _pidx     = _all_part.index(_def_part) if _def_part in _all_part else 0

    sel_partner = st.selectbox(t(STRINGS, "select_partner"), _all_part, index=_pidx, key=f"bdi_pair_partner_{country}")

    df_pair = (df_bdi[(df_bdi["country_name"] == country) & (df_bdi["partner_name"] == sel_partner)]
               .sort_values("year").copy())

    if df_pair.empty:
        st.info(t(STRINGS, "no_data_pair").format(country=country, partner=sel_partner))
    else:
        fig_pair = make_subplots(specs=[[{"secondary_y": True}]])

        _yrs  = df_pair["year"].tolist()
        # Percentages — consistent with Dependency Map
        _dep_pct  = [v * 100 for v in df_pair["dependency"].tolist()]
        _rev_pct  = [v * 100 for v in df_pair["dependency_reverse"].tolist()]
        _dep2 = df_pair["dependency"].apply(lambda v: f"{v*100:.1f}%").tolist()
        _rev2 = df_pair["dependency_reverse"].apply(lambda v: f"{v*100:.1f}%").tolist()
        _bdi2 = df_pair["bdi"].apply(lambda v: f"{v:.3f}").tolist()

        # Y-axis: absolute percentage ticks (no negative sign)
        _y_max  = max(df_pair["dependency"].max(), df_pair["dependency_reverse"].max()) * 100
        _y_step = 5 if _y_max < 25 else 10
        _y_nt   = int(_y_max / _y_step) + 2
        _y_tv   = [round(i * _y_step, 1) for i in range(-_y_nt, _y_nt + 1)]
        _y_tt   = [f"{int(abs(v))}%" for v in _y_tv]

        fig_pair.add_trace(go.Bar(
            x=_yrs,
            y=_dep_pct,
            name=f"{country} → {sel_partner}",
            marker_color="rgba(239,68,68,0.82)",
            customdata=[[v] for v in _dep2],
            hovertemplate=f"<b>{country} → {sel_partner}</b><br>%{{x}}: %{{customdata[0]}}<extra></extra>",
        ), secondary_y=False)

        fig_pair.add_trace(go.Bar(
            x=_yrs,
            y=[-v for v in _rev_pct],
            name=f"{sel_partner} → {country}",
            marker_color="rgba(16,185,129,0.82)",
            customdata=[[v] for v in _rev2],
            hovertemplate=f"<b>{sel_partner} → {country}</b><br>%{{x}}: %{{customdata[0]}}<extra></extra>",
        ), secondary_y=False)

        fig_pair.add_trace(go.Scatter(
            x=_yrs,
            y=df_pair["bdi"].tolist(),
            name="BDI",
            mode="lines+markers",
            line=dict(color="#A78BFA", width=2.5),
            marker=dict(size=7),
            customdata=[[v] for v in _bdi2],
            hovertemplate="BDI: %{customdata[0]}<extra></extra>",
        ), secondary_y=True)

        # Threshold lines: colors match BDI level palette (not bar colors)
        for thresh, color, lbl in [(0.20, "#60A5FA", t(STRINGS, "hline_moderate3")),
                                    (0.50, "#F59E0B", t(STRINGS, "hline_high3"))]:
            fig_pair.add_trace(go.Scatter(
                x=[_yrs[0], _yrs[-1]], y=[thresh, thresh],
                mode="lines",
                line=dict(color=color, width=1.2, dash="dash"),
                name=lbl, hoverinfo="skip", showlegend=True,
            ), secondary_y=True)

        last_dom = df_pair["dominant_partner"].iloc[-1]
        last_lv  = df_pair["bdi_level"].iloc[-1]
        last_bdi_val = df_pair["bdi"].iloc[-1]
        _annot_color = BDI_LV_COLORS.get(last_lv, "#6B7280")

        fig_pair.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            barmode="overlay",
            title=t(STRINGS, "fig_pair_title").format(country=country, partner=sel_partner),
            xaxis=dict(gridcolor="#304870", tickmode="array", tickvals=_yrs, title=t(STRINGS, "year_label")),
            yaxis=dict(gridcolor="#304870", title=t(STRINGS, "yaxis_dependency_pct"),
                       tickvals=_y_tv, ticktext=_y_tt),
            yaxis2=dict(range=[0, 1.05], title="BDI",
                        gridcolor="rgba(0,0,0,0)", showgrid=False),
            legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
            margin=dict(l=70, r=80, t=80, b=40),
            height=480,
        )
        # Level / Dominant — centré, visible, coloré par niveau
        fig_pair.add_annotation(
            xref="paper", yref="paper", x=0.5, y=1.06,
            text=t(STRINGS, "annot_level_bdi_dominant").format(lv=last_lv, bdi=f"{last_bdi_val:.3f}", dom=last_dom),
            showarrow=False,
            font=dict(size=13, color=_annot_color, family="IBM Plex Sans"),
            bgcolor="#1A2840", bordercolor=_annot_color,
            borderwidth=1.5, borderpad=6,
            xanchor="center", yanchor="bottom",
        )
        st.plotly_chart(fig_pair, width='stretch')

    thin_divider()

    # ── ④ BDI Heatmap — Partner × Year (conservé) ────────────────
    section_label(t(STRINGS, "section_bdi_heatmap_partner"))

    _df_hm_yr = df_bdi[(df_bdi["country_name"]==country) & (df_bdi["year"]==sel_year)]
    _top5_dep  = _df_hm_yr.nlargest(5, "dependency")["partner_name"].tolist()
    _top5_rev  = _df_hm_yr.nlargest(5, "dependency_reverse")["partner_name"].tolist()
    # Merge preserving order: dep first, then rev partners not already included
    _def_partners = _top5_dep + [p for p in _top5_rev if p not in _top5_dep]
    sel_partners = st.multiselect(
        t(STRINGS, "select_partners_label"),
        sorted(df_bdi[df_bdi["country_name"]==country]["partner_name"].unique().tolist()),
        default=_def_partners,
        max_selections=15, key=f"bdi_partners_{country}",
    )

    if sel_partners:
        df_bdi_hm = df_bdi[
            (df_bdi["country_name"]==country) &
            (df_bdi["partner_name"].isin(sel_partners))
        ].pivot_table(index="partner_name", columns="year", values="bdi_norm")
        df_bdi_hm = df_bdi_hm.reindex(sel_partners)

        fig_bdi_hm = go.Figure(go.Heatmap(
            z=df_bdi_hm.values,
            x=[str(int(y)) for y in df_bdi_hm.columns],
            y=df_bdi_hm.index.tolist(),
            colorscale=[[0,"#10B981"],[0.33,"#60A5FA"],
                         [0.66,"#F59E0B"],[1,"#EF4444"]],
            zmin=0, zmax=1,
            text=np.round(df_bdi_hm.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#EEF4FB"),
            showscale=True,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> — %{x}<br>BDI norm: %{z:.3f}<extra></extra>",
        ))
        fig_bdi_hm.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title=t(STRINGS, "fig_bdi_hm_title").format(country=country),
            xaxis=dict(type="category", gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"),
            legend=dict(bgcolor="#1F3050"),
            margin=dict(l=120, r=20, t=50, b=40),
            height=max(350, len(sel_partners)*35+80),
        )
        st.plotly_chart(fig_bdi_hm, width='stretch')

    thin_divider()

    # ── ⑤ G.8 · Dependency Heatmap — Countries × Years ──────────
    section_label(t(STRINGS, "section_dep_heatmap"))

    _def_g8 = [cc for cc in ["USA","China","Germany","France","Japan",
                              "Canada","Brazil","Qatar","Nigeria","Egypt","Tunisia","India"]
               if cc in df_dep_top["country_name"].values]
    g8_countries = st.multiselect(
        t(STRINGS, "countries_label"),
        sorted(df_dep_top["country_name"].unique().tolist()),
        default=_def_g8, max_selections=20, key="g8_countries",
    )

    if g8_countries:
        df_g8 = df_dep_top[df_dep_top["country_name"].isin(g8_countries)].copy()

        piv_dep     = df_g8.pivot_table(
            index="country_name", columns="year",
            values="dependency_top").reindex(g8_countries)
        piv_partner = df_g8.pivot_table(
            index="country_name", columns="year",
            values="top_partner", aggfunc="first").reindex(g8_countries)
        piv_bdi     = df_g8.pivot_table(
            index="country_name", columns="year",
            values="bdi_top").reindex(g8_countries)
        piv_ncrit   = df_g8.pivot_table(
            index="country_name", columns="year",
            values="n_critical", aggfunc="first").reindex(g8_countries).fillna(0).astype(int)

        _g8_ann = []
        for i in range(len(piv_dep)):
            _row_ann = []
            for j in range(len(piv_dep.columns)):
                dep    = piv_dep.iloc[i, j]
                if pd.isna(dep):
                    _row_ann.append("")
                    continue
                partner = piv_partner.iloc[i, j]
                bdi     = piv_bdi.iloc[i, j]
                ncrit   = piv_ncrit.iloc[i, j]
                p_str   = (str(partner)[:9] + ".") if pd.notna(partner) and len(str(partner)) > 9 else (str(partner) if pd.notna(partner) else "")
                _row_ann.append(
                    f"{p_str}<br>{dep*100:.1f}% | BDI:{bdi:.2f}<br>crit:{int(ncrit)}"
                    if pd.notna(bdi) else
                    f"{p_str}<br>{dep*100:.1f}%<br>crit:{int(ncrit)}"
                )
            _g8_ann.append(_row_ann)

        fig_g8 = go.Figure(go.Heatmap(
            z=piv_dep.values,
            x=[str(int(y)) for y in piv_dep.columns],
            y=piv_dep.index.tolist(),
            colorscale=[[0,"#10B981"],[0.20,"#60A5FA"],
                        [0.50,"#F59E0B"],[1.00,"#EF4444"]],
            zmin=0, zmax=1,
            text=_g8_ann,
            texttemplate="%{text}",
            textfont=dict(size=8, color="#EEF4FB", family="IBM Plex Mono"),
            showscale=True,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> — %{x}<br>Dep top: %{z:.1%}<extra></extra>",
        ))
        if country in piv_dep.index.tolist():
            _idx_g8 = piv_dep.index.tolist().index(country)
            fig_g8.add_shape(type="rect",
                x0=-0.5, x1=len(piv_dep.columns)-0.5,
                y0=_idx_g8-0.5, y1=_idx_g8+0.5,
                line=dict(color="#EEF4FB", width=2))
        fig_g8.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title=t(STRINGS, "fig_g8_title"),
            xaxis=dict(type="category", gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"),
            margin=dict(l=130, r=20, t=50, b=40),
            height=max(400, len(g8_countries)*45 + 80),
        )
        st.plotly_chart(fig_g8, width='stretch')

        with st.expander(t(STRINGS, "expander_how_to_read_heatmap")):
            st.markdown(t(STRINGS, "how_to_read_heatmap_md"))


# ── TAB 5: SCORING & PROFILE ─────────────────────────────────
with tab5:

    # ── Palette profils (df_final / df_full_periode) ─────────────
    FP_COLORS = {
        "Robust":                  "#10B981",
        "Stable but Concentrated": "#A78BFA",
        "Vulnerable":              "#F59E0B",
        "Critical Dependency":     "#EF4444",
    }
    FP_NUM = {
        "Robust": 1, "Stable but Concentrated": 2,
        "Vulnerable": 3, "Critical Dependency": 4,
    }
    FP_SHORT = {
        "Robust": "Robust", "Stable but Concentrated": "Stable+C",
        "Vulnerable": "Vuln.", "Critical Dependency": "Critical",
    }
    # Colorscale catégorielle 4 niveaux pour heatmaps
    FP_CSCALE = [
        [0.00, "#10B981"], [0.24, "#10B981"],
        [0.25, "#A78BFA"], [0.49, "#A78BFA"],
        [0.50, "#F59E0B"], [0.74, "#F59E0B"],
        [0.75, "#EF4444"], [1.00, "#EF4444"],
    ]

    # ── Sélecteur pays de référence ───────────────────────────────
    _sc_countries = sorted(df_full_periode["country_name"].dropna().unique().tolist())
    _REF = ["USA", "China", "France", "Germany", "Brazil", "Canada", "Qatar", "India", "Nigeria"]
    _anim_def = [country] if country in _sc_countries else []
    _sc_def   = (
        ([country] if country in _sc_countries else [])
        + [cc for cc in _REF if cc in _sc_countries and cc != country]
    )

    # ── ① KPIs — composantes du score pour le pays cible (année sel_year) ──
    section_label(t(STRINGS, "section_score_components").format(country=country, year=sel_year))

    # Annual values from df_final; fallback to full-period if year missing
    _fy_row = df_final[(df_final["country_name"] == country) & (df_final["year"] == sel_year)]
    _fp_row = df_full_periode[df_full_periode["country_name"] == country]
    if not _fy_row.empty:
        _fy = _fy_row.iloc[0]
        _kpi_src   = f"{sel_year}"
        _kpi_prof  = _fy.get("profile_logic_global", "—")
        _kpi_score = float(_fy.get("score_global", 0) or 0)
        _kpi_cs    = float(_fy.get("critical_share", 0) or 0)
        _kpi_dep   = float(_fy.get("dependency_top", 0) or 0)
        _kpi_hhim  = float(_fy.get("hhi_import", 0) or 0)
        _kpi_hhix  = float(_fy.get("hhi_export", 0) or 0)
        _kpi_nc    = int(_fy.get("n_critical", 0) or 0)
        _kpi_bdis  = float(_fy.get("bdis", 0) or 0)
    elif not _fp_row.empty:
        _fy = _fp_row.iloc[0]
        _kpi_src   = "Full period"
        _kpi_prof  = _fy.get("profile_full_periode", "—")
        _kpi_score = float(_fy.get("score_global", 0) or 0)
        _kpi_cs    = float(_fy.get("critical_share", 0) or 0)
        _kpi_dep   = float(_fy.get("dependency_top", 0) or 0)
        _kpi_hhim  = float(_fy.get("hhi_import", 0) or 0)
        _kpi_hhix  = float(_fy.get("hhi_export", 0) or 0)
        _kpi_nc    = int(_fy.get("n_critical", 0) or 0)
        _kpi_bdis  = float(_fy.get("bdis", 0) or 0)
    else:
        _kpi_src = None

    if _kpi_src:
        _prof_clr = FP_COLORS.get(str(_kpi_prof), "#6B7280")
        _hhi_avg  = (_kpi_hhim + _kpi_hhix) / 2

        # ── Level classifiers (notebook thresholds) ───────────────────
        def _lv_hhi(v):
            if v < 0.08:  return "Low",       "#10B981"
            if v < 0.15:  return "Moderate",  "#A78BFA"
            if v < 0.25:  return "High",      "#F59E0B"
            return               "Extreme",    "#EF4444"

        def _lv_score(v):
            if v < 0.25:  return "Robust",    "#10B981"
            if v < 0.35:  return "Stable",    "#A78BFA"
            if v < 0.45:  return "Vulnerable","#F59E0B"
            return               "Critical",   "#EF4444"

        # ── Trend badge helpers ───────────────────────────────────────
        def _str_to_badge(s):
            if s == "Increasing":  return "Increasing", "#EF4444"
            if s == "Decreasing":  return "Decreasing", "#10B981"
            if s == "Stable":      return "Stable",     "#6B7280"
            return                        "",            ""

        def _delta_to_badge(delta, threshold=0.025):
            if delta is None:       return "", ""
            if delta > threshold:   return "Increasing", "#EF4444"
            if delta < -threshold:  return "Decreasing", "#10B981"
            return                         "Stable",     "#6B7280"

        # ── BDIS level from df_final string column ────────────────────
        def _bdis_lv_from_str(s):
            m = {
                "Interdependence":     ("Interdep.", "#10B981"),
                "Moderate Dependance": ("Moderate",  "#A78BFA"),
                "High Dependance":     ("High",      "#F59E0B"),
                "Critical Dependance": ("Critical",  "#EF4444"),
            }
            return m.get(str(s), ("—", "#6B7280"))

        # ── HHI: level from avg, trend badge from worst flow ──────────
        _hhi_m_trend = str(_fy.get("hhi_import_trend", "") or "")
        _hhi_x_trend = str(_fy.get("hhi_export_trend", "") or "")
        _rank        = {"Increasing": 2, "Stable": 1, "Decreasing": 0}
        _hhi_worst   = _hhi_m_trend if _rank.get(_hhi_m_trend, 1) >= _rank.get(_hhi_x_trend, 1) else _hhi_x_trend
        lv_hhi  = _lv_hhi(_hhi_avg)
        tr_hhi  = _str_to_badge(_hhi_worst)

        # ── BDIS: level from df_final, trend from prev year ──────────
        lv_bdis = _bdis_lv_from_str(_fy.get("bdis_level", ""))

        # ── Top Dep: level + trend from df_final ─────────────────────
        _dep_lv_str  = str(_fy.get("dependency_top_level", "") or "")
        _dep_clr_map = {"Low": "#10B981", "Moderate": "#A78BFA",
                        "High": "#F59E0B", "Critical": "#EF4444"}
        lv_dep  = (_dep_lv_str or "—", _dep_clr_map.get(_dep_lv_str, "#6B7280"))
        tr_dep  = _str_to_badge(str(_fy.get("dependency_top_trend", "") or ""))

        # ── Critical Share: border color (custom thresholds) ──────────
        def _cs_border(v):
            if v < 0.15:  return "#10B981"
            if v < 0.30:  return "#A78BFA"
            if v < 0.50:  return "#F59E0B"
            return               "#EF4444"

        # ── N Critical: border color only ─────────────────────────────
        def _nc_border(v):
            if v == 0:  return "#10B981"
            if v == 1:  return "#F59E0B"
            return             "#EF4444"

        # ── Prev year for BDIS + Score trends ─────────────────────────
        _prev_row = None
        if not _fy_row.empty:
            _pq = df_final[
                (df_final["country_name"] == country) &
                (df_final["year"] == sel_year - 1)
            ]
            if not _pq.empty:
                _prev_row = _pq.iloc[0]

        def _d(col):
            if _prev_row is None: return None
            try: return float(_fy.get(col, 0) or 0) - float(_prev_row.get(col, 0) or 0)
            except: return None

        tr_bdis = _delta_to_badge(_d("bdis"))
        tr_sc   = _delta_to_badge(_d("score_global"))
        lv_sc   = _lv_score(_kpi_score)

        # Profile score (annual only)
        _prof_score     = str(_fy.get("profile_score_global", "—")) if not _fy_row.empty else "—"
        _prof_score_clr = FP_COLORS.get(_prof_score, "#6B7280")

        # ── Card builder ──────────────────────────────────────────────
        def _pill(text, color):
            if not text: return ""
            return (
                f'<span style="font-size:10px;font-weight:600;color:{color};'
                f'background:{color}22;border:1px solid {color}55;'
                f'border-radius:99px;padding:1px 8px;white-space:nowrap;">{text}</span>'
            )

        def _card(label, value, color, lv_lbl="", lv_clr="", tr_lbl="", tr_clr="", sub=None):
            sub_html = (
                f'<div style="font-size:10px;color:#7090A8;margin-top:2px;'
                f'font-family:\'IBM Plex Mono\',monospace;">{sub}</div>'
            ) if sub else ""
            b1 = _pill(lv_lbl, lv_clr)
            b2 = _pill(tr_lbl, tr_clr)
            bottom_html = (
                f'<div style="display:flex;justify-content:center;align-items:center;'
                f'flex-wrap:wrap;gap:5px;margin-top:6px;">{b1}{b2}</div>'
            ) if (b1 or b2) else ""
            return (
                f'<div style="background:#1F3050;border:1px solid #304870;'
                f'border-top:3px solid {color};border-radius:10px;'
                f'padding:.65rem .8rem;text-align:center;'
                f'min-height:110px;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div>'
                f'<div style="font-size:10px;font-weight:600;color:#6890B0;'
                f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;">{label}</div>'
                f'<div style="font-size:20px;font-weight:700;color:{color};line-height:1.1;">{value}</div>'
                f'{sub_html}</div>'
                f'{bottom_html}</div>'
            )

        # ── KPI grid : all cards same size, profiles centered under inputs
        def _wrap(html, gc, gr=None):
            _gr = f"grid-row:{gr};" if gr else ""
            return f'<div style="grid-column:{gc};{_gr}">{html}</div>'

        _c_hhi  = _card(t(STRINGS, "card_hhi_avg"), f"{_hhi_avg:.3f}", lv_hhi[1],
                    lv_hhi[0], lv_hhi[1], tr_hhi[0], tr_hhi[1],
                    sub=f"M {_kpi_hhim:.3f} · X {_kpi_hhix:.3f}")
        _c_dep  = _card(t(STRINGS, "card_top_dependency"), f"{_kpi_dep*100:.1f}%", lv_dep[1],
                    lv_dep[0], lv_dep[1], tr_dep[0], tr_dep[1])
        _c_cs   = _card(t(STRINGS, "card_critical_share"), f"{_kpi_cs*100:.1f}%", _cs_border(_kpi_cs))
        _c_nc   = _card(t(STRINGS, "card_n_critical"), str(_kpi_nc), _nc_border(_kpi_nc))
        _c_bdis = _card("BDIS", f"{_kpi_bdis:.3f}", lv_bdis[1],
                    lv_bdis[0], lv_bdis[1], tr_bdis[0], tr_bdis[1])
        _c_sc   = _card(t(STRINGS, "card_score_global"), f"{_kpi_score:.3f}", lv_sc[1],
                    lv_sc[0], lv_sc[1], tr_sc[0], tr_sc[1])

        _div_badge = ""
        if _prof_score not in ("—", "") and _prof_score != str(_kpi_prof):
            _div_badge = (
                f'<div style="font-size:9px;font-weight:600;color:#F59E0B;'
                f'background:#F59E0B18;border:1px solid #F59E0B55;'
                f'border-radius:99px;padding:1px 8px;margin-top:6px;'
                f'display:inline-block;">{t(STRINGS, "badge_divergence")}</div>'
            )

        def _pf_card(label, val, clr, extra=""):
            return (
                f'<div style="background:#1F3050;border:1px solid #304870;'
                f'border-top:3px solid {clr};border-radius:10px;'
                f'padding:.65rem .8rem;text-align:center;'
                f'min-height:110px;display:flex;flex-direction:column;justify-content:center;">'
                f'<div style="font-size:10px;font-weight:600;color:#6890B0;'
                f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">{label}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{clr};line-height:1.35;">{val}</div>'
                f'{extra}</div>'
            )

        _c_pfl = _pf_card(t(STRINGS, "pf_profile_logic"), _kpi_prof, _prof_clr)
        _c_pfs = _pf_card(t(STRINGS, "pf_profile_score"), _prof_score, _prof_score_clr, _div_badge)

        # 12-track grid: each indicator card = 2 tracks (1/6 width)
        # Profile Logic  centered under HHI(1-2)+TopDep(3-4)  → tracks 2/4
        # Profile Score  centered under CritSh..Score(5-12)   → tracks 8/10
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(12,1fr);gap:8px;">'
            + _wrap(_c_hhi,  "1/3")
            + _wrap(_c_dep,  "3/5")
            + _wrap(_c_cs,   "5/7")
            + _wrap(_c_nc,   "7/9")
            + _wrap(_c_bdis, "9/11")
            + _wrap(_c_sc,   "11/13")
            + _wrap(_c_pfl,  "2/4",  gr=2)
            + _wrap(_c_pfs,  "8/10", gr=2)
            + '</div>',
            unsafe_allow_html=True,
        )

        with st.expander(t(STRINGS, "expander_definitions_formulas")):
            st.markdown(t(STRINGS, "kpi_definitions_md"), unsafe_allow_html=True)

    thin_divider()

    # ── ② G.9.1 · Annual Risk Space — Animation ──────────────────
    section_label(t(STRINGS, "section_risk_space"))

    _anim_countries = st.multiselect(
        t(STRINGS, "countries_to_display_label"),
        _sc_countries, default=_anim_def, max_selections=25,
        key=f"sc_anim_countries_{country}",
    )

    if _anim_countries:
        import plotly.express as px
        _df_anim = df_final[df_final["country_name"].isin(_anim_countries)].copy()
        _df_anim = _df_anim.dropna(subset=["score_import","score_export","profile_logic_global"])
        _df_anim["year_str"]    = _df_anim["year"].astype(str)
        _df_anim["total_B"]     = (_df_anim["total_trade"].fillna(1e9) / 1e9).clip(0.5, None)
        _df_anim["profile_lbl"] = _df_anim["profile_logic_global"]

        # Dummy rows — one per profile per year at (-0.15, -0.15) to lock legend
        _all_yrs_a = sorted(_df_anim["year"].unique())
        _dummies = pd.DataFrame([{
            "country_name": "", "year": yr, "year_str": str(yr),
            "score_import": -0.15, "score_export": -0.15,
            "profile_lbl": prof, "total_B": 0.01,
        } for yr in _all_yrs_a for prof in FP_COLORS.keys()])
        _df_plot = pd.concat([_df_anim, _dummies], ignore_index=True)
        _df_plot  = _df_plot.sort_values(["year","country_name"])

        fig_anim = px.scatter(
            _df_plot,
            x="score_import", y="score_export",
            animation_frame="year_str",
            color="profile_lbl",
            color_discrete_map=FP_COLORS,
            category_orders={"profile_lbl": list(FP_COLORS.keys())},
            size="total_B", size_max=42,
            text="country_name",
            hover_name="country_name",
            hover_data={
                "score_import":  ":.3f",
                "score_export":  ":.3f",
                "profile_lbl":   True,
                "total_B":       ":.1f",
                "year_str":      False,
                "country_name":  False,
            },
            labels={
                "score_import": t(STRINGS, "lbl_import_risk_score"),
                "score_export": t(STRINGS, "lbl_export_risk_score"),
                "profile_lbl":  t(STRINGS, "lbl_profile"),
                "total_B":      t(STRINGS, "lbl_total_trade_b"),
            },
            title=t(STRINGS, "fig_anim_title"),
        )

        # Static background: trajectory lines + start ▲ + end ● for each country
        for _ctry in _anim_countries:
            _dc = _df_anim[_df_anim["country_name"] == _ctry].sort_values("year")
            if _dc.empty:
                continue
            # Trajectory line with year dots
            fig_anim.add_trace(go.Scatter(
                x=_dc["score_import"].tolist(),
                y=_dc["score_export"].tolist(),
                mode="lines+markers",
                line=dict(width=1.8, dash="dot", color="rgba(160,185,225,0.55)"),
                marker=dict(size=4, color="rgba(160,185,225,0.55)"),
                showlegend=False, hoverinfo="skip",
            ))
            # Start marker ▲ only (no label)
            _first = _dc.iloc[0]
            fig_anim.add_trace(go.Scatter(
                x=[_first["score_import"]], y=[_first["score_export"]],
                mode="markers",
                marker=dict(symbol="triangle-up", size=11,
                            color="rgba(200,215,245,0.70)",
                            line=dict(color="rgba(220,235,255,0.85)", width=1)),
                showlegend=False, hoverinfo="skip",
            ))
            # End marker ● filled only (no label)
            _last = _dc.iloc[-1]
            fig_anim.add_trace(go.Scatter(
                x=[_last["score_import"]], y=[_last["score_export"]],
                mode="markers",
                marker=dict(symbol="circle", size=13,
                            color="rgba(220,235,255,0.80)",
                            line=dict(color="#EEF4FB", width=1.5)),
                showlegend=False, hoverinfo="skip",
            ))

        # Quadrant threshold lines
        for _v, _lbl in [(0.30,"Stable→Vulnerable"), (0.50,"Vulnerable→Critical")]:
            fig_anim.add_vline(x=_v, line_dash="dot", line_color="#4A6090",
                               line_width=1, opacity=0.55)
            fig_anim.add_hline(y=_v, line_dash="dot", line_color="#4A6090",
                               line_width=1, opacity=0.55)

        # Dynamic zoom — axis range based on selected countries' actual values
        _pad = 0.06
        _x_min = max(0, _df_anim["score_import"].min() - _pad)
        _x_max = min(1, _df_anim["score_import"].max() + _pad)
        _y_min = max(0, _df_anim["score_export"].min() - _pad)
        _y_max = min(1, _df_anim["score_export"].max() + _pad)

        fig_anim.update_layout(
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
            title_font=dict(color="#EEF4FB", size=14),
            xaxis=dict(range=[_x_min, _x_max], gridcolor="#304870",
                       title=t(STRINGS, "axis_import_risk")),
            yaxis=dict(range=[_y_min, _y_max], gridcolor="#304870",
                       title=t(STRINGS, "axis_export_risk")),
            legend=dict(bgcolor="#1A2840", bordercolor="#304870", title=t(STRINGS, "lbl_profile")),
            margin=dict(l=70, r=50, t=70, b=70),
            height=560,
        )
        # 1 200 ms per frame, 400 ms transition
        fig_anim.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Sans"),
        )

        # Per-frame score annotation for selected country only
        for _frame in fig_anim.frames:
            try:
                _yr = int(_frame.name)
            except (ValueError, TypeError):
                continue
            _rc = _df_anim[
                (_df_anim["country_name"] == country) & (_df_anim["year"] == _yr)
            ]
            if _rc.empty:
                continue
            _r      = _rc.iloc[0]
            _fc     = FP_COLORS.get(str(_r.get("profile_logic_global", "")), "#6B7280")
            _sc_val = float(_r.get("score_global", 0) or 0)
            _frame.layout = {
                "annotations": [{
                    "x": float(_r["score_import"]),
                    "y": float(_r["score_export"]),
                    "xref": "x", "yref": "y",
                    "text": t(STRINGS, "anim_frame_annot").format(country=country, year=_yr, score=f"{_sc_val:.3f}"),
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowwidth": 1.5,
                    "arrowcolor": _fc,
                    "ax": 55, "ay": -50,
                    "xanchor": "left",
                    "font": {"size": 11, "color": "#EEF4FB",
                             "family": "IBM Plex Mono"},
                    "bgcolor": "#1A2840",
                    "bordercolor": _fc,
                    "borderwidth": 1.5,
                    "borderpad": 6,
                    "align": "left",
                }]
            }

        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
        st.plotly_chart(fig_anim, width='stretch')
        st.caption(t(STRINGS, "caption_anim"))

    thin_divider()

    # ── ③ G.9.2 · Profile Heatmap — Country × Year ───────────────
    section_label(t(STRINGS, "section_profile_heatmap"))

    _hm_countries = st.multiselect(
        t(STRINGS, "countries_label"),
        _sc_countries, default=_sc_def, max_selections=25,
        key="sc_hm_countries",
    )

    if _hm_countries:
        _df_hm = df_final[df_final["country_name"].isin(_hm_countries)].copy()
        _df_hm["profile_num"] = _df_hm["profile_logic_global"].map(FP_NUM).fillna(0)
        _piv_num  = _df_hm.pivot_table(
            index="country_name", columns="year",
            values="profile_num", aggfunc="first").reindex(_hm_countries)
        _piv_lbl  = _df_hm.pivot_table(
            index="country_name", columns="year",
            values="profile_logic_global", aggfunc="first").reindex(_hm_countries)

        _hm_text = _piv_lbl.map(lambda v: FP_SHORT.get(str(v), "") if pd.notna(v) else "")

        fig_phm = go.Figure(go.Heatmap(
            z=_piv_num.values,
            x=[str(int(y)) for y in _piv_num.columns],
            y=_piv_num.index.tolist(),
            colorscale=FP_CSCALE,
            zmin=0.5, zmax=4.5,
            text=_hm_text.values,
            texttemplate="%{text}",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Mono"),
            showscale=False,
            hoverongaps=False,
            customdata=_piv_lbl.values,
            hovertemplate="<b>%{y}</b> — %{x}<br>Profile: %{customdata}<extra></extra>",
        ))
        # Highlight selected country row
        if country in _piv_num.index.tolist():
            _idx_c = _piv_num.index.tolist().index(country)
            fig_phm.add_shape(type="rect",
                x0=-0.5, x1=len(_piv_num.columns)-0.5,
                y0=_idx_c-0.5, y1=_idx_c+0.5,
                line=dict(color="#EEF4FB", width=2))
        # Legend as annotations (no colorbar)
        for i, (lbl, clr) in enumerate(FP_COLORS.items()):
            fig_phm.add_annotation(
                xref="paper", yref="paper",
                x=1.01 + i*0, y=1.0 - i*0.06,
                text=f"<b style='color:{clr}'>■</b> {lbl}",
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(size=10, color=clr, family="IBM Plex Sans"),
            )
        fig_phm.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            title=t(STRINGS, "fig_phm_title"),
            xaxis=dict(type="category", gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"),
            margin=dict(l=140, r=200, t=50, b=40),
            height=max(380, len(_hm_countries)*36 + 80),
        )
        st.plotly_chart(fig_phm, width='stretch')

    thin_divider()

    # ── ④ G.9.3 · Logic vs Score — Divergence Analysis ───────────
    section_label(t(STRINGS, "section_divergence").format(country=country))

    _df_div = df_final[df_final["country_name"] == country].copy().sort_values("year")
    if not _df_div.empty:
        _yrs_div  = _df_div["year"].astype(str).tolist()
        _log_num  = _df_div["profile_logic_global"].map(FP_NUM).fillna(0).tolist()
        _sc_num   = _df_div["profile_score_global"].map(FP_NUM).fillna(0).tolist()
        _log_lbl  = _df_div["profile_logic_global"].tolist()
        _sc_lbl   = _df_div["profile_score_global"].tolist()
        _conv     = _df_div["convergence"].tolist()
        _diag     = _df_div.get("diagnostic_note", _df_div["convergence_type"]).tolist()

        _bar_w    = 0.35

        fig_div = go.Figure()
        # Logic profile bars
        fig_div.add_trace(go.Bar(
            name=t(STRINGS, "trace_logic_profile"),
            x=_yrs_div, y=_log_num,
            offsetgroup=0,
            marker_color=[FP_COLORS.get(str(p), "#6B7280") for p in _log_lbl],
            text=[FP_SHORT.get(str(p),"") for p in _log_lbl],
            textposition="inside",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Mono"),
            customdata=list(zip(_log_lbl, _conv, _diag)),
            hovertemplate="<b>" + t(STRINGS, "hover_logic_prefix") + "</b> %{x}<br>" + t(STRINGS, "hover_profile") + " %{customdata[0]}<br>"
                          + t(STRINGS, "hover_convergent") + " %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
        # Score profile bars
        fig_div.add_trace(go.Bar(
            name=t(STRINGS, "trace_score_profile"),
            x=_yrs_div, y=_sc_num,
            offsetgroup=1,
            marker_color=[f"rgba({int(FP_COLORS.get(str(p),'#6B7280')[1:3],16)},"
                          f"{int(FP_COLORS.get(str(p),'#6B7280')[3:5],16)},"
                          f"{int(FP_COLORS.get(str(p),'#6B7280')[5:7],16)},0.50)"
                          for p in _sc_lbl],
            text=[FP_SHORT.get(str(p),"") for p in _sc_lbl],
            textposition="inside",
            textfont=dict(size=9, color="#EEF4FB", family="IBM Plex Mono"),
            customdata=list(zip(_sc_lbl, _conv)),
            hovertemplate="<b>Score</b> %{x}<br>" + t(STRINGS, "hover_profile") + " %{customdata[0]}<br>"
                          + t(STRINGS, "hover_convergent") + " %{customdata[1]}<extra></extra>",
        ))
        # Divergence markers (red asterisk at top of divergent years)
        _div_yrs = [_yrs_div[i] for i, c in enumerate(_conv)
                    if str(c).lower() not in ("true","1","yes")]
        _div_vals = [max(_log_num[i], _sc_num[i]) + 0.15
                     for i, c in enumerate(_conv)
                     if str(c).lower() not in ("true","1","yes")]
        if _div_yrs:
            fig_div.add_trace(go.Scatter(
                x=_div_yrs, y=_div_vals,
                mode="markers+text",
                marker=dict(symbol="x", size=12, color="#EF4444"),
                text=["divergence"] * len(_div_yrs),
                textposition="top center",
                textfont=dict(size=8, color="#EF4444"),
                name="Divergence",
                hoverinfo="skip",
            ))
        fig_div.update_layout(**PLC,
            paper_bgcolor="#1F3050", plot_bgcolor="#162035",
            barmode="group",
            title=t(STRINGS, "fig_div_title").format(country=country),
            xaxis=dict(type="category", gridcolor="#304870"),
            yaxis=dict(
                gridcolor="#304870",
                tickvals=[1, 2, 3, 4],
                ticktext=["Robust", "Stable+C", "Vulnerable", "Critical"],
                range=[0, 4.8],
            ),
            legend=dict(bgcolor="#1A2840", bordercolor="#304870"),
            margin=dict(l=100, r=40, t=50, b=40),
            height=380,
        )
        st.plotly_chart(fig_div, width='stretch')

        # Divergence summary line
        _n_div = sum(1 for c in _conv if str(c).lower() not in ("true","1","yes"))
        _div_pct = _n_div / len(_conv) * 100 if _conv else 0
        _div_color = "#EF4444" if _div_pct > 30 else "#F59E0B" if _div_pct > 10 else "#10B981"
        _div_status = t(STRINGS, "status_frequent") if _div_pct > 30 else t(STRINGS, "status_convergent")
        st.markdown(
            f'<div style="font-size:12px;color:{_div_color};'
            f'font-family:\'IBM Plex Mono\',monospace;padding:4px 0 8px 2px;">'
            f'{t(STRINGS, "divergence_summary").format(n=_n_div, total=len(_conv), pct=_div_pct, status=_div_status)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(t(STRINGS, "no_annual_data").format(country=country))

    thin_divider()

    # ── ⑤ G.10 · Full Period — Drift & Stability ─────────────────
    section_label(t(STRINGS, "section_drift_stability"))

    _fp_countries = st.multiselect(
        t(STRINGS, "countries_label"),
        _sc_countries, default=_anim_def, max_selections=25,
        key=f"sc_fp_countries_{country}",
    )

    if _fp_countries:
        _df_fp = df_full_periode[df_full_periode["country_name"].isin(_fp_countries)].copy()
        _df_fp = _df_fp.sort_values("score_full_periode", ascending=False)

        # Dominant logic profile — computed on the fly from df_final
        _dom_logic = (
            df_final[df_final["country_name"].isin(_fp_countries)]
            .groupby("country_name")["profile_logic_global"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
            .reset_index()
            .rename(columns={"profile_logic_global": "dominant_logic"})
        )
        _df_fp = _df_fp.merge(_dom_logic, on="country_name", how="left")

        # ── Column header labels — built once via t() so every downstream
        # lookup (format map, numeric-label set, column widths) stays in
        # sync with the currently selected language.
        _L_COUNTRY       = t(STRINGS, "col_country")
        _L_PROFILE_FP    = t(STRINGS, "col_profile_full_period")
        _L_DOM_SCORE     = t(STRINGS, "col_dominant_annual_score")
        _L_DOM_LOGIC     = t(STRINGS, "col_dominant_annual_logic")
        _L_DIV_FULL_ANN  = t(STRINGS, "col_divergence_full_vs_annual")
        _L_DIV_CATEGORY  = t(STRINGS, "col_divergence_category")
        _L_CONV_SC_LOGIC = t(STRINGS, "col_convergence_score_logic")
        _L_PROFILE_STAB  = t(STRINGS, "col_profile_stability")
        _L_STAB_SCORE    = t(STRINGS, "col_stability_score")
        _L_PROFILE_DRIFT = t(STRINGS, "col_profile_drift")
        _L_DRIFT_MAG     = t(STRINGS, "col_drift_magnitude")
        _L_HHI_MEDIAN    = t(STRINGS, "col_hhi_median")
        _L_TOP_PARTNER_DEP = t(STRINGS, "col_top_partner_dependency")
        _L_CRIT_PARTNERS_SH = t(STRINGS, "col_critical_partners_share")
        _L_N_CRIT_PARTNERS = t(STRINGS, "col_n_critical_partners")
        _L_BDIS_MEDIAN   = t(STRINGS, "col_bdis_median")
        _L_SCORE_FP      = t(STRINGS, "col_score_full_period")

        # ── Master column order (source, label, group) ────────────────
        # group=None → always visible  "d" → Drift toggle  "i" → Indicators toggle
        _ALL_COLS = [
            ("country_name",         _L_COUNTRY,          None),
            ("profile_full_periode", _L_PROFILE_FP,       None),
            ("dominant_profile",     _L_DOM_SCORE,        None),
            ("dominant_logic",       _L_DOM_LOGIC,        None),
            ("divergence_freq",      _L_DIV_FULL_ANN,     None),
            ("divergence_category",  _L_DIV_CATEGORY,     "d"),
            ("convergence_rate",     _L_CONV_SC_LOGIC,     "d"),
            ("stability_label",      _L_PROFILE_STAB,     None),
            ("stability_score",      _L_STAB_SCORE,       "d"),
            ("drift",                _L_PROFILE_DRIFT,    None),
            ("drift_magnitude",      _L_DRIFT_MAG,        "d"),
            ("hhi_avg",              _L_HHI_MEDIAN,       "i"),
            ("dependency_top",       _L_TOP_PARTNER_DEP,  "i"),
            ("critical_share",       _L_CRIT_PARTNERS_SH, "i"),
            ("n_critical",           _L_N_CRIT_PARTNERS,  "i"),
            ("bdis",                 _L_BDIS_MEDIAN,      "i"),
            ("score_full_periode",   _L_SCORE_FP,         "i"),
        ]
        _FMT = {
            _L_DIV_FULL_ANN:     "{:.1%}",
            _L_CONV_SC_LOGIC:    "{:.1%}",
            _L_STAB_SCORE:       "{:.2f}",
            _L_DRIFT_MAG:        "{:.0f}",
            _L_HHI_MEDIAN:       "{:.3f}",
            _L_TOP_PARTNER_DEP:  "{:.1%}",
            _L_CRIT_PARTNERS_SH: "{:.1%}",
            _L_N_CRIT_PARTNERS:  "{:.0f}",
            _L_BDIS_MEDIAN:      "{:.3f}",
            _L_SCORE_FP:         "{:.3f}",
        }

        # ── Column group selector ──────────────────────────────────────
        _OPT_DRIFT = t(STRINGS, "opt_drift_stability_conv")
        _OPT_INDIC = t(STRINGS, "opt_indicators")

        _selected = st.pills(
            t(STRINGS, "pills_add_columns_label"),
            options=[_OPT_DRIFT, _OPT_INDIC],
            selection_mode="multi",
            default=[],
            key="sc_fp_pills",
        )

        # ── Build visible columns in master order ─────────────────────
        _show_d = _OPT_DRIFT in (_selected or [])
        _show_i = _OPT_INDIC in (_selected or [])

        _sel = [
            (src, lbl) for src, lbl, grp in _ALL_COLS
            if (grp is None or (grp == "d" and _show_d) or (grp == "i" and _show_i))
            and src in _df_fp.columns
        ]
        _df_show = _df_fp[[s for s, _ in _sel]].copy()
        _df_show.columns = [l for _, l in _sel]

        # Force numeric dtype on all numeric columns to avoid format errors
        _NUMERIC_LABELS = {
            _L_DIV_FULL_ANN, _L_CONV_SC_LOGIC, _L_STAB_SCORE,
            _L_DRIFT_MAG, _L_HHI_MEDIAN, _L_TOP_PARTNER_DEP,
            _L_CRIT_PARTNERS_SH, _L_N_CRIT_PARTNERS, _L_BDIS_MEDIAN, _L_SCORE_FP,
        }
        for _nc in _NUMERIC_LABELS & set(_df_show.columns):
            _df_show[_nc] = pd.to_numeric(_df_show[_nc], errors="coerce")

        # ── Styling helpers ────────────────────────────────────────────
        def _color_profile(val):
            return f"color: {FP_COLORS.get(str(val), '#C4D5E8')}"

        def _color_drift(val):
            if str(val) == "Worsening":  return "color: #EF4444"
            if str(val) == "Improving":  return "color: #10B981"
            return "color: #6890B0"

        def _color_score(val):
            try:
                v = float(val)
                if v >= 0.45: return "color: #EF4444"
                if v >= 0.35: return "color: #F59E0B"
                if v >= 0.25: return "color: #A78BFA"
                return "color: #10B981"
            except Exception:
                return ""

        def _color_stability(val):
            s = str(val)
            if s == "Very Stable": return "color: #10B981"
            if s == "Stable":      return "color: #A78BFA"
            if s == "Moderate":    return "color: #F59E0B"
            if s == "Unstable":    return "color: #EF4444"
            return "color: #6890B0"

        _style = _df_show.style
        for _col in [_L_PROFILE_FP, _L_DOM_SCORE, _L_DOM_LOGIC]:
            if _col in _df_show.columns:
                _style = _style.map(_color_profile, subset=[_col])
        if _L_PROFILE_STAB in _df_show.columns:
            _style = _style.map(_color_stability, subset=[_L_PROFILE_STAB])
        if _L_PROFILE_DRIFT in _df_show.columns:
            _style = _style.map(_color_drift, subset=[_L_PROFILE_DRIFT])
        if _L_SCORE_FP in _df_show.columns:
            _style = _style.map(_color_score, subset=[_L_SCORE_FP])
        _active_fmt = {k: v for k, v in _FMT.items() if k in _df_show.columns}
        if _active_fmt:
            _style = _style.format(_active_fmt)

        # Compact column widths — headers truncate, data fits without horizontal scroll
        _COL_W = {
            _L_COUNTRY:          90,
            _L_PROFILE_FP:       125,
            _L_DOM_SCORE:        115,
            _L_DOM_LOGIC:        115,
            _L_DIV_FULL_ANN:     80,
            _L_DIV_CATEGORY:     90,
            _L_CONV_SC_LOGIC:    75,
            _L_PROFILE_STAB:     90,
            _L_STAB_SCORE:       75,
            _L_PROFILE_DRIFT:    85,
            _L_DRIFT_MAG:        70,
            _L_HHI_MEDIAN:       70,
            _L_TOP_PARTNER_DEP:  75,
            _L_CRIT_PARTNERS_SH: 75,
            _L_N_CRIT_PARTNERS:  70,
            _L_BDIS_MEDIAN:      70,
            _L_SCORE_FP:         75,
        }
        _col_cfg = {
            col: st.column_config.Column(width=_COL_W[col])
            for col in _df_show.columns if col in _COL_W
        }

        st.dataframe(
            _style,
            width='stretch',
            column_config=_col_cfg,
            height=min(600, len(_df_show) * 38 + 60),
        )

        with st.expander(t(STRINGS, "expander_column_definitions")):
            st.markdown(t(STRINGS, "fp_definitions_md"), unsafe_allow_html=True)