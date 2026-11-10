"""
Trade Analysis — T3 · Country Trade Profile
Covers: F.3 — Country Trade Structure
Interactive: selectbox country → all charts regenerate
- F.3.2 Export/Import trends
- F.3.3 Share trends
- F.3.4 Trade balance & coverage
- F.3.5 Growth rates
- F.3.6 Global overview (4 subplots)
- F.3.7 Structural scoring (radar + scorecard + heatmap)
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, format_value
from utils.i18n import t

inject_styles()

# ════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ════════════════════════════════════════════════════════════════
# NOTE: category/profile values that come from df_structural columns
# (e.g. "Dominant", "Star", "Stars", "Leader", "Global Leader", ...)
# are DATA and are never translated here, even where they are shown
# as plain text (badges, distribution charts, scorecards). Only the
# surrounding UI copy (titles, axis labels, descriptive sub-text) is
# translated.

STRINGS = {
    "en": {
        "eyebrow": "Trade Analysis · Page 3 of 6",
        "title": "Country Trade Profile",
        "caption": "F.3 — Individual country trade structure · trends · balance · scoring",
        "lbl_select_country": "Select country",
        "kpi_exports_latest": "Exports (latest)",
        "kpi_export_share": "Export share",
        "kpi_structural_score": "Structural score",
        "kpi_profile": "Profile",
        "warn_no_data": "No data available for {country}.",
        "warn_no_structural": "No structural data for {country}.",

        "tab_trends": "📈 Trends",
        "tab_balance": "⚖️ Balance",
        "tab_growth": "📊 Growth",
        "tab_global_view": "🌐 Global View",
        "tab_scoring": "🏅 Scoring",
        "tab_positioning": "🎯 Positioning",

        "sec_trends": "F.3.2 · Export & Import Trends",
        "sec_share_trends": "F.3.3 · Share Trends",
        "opt_exports": "Exports",
        "opt_imports": "Imports",
        "opt_export_share": "Export Share",
        "opt_import_share": "Import Share",
        "unit_usd_billion": "USD Billion",
        "unit_share": "Share",
        "title_trends": "{country} — Exports & Imports (USD Billion)",
        "title_share_trends": "{country} — Export & Import Share of World Trade",

        "sec_balance": "F.3.4 · Trade Balance & Coverage Ratio",
        "trace_trade_balance": "Trade Balance",
        "title_balance": "{country} — Trade Balance (USD Billion)",
        "trace_coverage_ratio": "Coverage Ratio (%)",
        "balanced_100": "Balanced (100%)",
        "title_coverage": "{country} — Coverage Ratio (Exports/Imports × 100%)",
        "unit_coverage_pct": "Coverage (%)",

        "sec_growth": "F.3.5 · Annual Growth Rates",
        "title_growth": "{country} — Annual Export & Import Growth Rate (%)",
        "unit_pct": "%",

        "sec_global_view": "F.3.6 · Global Trade View — Overview",
        "title_volume": "Volume (USD B)",
        "unit_usd_b": "USD B",
        "title_world_share": "World Share (%)",
        "title_export_growth": "Export Growth (%)",
        "trace_export_pct": "Export %",
        "title_trade_balance_b": "Trade Balance (USD B)",
        "trace_balance": "Balance",
        "card_stability": "📊 Stability — Export Volatility",
        "card_stability_sub": "σ of annual export growth · lower = more stable",
        "card_consistency": "📉 Consistency — Negative Growth Freq.",
        "card_consistency_sub": "{years} with negative export growth",
        "years_of_10": "{n:.1f} / 10 years",

        "sec_positioning": "F.4.5 · Global Positioning — Where does this country stand?",
        "quad_tab_static": "📐 Static Quadrant",
        "quad_tab_dynamic": "⚡ Dynamic Quadrant",
        "quad_tab_strategic": "⭐ Strategic Quadrant",
        "quad_tab_meta": "🌐 Meta Profile",
        "other_countries": "Other countries",
        "others": "Others",

        "lbl_countries_to_animate": "Countries to animate",
        "title_static_anim": "Static Quadrant — Animated ({start}–{end})",
        "axis_export_share_size": "Export Share (%) — Trade Size",
        "axis_coverage_balance": "Coverage Ratio (%) — Trade Balance",
        "quad_dominant_desc": "🟢 Dominant<br><sub>Large & Surplus</sub>",
        "quad_balanced_desc": "🔵 Balanced<br><sub>Small & Surplus</sub>",
        "quad_vulnerable_desc": "🟡 Vulnerable<br><sub>Large & Deficit</sub>",
        "quad_marginal_desc": "⚫ Marginal<br><sub>Small & Deficit</sub>",
        "btn_play": "▶ Play",
        "btn_pause": "⏸ Pause",
        "slider_year_prefix": "Year: ",
        "badge_static_quadrant": "→ Static Quadrant: ",
        "title_static_dist": "Static Quadrant Distribution — All Countries",
        "axis_countries": "Countries",

        "lbl_highlight_countries": "Highlight countries",
        "hover_cagr": "CAGR",
        "hover_volatility": "Volatility",
        "hover_quadrant": "Quadrant",
        "hover_dynamic_quadrant": "Dynamic Quadrant",
        "quad_star_desc": "⭐ Star<br>(High Growth · Low Vol.)",
        "quad_stable_desc": "😴 Stable Stagnant<br>(Low Growth · Low Vol.)",
        "quad_risky_desc": "📈 Risky Grower<br>(High Growth · High Vol.)",
        "quad_declining_desc": "📉 Declining<br>(Low Growth · High Vol.)",
        "title_dynamic_quadrant": "Dynamic Quadrant — {country} vs all countries",
        "axis_cagr_export": "CAGR Export (%)",
        "axis_export_volatility": "Export Volatility",
        "badge_dynamic_quadrant": "→ Dynamic Quadrant: ",

        "hover_export_share": "Export Share",
        "hover_strategic_quadrant": "Strategic Quadrant",
        "quad_stars_desc": "⭐ Stars<br><sub>Large & Growing</sub>",
        "quad_emerging_desc": "🌱 Emerging<br><sub>Small & Growing</sub>",
        "quad_cashcows_desc": "🐄 Cash Cows<br><sub>Large & Stagnant</sub>",
        "quad_marginal2_desc": "📉 Marginal<br><sub>Small & Shrinking</sub>",
        "title_strategic_quadrant": "Strategic Quadrant (BCG) — {country}",
        "axis_export_share_pct": "Export Share (%)",
        "badge_strategic_quadrant": "→ Strategic Quadrant: ",

        "title_meta_positioning": "Meta Profile Positioning — {country} highlighted",
        "axis_structural_score": "Structural Score",
        "hover_structural_score": "Structural Score",
        "hover_meta_profile": "Meta Profile",
        "title_meta_dist": "Meta Profile Distribution",
        "badge_meta_profile": "{country} Meta Profile: ",

        "expander_quad_defs": "Quadrant & Meta Profile definitions",
        "quad_defs_col1": (
            "**Static** (Export Share × Coverage Ratio)\n"
            "- 🟢 **Dominant** — large exporter + trade surplus\n"
            "- 🔵 **Balanced** — small exporter + trade surplus\n"
            "- 🟡 **Vulnerable** — large exporter + trade deficit\n"
            "- ⚫ **Marginal** — small exporter + trade deficit\n\n"
            "**Dynamic** (CAGR × Volatility, thresholds: z=0 × median vol)\n"
            "- ⭐ **Star** — high CAGR, low volatility\n"
            "- 📈 **Risky Grower** — high CAGR, high volatility\n"
            "- 📉 **Declining** — low CAGR, low volatility\n"
            "- 😴 **Stable Stagnant** — low CAGR, high volatility\n\n"
            "**Strategic** (Export Share × CAGR, thresholds: 1% × 0)\n"
            "- ⭐ **Stars** — large share & positive CAGR\n"
            "- 🌱 **Emerging** — small share & positive CAGR\n"
            "- 🐄 **Cash Cows** — large share & negative CAGR\n"
            "- 📉 **Marginal** — small share & negative CAGR"
        ),
        "quad_defs_col2": (
            "**Meta Profile** (composite of 3 quadrants)\n\n"
            "Derived from Static × Dynamic × Strategic combination:\n\n"
            "- 🌍 **Global Leader** — Dominant + Stars\n"
            "- 💪 **Established Power** — Dominant/Vulnerable + Stars/Cash Cow\n"
            "- ⬆️ **Rising Challenger** — Emerging + Star (small but stable & growing)\n"
            "- 🌪️ **Volatile Emerging** — Emerging + Risky Grower (small + unstable)\n"
            "- 📍 **Peripheral** — otherwise (low score, low growth)\n\n"
            "*Not a 4-quadrant system — composite classification\n"
            "from the interaction of 3 quadrant dimensions*"
        ),

        "sec_scoring": "F.3.7 · Structural Scoring",
        "md_profile_scorecard": "**Profile Scorecard**",
        "score_size_profile": "Size Profile",
        "score_balance_profile": "Balance Profile",
        "score_growth_profile": "Growth Profile",
        "score_stability_profile": "Stability Profile",
        "score_consistency_profile": "Consistency Profile",
        "lbl_structural_score": "Structural Score",
        "lbl_structural_profile": "Structural Profile",
        "lbl_meta_profile": "Meta Profile",
        "lbl_quadrants": "Quadrants",
        "lbl_static": "Static",
        "lbl_dynamic": "Dynamic",
        "lbl_strategic": "Strategic",
        "sec_radar": "Radar — 5 Dimensions",
        "lbl_compare_with": "Compare with",
        "dim_size": "Size",
        "dim_balance": "Balance",
        "dim_growth": "Growth",
        "dim_stability": "Stability",
        "dim_consistency": "Consistency",
        "sec_structural_heatmap": "Structural Heatmap — Top & Bottom",
        "lbl_view": "View",
        "opt_top_performers": "Top performers",
        "opt_bottom_performers": "Bottom performers",
        "lbl_n_countries": "N countries",
        "expander_scoring_methodology": "Scoring methodology",
        "md_scoring_methodology": (
            "**Structural Score** = `0.60 × z_export_share + 0.10 × z_coverage_ratio "
            "+ 0.15 × z_cagr_export − 0.10 × z_volatility_export − 0.05 × z_negfreq_export`\n\n"
            "**Thresholds** : q85 → Strong · q50 → Good · q15 → Moderate · below → Weak\n\n"
            "**Radar dimensions** : all mapped 0→1 where 1 = best profile\n"
            "(e.g. Leader=1.0, Major=0.67, Mid=0.33, Small=0.0)"
        ),
    },
    "fr": {
        "eyebrow": "Analyse commerciale · Page 3 sur 6",
        "title": "Profil pays",
        "caption": "F.3 — Structure commerciale du pays · tendances · balance · scoring",
        "lbl_select_country": "Sélectionner un pays",
        "kpi_exports_latest": "Exportations (dernière année)",
        "kpi_export_share": "Part des exportations",
        "kpi_structural_score": "Score structurel",
        "kpi_profile": "Profil",
        "warn_no_data": "Aucune donnée disponible pour {country}.",
        "warn_no_structural": "Aucune donnée structurelle pour {country}.",

        "tab_trends": "📈 Tendances",
        "tab_balance": "⚖️ Balance",
        "tab_growth": "📊 Croissance",
        "tab_global_view": "🌐 Vue globale",
        "tab_scoring": "🏅 Scoring",
        "tab_positioning": "🎯 Positionnement",

        "sec_trends": "F.3.2 · Tendances des exportations & importations",
        "sec_share_trends": "F.3.3 · Tendances des parts de marché",
        "opt_exports": "Exportations",
        "opt_imports": "Importations",
        "opt_export_share": "Part des exportations",
        "opt_import_share": "Part des importations",
        "unit_usd_billion": "Milliards USD",
        "unit_share": "Part",
        "title_trends": "{country} — Exportations & importations (milliards USD)",
        "title_share_trends": "{country} — Part des exportations & importations dans le commerce mondial",

        "sec_balance": "F.3.4 · Balance commerciale & taux de couverture",
        "trace_trade_balance": "Balance commerciale",
        "title_balance": "{country} — Balance commerciale (milliards USD)",
        "trace_coverage_ratio": "Taux de couverture (%)",
        "balanced_100": "Équilibré (100 %)",
        "title_coverage": "{country} — Taux de couverture (Exportations/Importations × 100 %)",
        "unit_coverage_pct": "Couverture (%)",

        "sec_growth": "F.3.5 · Taux de croissance annuels",
        "title_growth": "{country} — Taux de croissance annuel des exportations & importations (%)",
        "unit_pct": "%",

        "sec_global_view": "F.3.6 · Vue globale du commerce — Aperçu",
        "title_volume": "Volume (milliards USD)",
        "unit_usd_b": "Milliards USD",
        "title_world_share": "Part mondiale (%)",
        "title_export_growth": "Croissance des exportations (%)",
        "trace_export_pct": "Export %",
        "title_trade_balance_b": "Balance commerciale (milliards USD)",
        "trace_balance": "Balance",
        "card_stability": "📊 Stabilité — Volatilité des exportations",
        "card_stability_sub": "σ de la croissance annuelle des exportations · plus bas = plus stable",
        "card_consistency": "📉 Constance — Fréquence de croissance négative",
        "card_consistency_sub": "{years} avec croissance des exportations négative",
        "years_of_10": "{n:.1f} / 10 ans",

        "sec_positioning": "F.4.5 · Positionnement mondial — Où se situe ce pays ?",
        "quad_tab_static": "📐 Quadrant statique",
        "quad_tab_dynamic": "⚡ Quadrant dynamique",
        "quad_tab_strategic": "⭐ Quadrant stratégique",
        "quad_tab_meta": "🌐 Profil méta",
        "other_countries": "Autres pays",
        "others": "Autres",

        "lbl_countries_to_animate": "Pays à animer",
        "title_static_anim": "Quadrant statique — Animé ({start}–{end})",
        "axis_export_share_size": "Part des exportations (%) — Taille commerciale",
        "axis_coverage_balance": "Taux de couverture (%) — Balance commerciale",
        "quad_dominant_desc": "🟢 Dominant<br><sub>Grande taille & excédent</sub>",
        "quad_balanced_desc": "🔵 Balanced<br><sub>Petite taille & excédent</sub>",
        "quad_vulnerable_desc": "🟡 Vulnerable<br><sub>Grande taille & déficit</sub>",
        "quad_marginal_desc": "⚫ Marginal<br><sub>Petite taille & déficit</sub>",
        "btn_play": "▶ Lecture",
        "btn_pause": "⏸ Pause",
        "slider_year_prefix": "Année : ",
        "badge_static_quadrant": "→ Quadrant statique : ",
        "title_static_dist": "Répartition du quadrant statique — Tous les pays",
        "axis_countries": "Pays",

        "lbl_highlight_countries": "Pays à mettre en évidence",
        "hover_cagr": "CAGR",
        "hover_volatility": "Volatilité",
        "hover_quadrant": "Quadrant",
        "hover_dynamic_quadrant": "Quadrant dynamique",
        "quad_star_desc": "⭐ Star<br>(Forte croissance · Faible volatilité)",
        "quad_stable_desc": "😴 Stable Stagnant<br>(Faible croissance · Faible volatilité)",
        "quad_risky_desc": "📈 Risky Grower<br>(Forte croissance · Forte volatilité)",
        "quad_declining_desc": "📉 Declining<br>(Faible croissance · Forte volatilité)",
        "title_dynamic_quadrant": "Quadrant dynamique — {country} vs tous les pays",
        "axis_cagr_export": "CAGR des exportations (%)",
        "axis_export_volatility": "Volatilité des exportations",
        "badge_dynamic_quadrant": "→ Quadrant dynamique : ",

        "hover_export_share": "Part des exportations",
        "hover_strategic_quadrant": "Quadrant stratégique",
        "quad_stars_desc": "⭐ Stars<br><sub>Grande taille & croissance</sub>",
        "quad_emerging_desc": "🌱 Emerging<br><sub>Petite taille & croissance</sub>",
        "quad_cashcows_desc": "🐄 Cash Cows<br><sub>Grande taille & stagnation</sub>",
        "quad_marginal2_desc": "📉 Marginal<br><sub>Petite taille & déclin</sub>",
        "title_strategic_quadrant": "Quadrant stratégique (BCG) — {country}",
        "axis_export_share_pct": "Part des exportations (%)",
        "badge_strategic_quadrant": "→ Quadrant stratégique : ",

        "title_meta_positioning": "Positionnement du profil méta — {country} mis en évidence",
        "axis_structural_score": "Score structurel",
        "hover_structural_score": "Score structurel",
        "hover_meta_profile": "Profil méta",
        "title_meta_dist": "Répartition des profils méta",
        "badge_meta_profile": "Profil méta de {country} : ",

        "expander_quad_defs": "Définitions des quadrants & profils méta",
        "quad_defs_col1": (
            "**Statique** (Part des exportations × Taux de couverture)\n"
            "- 🟢 **Dominant** — grand exportateur + excédent commercial\n"
            "- 🔵 **Balanced** — petit exportateur + excédent commercial\n"
            "- 🟡 **Vulnerable** — grand exportateur + déficit commercial\n"
            "- ⚫ **Marginal** — petit exportateur + déficit commercial\n\n"
            "**Dynamique** (CAGR × Volatilité, seuils : z=0 × volatilité médiane)\n"
            "- ⭐ **Star** — CAGR élevé, faible volatilité\n"
            "- 📈 **Risky Grower** — CAGR élevé, forte volatilité\n"
            "- 📉 **Declining** — CAGR faible, faible volatilité\n"
            "- 😴 **Stable Stagnant** — CAGR faible, forte volatilité\n\n"
            "**Stratégique** (Part des exportations × CAGR, seuils : 1 % × 0)\n"
            "- ⭐ **Stars** — part élevée & CAGR positif\n"
            "- 🌱 **Emerging** — part faible & CAGR positif\n"
            "- 🐄 **Cash Cows** — part élevée & CAGR négatif\n"
            "- 📉 **Marginal** — part faible & CAGR négatif"
        ),
        "quad_defs_col2": (
            "**Profil méta** (composite des 3 quadrants)\n\n"
            "Dérivé de la combinaison Statique × Dynamique × Stratégique :\n\n"
            "- 🌍 **Global Leader** — Dominant + Stars\n"
            "- 💪 **Established Power** — Dominant/Vulnerable + Stars/Cash Cow\n"
            "- ⬆️ **Rising Challenger** — Emerging + Star (petit mais stable & en croissance)\n"
            "- 🌪️ **Volatile Emerging** — Emerging + Risky Grower (petit + instable)\n"
            "- 📍 **Peripheral** — sinon (score faible, croissance faible)\n\n"
            "*Pas un système à 4 quadrants — classification composite\n"
            "issue de l'interaction de 3 dimensions de quadrants*"
        ),

        "sec_scoring": "F.3.7 · Scoring structurel",
        "md_profile_scorecard": "**Fiche de score du profil**",
        "score_size_profile": "Profil de taille",
        "score_balance_profile": "Profil de balance",
        "score_growth_profile": "Profil de croissance",
        "score_stability_profile": "Profil de stabilité",
        "score_consistency_profile": "Profil de constance",
        "lbl_structural_score": "Score structurel",
        "lbl_structural_profile": "Profil structurel",
        "lbl_meta_profile": "Profil méta",
        "lbl_quadrants": "Quadrants",
        "lbl_static": "Statique",
        "lbl_dynamic": "Dynamique",
        "lbl_strategic": "Stratégique",
        "sec_radar": "Radar — 5 dimensions",
        "lbl_compare_with": "Comparer avec",
        "dim_size": "Taille",
        "dim_balance": "Équilibre",
        "dim_growth": "Croissance",
        "dim_stability": "Stabilité",
        "dim_consistency": "Constance",
        "sec_structural_heatmap": "Carte de chaleur structurelle — Meilleurs & pires",
        "lbl_view": "Vue",
        "opt_top_performers": "Meilleures performances",
        "opt_bottom_performers": "Moins bonnes performances",
        "lbl_n_countries": "N pays",
        "expander_scoring_methodology": "Méthodologie de scoring",
        "md_scoring_methodology": (
            "**Score structurel** = `0,60 × z_export_share + 0,10 × z_coverage_ratio "
            "+ 0,15 × z_cagr_export − 0,10 × z_volatility_export − 0,05 × z_negfreq_export`\n\n"
            "**Seuils** : q85 → Strong · q50 → Good · q15 → Moderate · en dessous → Weak\n\n"
            "**Dimensions du radar** : toutes mappées sur 0→1, où 1 = meilleur profil\n"
            "(ex. Leader=1,0, Major=0,67, Mid=0,33, Small=0,0)"
        ),
    },
}

def _fmt(x):
    return format_value(x)


def _add_quad_labels(fig, x_thr, y_thr, x_range, y_range, labels):
    """Place quadrant labels in data coordinates relative to threshold + axis range.
    labels = [(quadrant, text, color), ...]  where quadrant in TL/TR/BL/BR
    """
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    positions = {
        "TR": ((x_thr + x_hi) / 2, (y_thr + y_hi) / 2),
        "TL": ((x_lo + x_thr) / 2, (y_thr + y_hi) / 2),
        "BR": ((x_thr + x_hi) / 2, (y_lo + y_thr) / 2),
        "BL": ((x_lo + x_thr) / 2, (y_lo + y_thr) / 2),
    }
    for quad, text, color in labels:
        xp, yp = positions.get(quad, (x_thr, y_thr))
        fig.add_annotation(x=xp, y=yp, text=text, showarrow=False,
                           font=dict(size=11, color=color, family="IBM Plex Sans"),
                           align="center", bgcolor="rgba(22,32,53,0.6)",
                           borderpad=3)
    return fig


# ── Paths ─────────────────────────────────────────────────────
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_trade      = pd.read_csv(TRADE_FILES / "df_trade.csv")
    df_structural = pd.read_csv(TRADE_FILES / "df_structural.csv")
    return df_trade, df_structural

df_trade, df_structural = load_data()
COUNTRIES = sorted(df_trade["country_name"].dropna().unique().tolist())
YEARS     = sorted(df_trade["year"].unique())

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

COLORS = {
    "export":   "#F59E0B",
    "import":   "#60A5FA",
    "balance":  "#10B981",
    "coverage": "#A78BFA",
}

PROFILE_COLORS = {
    "Strong":   "#10B981",
    "Good":     "#60A5FA",
    "Moderate": "#F59E0B",
    "Weak":     "#EF4444",
}

LEVEL_COLORS = {1: "#10B981", 2: "#60A5FA", 3: "#F59E0B", 4: "#EF4444"}

LABEL_LEVELS = {
    "size_profile":        {"Leader": 1, "Major": 2, "Mid": 3, "Small": 4},
    "balance_profile":     {"Strong Surplus": 1, "Slight Surplus": 2,
                            "Slight Deficit": 3, "Structural Deficit": 4},
    "growth_profile":      {"Strong Grower": 1, "Moderate Grower": 2,
                            "Weak Grower": 3, "Declining": 4},
    "stability_profile":   {"Stable": 1, "Moderate": 2, "Volatile": 3, "Very Volatile": 4},
    "consistency_profile": {"Very Consistent": 1, "Consistent": 2,
                            "Inconsistent": 3, "Very Inconsistent": 4},
}

RADAR_LEVELS = {
    "size_profile":        {"Leader": 1.0, "Major": 0.67, "Mid": 0.33, "Small": 0.0},
    "balance_profile":     {"Strong Surplus": 1.0, "Slight Surplus": 0.67,
                            "Slight Deficit": 0.33, "Structural Deficit": 0.0},
    "growth_profile":      {"Strong Grower": 1.0, "Moderate Grower": 0.67,
                            "Weak Grower": 0.33, "Declining": 0.0},
    "stability_profile":   {"Stable": 1.0, "Moderate": 0.67, "Volatile": 0.33, "Very Volatile": 0.0},
    "consistency_profile": {"Very Consistent": 1.0, "Consistent": 0.67,
                            "Inconsistent": 0.33, "Very Inconsistent": 0.0},
}

PLOTLY_LAYOUT_COMPACT = {k: v for k, v in PLOTLY_LAYOUT.items()
                         if k not in ("margin","yaxis","xaxis","legend",
                                      "paper_bgcolor","plot_bgcolor")}

DIMS = ["size_profile", "balance_profile", "growth_profile",
        "stability_profile", "consistency_profile"]
DIM_LABELS = [t(STRINGS, "dim_size"), t(STRINGS, "dim_balance"), t(STRINGS, "dim_growth"),
              t(STRINGS, "dim_stability"), t(STRINGS, "dim_consistency")]


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>', unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()


# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR
# ════════════════════════════════════════════════════════════════

col_sel, col_info = st.columns([1, 3])
with col_sel:
    default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox(t(STRINGS, "lbl_select_country"), COUNTRIES, index=default_idx, key="t3_country")

# Filter data for selected country
df_c = df_trade[df_trade["country_name"] == country].sort_values("year").copy()
df_s = df_structural[df_structural["country_name"] == country]

# Country KPIs
with col_info:
    if not df_c.empty and not df_s.empty:
        row_s = df_s.iloc[0]
        latest = df_c[df_c["year"] == df_c["year"].max()].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi(t(STRINGS, "kpi_exports_latest"),
            f"${latest['X']/1e9:.1f}B", "amber"), unsafe_allow_html=True)
        k2.markdown(kpi(t(STRINGS, "kpi_export_share"),
            f"{row_s['export_share_structural']:.2f}%", "teal"), unsafe_allow_html=True)
        k3.markdown(kpi(t(STRINGS, "kpi_structural_score"),
            f"{row_s['structural_score']:.2f}", "blue"), unsafe_allow_html=True)
        k4.markdown(kpi(t(STRINGS, "kpi_profile"),
            row_s.get("structural_profile", "—"), ""), unsafe_allow_html=True)

thin_divider()

if df_c.empty:
    st.warning(t(STRINGS, "warn_no_data").format(country=country))
    st.stop()


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t(STRINGS, "tab_trends"),
    t(STRINGS, "tab_balance"),
    t(STRINGS, "tab_growth"),
    t(STRINGS, "tab_global_view"),
    t(STRINGS, "tab_scoring"),
    t(STRINGS, "tab_positioning"),
])


# ── TAB 1 : TRENDS ───────────────────────────────────────────
with tab1:
    section_label(t(STRINGS, "sec_trends"))
    col1, col2 = st.columns(2)

    with col1:
        fig_trend = go.Figure()
        df_c["_fmtX"] = df_c["X"].apply(format_value)
        df_c["_fmtM"] = df_c["M"].apply(format_value)
        fig_trend.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["X"] / 1e9,
            mode="lines+markers", name=t(STRINGS, "opt_exports"),
            line=dict(color=COLORS["export"], width=2.5),
            marker=dict(size=7),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
            customdata=df_c[["_fmtX"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_exports") + ": $%{customdata[0]}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["M"] / 1e9,
            mode="lines+markers", name=t(STRINGS, "opt_imports"),
            line=dict(color=COLORS["import"], width=2.5, dash="dash"),
            marker=dict(size=7),
            customdata=df_c[["_fmtM"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_imports") + ": $%{customdata[0]}<extra></extra>",
        ))
        fig_trend.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_trends").format(country=country),
            yaxis_title=t(STRINGS, "unit_usd_billion"),
        )
        st.plotly_chart(fig_trend, width='stretch')

    with col2:
        section_label(t(STRINGS, "sec_share_trends"))
        fig_share = go.Figure()
        fig_share.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["export_share"],
            mode="lines+markers", name=t(STRINGS, "opt_export_share"),
            line=dict(color=COLORS["export"], width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_export_share") + ": %{y:.3f}%<extra></extra>",
        ))
        fig_share.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["import_share"],
            mode="lines+markers", name=t(STRINGS, "opt_import_share"),
            line=dict(color=COLORS["import"], width=2.5, dash="dash"),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_import_share") + ": %{y:.3f}%<extra></extra>",
        ))
        fig_share.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_share_trends").format(country=country),
            yaxis_title=t(STRINGS, "unit_share"),
        )
        st.plotly_chart(fig_share, width='stretch')


# ── TAB 2 : BALANCE ──────────────────────────────────────────
with tab2:
    section_label(t(STRINGS, "sec_balance"))
    col1, col2 = st.columns(2)

    with col1:
        fig_bal = go.Figure()
        # Fill surplus / deficit
        df_c["_fmtBal"] = df_c["trade_balance_B"].apply(lambda v: f"${_fmt(v)}B")
        fig_bal.add_trace(go.Bar(
            x=df_c["year"],
            y=df_c["trade_balance_B"],
            name=t(STRINGS, "trace_trade_balance"),
            marker_color=[
                COLORS["balance"] if v >= 0 else "#EF4444"
                for v in df_c["trade_balance_B"]
            ],
            opacity=0.8,
            customdata=df_c[["_fmtBal"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "trace_trade_balance") + ": %{customdata[0]}<extra></extra>",
        ))
        fig_bal.add_hline(y=0, line_color="#EEF4FB", line_width=1)
        fig_bal.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_balance").format(country=country),
            yaxis_title=t(STRINGS, "unit_usd_billion"),
        )
        st.plotly_chart(fig_bal, width='stretch')

    with col2:
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Scatter(
            x=df_c["year"], y=df_c["coverage_ratio"] * 100,
            mode="lines+markers", name=t(STRINGS, "trace_coverage_ratio"),
            line=dict(color=COLORS["coverage"], width=2.5),
            marker=dict(size=7),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
            customdata=df_c[["coverage_ratio"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "unit_coverage_pct") + ": %{customdata[0]:.1%}<extra></extra>",
        ))
        fig_cov.add_hline(y=100, line_dash="dash", line_color="#EEF4FB",
                          annotation_text=t(STRINGS, "balanced_100"))
        fig_cov.update_layout(**PLOTLY_LAYOUT,
            title=t(STRINGS, "title_coverage").format(country=country),
            yaxis_title=t(STRINGS, "unit_coverage_pct"),
        )
        st.plotly_chart(fig_cov, width='stretch')


# ── TAB 3 : GROWTH ───────────────────────────────────────────
with tab3:
    section_label(t(STRINGS, "sec_growth"))
    df_g = df_c.dropna(subset=["export_growth", "import_growth"]).copy()

    fig_growth = go.Figure()
    fig_growth.add_hline(y=0, line_color="#EEF4FB", line_width=1, line_dash="dash")

    fig_growth.add_trace(go.Bar(
        x=df_g["year"], y=df_g["export_growth"],
        name=t(STRINGS, "opt_exports"),
        marker_color=["rgba(245,158,11,0.9)" if v >= 0 else "rgba(245,158,11,0.35)"
                      for v in df_g["export_growth"]],
        offsetgroup=0,
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_exports") + ": %{y:.1f}%<extra></extra>",
    ))
    fig_growth.add_trace(go.Bar(
        x=df_g["year"], y=df_g["import_growth"],
        name=t(STRINGS, "opt_imports"),
        marker_color=["rgba(96,165,250,0.9)" if v >= 0 else "rgba(96,165,250,0.35)"
                      for v in df_g["import_growth"]],
        offsetgroup=1,
        hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_imports") + ": %{y:.1f}%<extra></extra>",
    ))
    fig_growth.update_layout(**PLOTLY_LAYOUT,
        title=t(STRINGS, "title_growth").format(country=country),
        yaxis_title=t(STRINGS, "unit_pct"),
        barmode="group",
        height=420,
    )
    st.plotly_chart(fig_growth, width='stretch')


# ── TAB 4 : GLOBAL VIEW ──────────────────────────────────────
with tab4:
    section_label(t(STRINGS, "sec_global_view"))
    df_g = df_c.dropna(subset=["export_growth", "import_growth"]).copy()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    # Subplot 1 — Volume
    with r1c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_c["year"], y=df_c["X"]/1e9,
            mode="lines+markers", name=t(STRINGS, "opt_exports"),
            line=dict(color=COLORS["export"], width=2),
            customdata=df_c[["_fmtX"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_exports") + ": $%{customdata[0]}<extra></extra>"))
        fig1.add_trace(go.Scatter(x=df_c["year"], y=df_c["M"]/1e9,
            mode="lines+markers", name=t(STRINGS, "opt_imports"),
            line=dict(color=COLORS["import"], width=2, dash="dash"),
            customdata=df_c[["_fmtM"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_imports") + ": $%{customdata[0]}<extra></extra>"))
        fig1.update_layout(**PLOTLY_LAYOUT_COMPACT,
            title=t(STRINGS, "title_volume"), yaxis_title=t(STRINGS, "unit_usd_b"), height=300,
            margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig1, width='stretch')

    # Subplot 2 — Share
    with r1c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_c["year"], y=df_c["export_share"],
            mode="lines+markers", name=t(STRINGS, "opt_export_share"),
            line=dict(color=COLORS["export"], width=2),
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_export_share") + ": %{y:.3f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=df_c["year"], y=df_c["import_share"],
            mode="lines+markers", name=t(STRINGS, "opt_import_share"),
            line=dict(color=COLORS["import"], width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_import_share") + ": %{y:.3f}%<extra></extra>"))
        fig2.update_layout(**PLOTLY_LAYOUT_COMPACT,
            title=t(STRINGS, "title_world_share"), yaxis_title=t(STRINGS, "unit_pct"), height=300,
            margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig2, width='stretch')

    # Subplot 3 — Growth
    with r2c1:
        fig3 = go.Figure()
        fig3.add_hline(y=0, line_color="#EEF4FB", line_width=1)
        fig3.add_trace(go.Bar(x=df_g["year"], y=df_g["export_growth"],
            name=t(STRINGS, "trace_export_pct"),
            marker_color=[COLORS["export"] if v >= 0 else "#EF4444"
                          for v in df_g["export_growth"]], opacity=0.8,
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "opt_exports") + ": %{y:.1f}%<extra></extra>"))
        fig3.update_layout(**PLOTLY_LAYOUT_COMPACT,
            title=t(STRINGS, "title_export_growth"), yaxis_title=t(STRINGS, "unit_pct"), height=300,
            margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig3, width='stretch')

    # Subplot 4 — Balance
    with r2c2:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=df_c["year"], y=df_c["trade_balance_B"],
            name=t(STRINGS, "trace_balance"),
            marker_color=[COLORS["balance"] if v >= 0 else "#EF4444"
                          for v in df_c["trade_balance_B"]], opacity=0.8,
            customdata=df_c[["_fmtBal"]],
            hovertemplate="<b>%{x}</b><br>" + t(STRINGS, "trace_balance") + ": %{customdata[0]}<extra></extra>"))
        fig4.add_hline(y=0, line_color="#EEF4FB", line_width=1)
        fig4.update_layout(**PLOTLY_LAYOUT_COMPACT,
            title=t(STRINGS, "title_trade_balance_b"), yaxis_title=t(STRINGS, "unit_usd_b"), height=300,
            margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig4, width='stretch')

    # Row 3 — Stability & Consistency metrics
    st.markdown("<div style='margin-top:.5rem;'></div>", unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        # Stability — volatility_export from df_structural
        _vol_val = df_s.iloc[0].get("volatility_export", None) if not df_s.empty else None
        _vol_lbl = "—" if _vol_val is None else f"{_vol_val:.2f}"
        _vol_color = ("#10B981" if _vol_val is not None and _vol_val < 10
                      else "#60A5FA" if _vol_val is not None and _vol_val < 20
                      else "#F59E0B" if _vol_val is not None and _vol_val < 40
                      else "#EF4444")
        st.markdown(f"""
        <div style="background:#1F3050;border:1px solid #304870;
                    border-top:3px solid {_vol_color};border-radius:12px;
                    padding:.85rem 1rem;text-align:center;height:100%;">
          <div style="font-size:11px;color:#6890B0;text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:.4rem;">
            {t(STRINGS, "card_stability")}
          </div>
          <div style="font-size:32px;font-weight:700;color:{_vol_color};">
            {_vol_lbl}
          </div>
          <div style="font-size:12px;color:#6890B0;margin-top:.3rem;">
            {t(STRINGS, "card_stability_sub")}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r3c2:
        # Consistency — negfreq_export from df_structural
        _neg_val = df_s.iloc[0].get("negfreq_export", None) if not df_s.empty else None
        _neg_lbl = "—" if _neg_val is None else f"{_neg_val:.0%}"
        _neg_years = "—" if _neg_val is None else t(STRINGS, "years_of_10").format(n=_neg_val * 10)
        _neg_color = ("#10B981" if _neg_val is not None and _neg_val < 0.2
                      else "#60A5FA" if _neg_val is not None and _neg_val < 0.3
                      else "#F59E0B" if _neg_val is not None and _neg_val < 0.5
                      else "#EF4444")
        st.markdown(f"""
        <div style="background:#1F3050;border:1px solid #304870;
                    border-top:3px solid {_neg_color};border-radius:12px;
                    padding:.85rem 1rem;text-align:center;height:100%;">
          <div style="font-size:11px;color:#6890B0;text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:.4rem;">
            {t(STRINGS, "card_consistency")}
          </div>
          <div style="font-size:32px;font-weight:700;color:{_neg_color};">
            {_neg_lbl}
          </div>
          <div style="font-size:12px;color:#6890B0;margin-top:.3rem;">
            {t(STRINGS, "card_consistency_sub").format(years=_neg_years)}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 6 : POSITIONING ──────────────────────────────────────
with tab6:
    section_label(t(STRINGS, "sec_positioning"))

    if df_s.empty:
        st.warning(t(STRINGS, "warn_no_structural").format(country=country))
    else:
        row_s = df_s.iloc[0]

        quad_tabs = st.tabs([t(STRINGS, "quad_tab_static"),
                              t(STRINGS, "quad_tab_dynamic"),
                              t(STRINGS, "quad_tab_strategic"),
                              t(STRINGS, "quad_tab_meta")])

        QUAD_COLORS = {
            # Static
            "Dominant":       "#10B981",
            "Balanced":       "#60A5FA",
            "Vulnerable":     "#F59E0B",
            "Marginal":       "#6B7280",
            # Dynamic
            "Star":           "#10B981",
            "Risky Grower":   "#F59E0B",
            "Declining":      "#EF4444",
            "Stable Stagnant":"#60A5FA",
            # Strategic
            "Stars":          "#10B981",
            "Emerging":       "#60A5FA",
        }

        def _scatter_quadrant(x_col, y_col, x_thr, y_thr,
                              x_label, y_label, title,
                              q_col, highlight):
            df_all = df_structural.copy().dropna(subset=[x_col, y_col])

            # Size proportional to structural_score rank
            df_all["_size"] = (df_all["structural_score"].rank() /
                               len(df_all) * 25 + 6).clip(6, 31)

            fig = go.Figure()

            # All countries — grey background
            df_bg = df_all[df_all["country_name"] != highlight]
            fig.add_trace(go.Scatter(
                x=df_bg[x_col], y=df_bg[y_col],
                mode="markers",
                name=t(STRINGS, "other_countries"),
                marker=dict(
                    color=[QUAD_COLORS.get(q, "#6B7280")
                            for q in df_bg[q_col]],
                    size=df_bg["_size"], opacity=0.45,
                    line=dict(width=0),
                ),
                customdata=df_bg[["country_name", q_col]],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              f"{x_label}: %{{x:.3f}}<br>"
                              f"{y_label}: %{{y:.3f}}<br>"
                              "Quadrant: %{customdata[1]}<extra></extra>",
                showlegend=False,
            ))

            # Highlighted country
            row_h = df_all[df_all["country_name"] == highlight]
            if not row_h.empty:
                r = row_h.iloc[0]
                color_h = QUAD_COLORS.get(r.get(q_col, ""), "#EEF4FB")
                fig.add_trace(go.Scatter(
                    x=[r[x_col]], y=[r[y_col]],
                    mode="markers+text",
                    name=highlight,
                    marker=dict(color=color_h, size=18, symbol="star",
                                line=dict(width=1.5, color="#EEF4FB")),
                    text=[highlight],
                    textposition="top center",
                    textfont=dict(size=12, color="#EEF4FB"),
                    hovertemplate=f"<b>{highlight}</b><br>"
                                  f"{x_label}: {r[x_col]:.3f}<br>"
                                  f"{y_label}: {r[y_col]:.3f}<br>"
                                  f"Quadrant: {r.get(q_col, '—')}"
                                  "<extra></extra>",
                ))

            # Threshold lines
            fig.add_vline(x=x_thr, line_dash="dash",
                          line_color="#304870", line_width=1.5,
                          annotation_text=f"{x_label}={x_thr}")
            fig.add_hline(y=y_thr, line_dash="dash",
                          line_color="#304870", line_width=1.5,
                          annotation_text=f"{y_label}={y_thr}")

            fig.update_layout(**PLOTLY_LAYOUT,
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=520,
            )
            return fig

        with quad_tabs[0]:
            # Animation: export_share × coverage_ratio per year from df_trade
            _hl_sq = st.multiselect(
                t(STRINGS, "lbl_countries_to_animate"),
                options=[c for c in COUNTRIES if c != country],
                default=[cc for cc in ["USA","China","Germany","Canada","Brazil","Japan","Tunisia","Qatar"]
                         if cc in COUNTRIES and cc != country],
                max_selections=9,
                key="sq_anim_hl",
            )
            _anim_countries = [country] + _hl_sq

            df_anim = df_trade[df_trade["country_name"].isin(_anim_countries)].copy()
            df_anim["export_share_pct"] = df_anim["export_share"]     # already in %
            df_anim["coverage_ratio_pct"] = df_anim["coverage_ratio"] * 100  # decimal → %
            df_anim["coverage_ratio_pct"] = df_anim["coverage_ratio"] * 100

            # Thresholds from full period
            x_thr = 1.0    # notebook: export_share >= 1.0%
            y_thr = 100.0  # notebook: coverage_ratio >= 1.0 → ×100 = 100%

            ANIM_PAL = ["#EEF4FB","#60A5FA","#10B981","#F59E0B",
                        "#EF4444","#A78BFA","#F472B6","#34D399",
                        "#FB923C","#38BDF8"]
            anim_cmap = {cc: ANIM_PAL[i] for i, cc in enumerate(_anim_countries)}

            fig_sq_anim = go.Figure()

            # Build frames
            frames = []
            years_anim = sorted(df_anim["year"].unique())
            for yr in years_anim:
                df_yr = df_anim[df_anim["year"] == yr]
                frame_data = []
                for cc in _anim_countries:
                    row_yr = df_yr[df_yr["country_name"] == cc]
                    if row_yr.empty:
                        frame_data.append(go.Scatter(x=[None], y=[None]))
                        continue
                    r = row_yr.iloc[0]
                    frame_data.append(go.Scatter(
                        x=[r["export_share_pct"]],
                        y=[r["coverage_ratio_pct"]],
                        mode="markers+text",
                        marker=dict(
                            color=anim_cmap[cc],
                            size=16 if cc == country else 12,
                            symbol="star" if cc == country else "circle",
                            line=dict(width=1.5, color="#304870"),
                        ),
                        text=[cc],
                        textposition="top center",
                        textfont=dict(size=11, color="#EEF4FB"),
                        name=cc,
                        showlegend=True,
                    ))
                frames.append(go.Frame(data=frame_data, name=str(yr)))

            # Initial frame
            df_yr0 = df_anim[df_anim["year"] == years_anim[0]]
            for cc in _anim_countries:
                row_yr = df_yr0[df_yr0["country_name"] == cc]
                if row_yr.empty:
                    fig_sq_anim.add_trace(go.Scatter(x=[None], y=[None],
                        name=cc, mode="markers+text",
                        marker=dict(color=anim_cmap[cc], size=12)))
                    continue
                r = row_yr.iloc[0]
                fig_sq_anim.add_trace(go.Scatter(
                    x=[r["export_share_pct"]],
                    y=[r["coverage_ratio_pct"]],
                    mode="markers+text",
                    name=cc,
                    marker=dict(color=anim_cmap[cc],
                                size=16 if cc == country else 12,
                                symbol="star" if cc == country else "circle",
                                line=dict(width=1.5, color="#304870")),
                    text=[cc],
                    textposition="top center",
                    textfont=dict(size=11, color="#EEF4FB"),
                ))

            fig_sq_anim.frames = frames

            # Fixed range from ALL selected countries across all years
            _df_sel = df_trade[df_trade["country_name"].isin(_anim_countries)]
            _x_max = _df_sel["export_share"].max() * 1.05
            _y_min = _df_sel["coverage_ratio"].min() * 100 * 0.95
            _y_max = _df_sel["coverage_ratio"].max() * 100 * 1.05

            # Quadrant boundary lines
            fig_sq_anim.add_vline(x=x_thr, line_dash="dash",
                                   line_color="#6890B0", line_width=2)
            fig_sq_anim.add_hline(y=y_thr, line_dash="dash",
                                   line_color="#6890B0", line_width=2)

            # Label range matches axis range
            _sq_xlo = 0
            _sq_xhi = _x_max
            _sq_ylo = _y_min   # already ×100
            _sq_yhi = _y_max   # already ×100

            # Quadrant label positions — midpoint of each quadrant
            _sq_x_right = (x_thr + _sq_xhi) / 2
            _sq_x_left  = (_sq_xlo + x_thr) / 2
            _sq_y_top   = (y_thr + _sq_yhi) / 2
            _sq_y_bot   = (_sq_ylo + y_thr) / 2

            # Build base annotations (quadrant labels) — must persist through frames
            _quad_annotations = [
                dict(x=_sq_x_right, y=_sq_y_top, text=t(STRINGS, "quad_dominant_desc"),
                     showarrow=False, font=dict(size=10, color="#10B981"),
                     align="center", bgcolor="rgba(22,32,53,0.75)", borderpad=4),
                dict(x=_sq_x_left,  y=_sq_y_top, text=t(STRINGS, "quad_balanced_desc"),
                     showarrow=False, font=dict(size=10, color="#60A5FA"),
                     align="center", bgcolor="rgba(22,32,53,0.75)", borderpad=4),
                dict(x=_sq_x_right, y=_sq_y_bot, text=t(STRINGS, "quad_vulnerable_desc"),
                     showarrow=False, font=dict(size=10, color="#F59E0B"),
                     align="center", bgcolor="rgba(22,32,53,0.75)", borderpad=4),
                dict(x=_sq_x_left,  y=_sq_y_bot, text=t(STRINGS, "quad_marginal_desc"),
                     showarrow=False, font=dict(size=10, color="#6B7280"),
                     align="center", bgcolor="rgba(22,32,53,0.75)", borderpad=4),
            ]

            # Year watermark + quadrant labels in every frame
            for fr in fig_sq_anim.frames:
                yr_annotation = dict(
                    x=0.98, y=0.95, xref="paper", yref="paper",
                    text=fr.name, showarrow=False,
                    font=dict(size=28, color="rgba(255,255,255,0.15)"),
                    xanchor="right",
                )
                fr.layout = go.Layout(
                    annotations=_quad_annotations + [yr_annotation]
                )

            fig_sq_anim.update_layout(
                **PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_static_anim").format(start=years_anim[0], end=years_anim[-1]),
                xaxis=dict(title=t(STRINGS, "axis_export_share_size"),
                           range=[0, _x_max],
                           gridcolor="#304870", linecolor="#304870"),
                yaxis=dict(title=t(STRINGS, "axis_coverage_balance"),
                           range=[_y_min, _y_max],
                           gridcolor="#304870", linecolor="#304870"),
                annotations=_quad_annotations + [dict(
                    x=0.98, y=0.95, xref="paper", yref="paper",
                    text=str(years_anim[0]), showarrow=False,
                    font=dict(size=28, color="rgba(255,255,255,0.15)"),
                    xanchor="right",
                )],
                height=580,
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    direction="left",
                    y=1.08, x=0.5, xanchor="center",
                    pad={"r": 10, "t": 10},
                    buttons=[
                        dict(label=t(STRINGS, "btn_play"),
                             method="animate",
                             args=[None, dict(frame=dict(duration=1200, redraw=True),
                                             fromcurrent=True, mode="immediate")]),
                        dict(label=t(STRINGS, "btn_pause"),
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode="immediate")]),
                    ],
                )],
                sliders=[dict(
                    active=0,
                    steps=[dict(args=[[str(yr)],
                                      dict(frame=dict(duration=600, redraw=True),
                                           mode="immediate")],
                                label=str(yr), method="animate")
                           for yr in years_anim],
                    x=0, y=0, len=1.0,
                    currentvalue=dict(prefix=t(STRINGS, "slider_year_prefix"), font=dict(size=14)),
                    pad=dict(t=50),
                )],
            )
            st.plotly_chart(fig_sq_anim, width='stretch')

            # Static quadrant badge
            if not df_s.empty:
                _sq_val   = df_s.iloc[0].get("static_quadrant", "—")
                _sq_color = QUAD_COLORS.get(_sq_val, "#C4D5E8")
                st.markdown(f"""
                <div style="background:#1A2C48;border-left:4px solid {_sq_color};
                            border-radius:0 10px 10px 0;padding:.75rem 1rem;">
                  <span style="color:{_sq_color};font-weight:600;">{country}</span>
                  <span style="color:#A8C0D8;"> {t(STRINGS, "badge_static_quadrant")}</span>
                  <span style="color:{_sq_color};font-weight:600;">{_sq_val}</span>
                </div>
                """, unsafe_allow_html=True)

            # Distribution bar
            q_counts = df_structural["static_quadrant"].value_counts().reset_index()
            q_counts.columns = ["Quadrant","Count"]
            fig_qdist = go.Figure(go.Bar(
                x=q_counts["Count"], y=q_counts["Quadrant"],
                orientation="h",
                marker_color=[QUAD_COLORS.get(q,"#6B7280") for q in q_counts["Quadrant"]],
                text=q_counts["Count"], textposition="outside", opacity=0.85,
            ))
            fig_qdist.update_layout(**PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                xaxis=dict(title=t(STRINGS, "axis_countries"), gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"),
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_static_dist"),
                height=280,
                margin=dict(l=40,r=40,t=40,b=20),
            )
            st.plotly_chart(fig_qdist, width='stretch')

        with quad_tabs[1]:
            _hl_dq = st.multiselect(
                t(STRINGS, "lbl_highlight_countries"),
                [c for c in COUNTRIES if c != country],
                default=[cc for cc in ["China","USA","Germany","Canada","Brazil","Qatar"]
                         if cc in COUNTRIES and cc != country],
                max_selections=6, key="dq_hl",
            )
            _dq_all = [country] + _hl_dq
            _DQ_PAL = ["#EEF4FB","#60A5FA","#10B981","#F59E0B","#EF4444","#A78BFA","#F472B6"]
            DYN_COLORS = {"Star":"#10B981","Risky Grower":"#F59E0B",
                          "Declining":"#EF4444","Stable Stagnant":"#60A5FA"}

            # Use actual cagr_export and volatility_export (not z-scores)
            df_dq_all = df_structural.copy().dropna(
                subset=["cagr_export","volatility_export","dynamic_quadrant"])

            _vol_median = df_dq_all["volatility_export"].median()
            _cagr_thr   = 0.0  # notebook threshold: cagr >= 0

            fig_dq = go.Figure()

            df_dq_bg = df_dq_all[~df_dq_all["country_name"].isin(_dq_all)]
            fig_dq.add_trace(go.Scatter(
                x=df_dq_bg["cagr_export"] * 100,
                y=df_dq_bg["volatility_export"],
                mode="markers", name=t(STRINGS, "others"),
                marker=dict(color=[DYN_COLORS.get(q,"#6B7280")
                                   for q in df_dq_bg["dynamic_quadrant"]],
                            size=7, opacity=0.35, line=dict(width=0)),
                customdata=df_dq_bg[["country_name","dynamic_quadrant","cagr_export","volatility_export"]],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              + t(STRINGS, "hover_cagr") + ": %{customdata[2]:.2%}<br>"
                              + t(STRINGS, "hover_volatility") + ": %{customdata[3]:.2f}<br>"
                              + t(STRINGS, "hover_quadrant") + ": %{customdata[1]}<extra></extra>",
                showlegend=False,
            ))
            for i, cc in enumerate(_dq_all):
                rr = df_dq_all[df_dq_all["country_name"] == cc]
                if rr.empty: continue
                rr = rr.iloc[0]
                fig_dq.add_trace(go.Scatter(
                    x=[rr["cagr_export"] * 100], y=[rr["volatility_export"]],
                    mode="markers+text", name=cc,
                    marker=dict(color=_DQ_PAL[i % len(_DQ_PAL)],
                                size=18 if cc == country else 14,
                                symbol="star" if cc == country else "circle",
                                line=dict(width=1.5, color="#304870")),
                    text=[cc], textposition="top center",
                    textfont=dict(size=11, color="#EEF4FB"),
                    customdata=[[rr.get("dynamic_quadrant","—"),
                                 f"{rr['cagr_export']:.2%}",
                                 f"{rr['volatility_export']:.2f}"]],
                    hovertemplate=(
                        f"<b>{cc}</b><br>"
                        + t(STRINGS, "hover_cagr") + ": %{customdata[1]}<br>"
                        + t(STRINGS, "hover_volatility") + ": %{customdata[2]}<br>"
                        + t(STRINGS, "hover_dynamic_quadrant") + ": %{customdata[0]}<extra></extra>"
                    ),
                ))

            # Threshold lines
            fig_dq.add_vline(x=_cagr_thr * 100, line_dash="dash",
                             line_color="#6890B0", line_width=2)
            fig_dq.add_hline(y=_vol_median, line_dash="dash",
                             line_color="#6890B0", line_width=2)

            # Labels using notebook formula
            _ylim_cap = df_dq_all["volatility_export"].quantile(0.95)
            _xlim = [df_dq_all["cagr_export"].quantile(0.03) * 100,
                     df_dq_all["cagr_export"].quantile(0.97) * 100]
            # Use visual cap for label positioning (not raw max)
            _ylim = [df_dq_all["volatility_export"].min(), _ylim_cap]

            # Label positions: midpoint of each quadrant
            _x_right = (_cagr_thr * 100 + _xlim[1]) / 2    # midpoint right half
            _x_left  = (_xlim[0] + _cagr_thr * 100) / 2    # midpoint left half
            _y_low   = (_ylim[0] + _vol_median) / 2         # midpoint bottom half
            _y_high  = (_vol_median + _ylim[1]) / 2         # midpoint top half

            for txt, xp, yp, clr in [
                (t(STRINGS, "quad_star_desc"),     _x_right, _y_low,  "#10B981"),
                (t(STRINGS, "quad_stable_desc"),   _x_left,  _y_low,  "#60A5FA"),
                (t(STRINGS, "quad_risky_desc"),    _x_right, _y_high, "#F59E0B"),
                (t(STRINGS, "quad_declining_desc"),_x_left,  _y_high, "#EF4444"),
            ]:
                fig_dq.add_annotation(
                    x=xp, y=yp, text=txt, showarrow=False,
                    font=dict(size=10, color=clr),
                    align="center",
                    bgcolor="rgba(22,32,53,0.75)", borderpad=4,
                )

            fig_dq.update_layout(**PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_dynamic_quadrant").format(country=country),
                xaxis=dict(title=t(STRINGS, "axis_cagr_export"),
                           range=[_xlim[0], _xlim[1]], gridcolor="#304870"),
                yaxis=dict(title=t(STRINGS, "axis_export_volatility"),
                           range=[0, _ylim_cap], gridcolor="#304870"),
                height=520,
            )
            st.plotly_chart(fig_dq, width='stretch')

            q_val = row_s.get("dynamic_quadrant", "—")
            q_color = QUAD_COLORS.get(q_val, "#C4D5E8")
            st.markdown(f"""
            <div style="background:#1A2C48;border-left:4px solid {q_color};
                        border-radius:0 10px 10px 0;padding:.75rem 1rem;">
              <span style="color:{q_color};font-weight:600;">{country}</span>
              <span style="color:#A8C0D8;"> {t(STRINGS, "badge_dynamic_quadrant")}</span>
              <span style="color:{q_color};font-weight:600;">{q_val}</span>
            </div>
            """, unsafe_allow_html=True)

        with quad_tabs[2]:
            _hl_strat = st.multiselect(
                t(STRINGS, "lbl_highlight_countries"),
                [c for c in COUNTRIES if c != country],
                default=[cc for cc in ["USA","China","Germany","Canada","Brazil","Qatar"]
                         if cc in COUNTRIES and cc != country],
                max_selections=6, key="strat_hl",
            )
            _p95_share = df_structural["export_share_structural"].quantile(0.97)
            _p95_cagr  = df_structural["cagr_export"].quantile(0.97)
            _p05_cagr  = df_structural["cagr_export"].quantile(0.03)
            _x_thr_s   = 1.0  # notebook: export_share >= 1.0% threshold
            _strat_all = [country] + _hl_strat
            _STRAT_PAL = ["#EEF4FB","#60A5FA","#10B981","#F59E0B","#EF4444","#A78BFA","#F472B6"]
            STRAT_COLORS = {"Stars":"#10B981","Emerging":"#60A5FA",
                            "Cash Cows":"#A78BFA","Marginal":"#6B7280"}

            df_strat_all = df_structural.dropna(
                subset=["export_share_structural","cagr_export","strategic_quadrant"])
            fig_strat = go.Figure()

            df_strat_bg = df_strat_all[~df_strat_all["country_name"].isin(_strat_all)]
            fig_strat.add_trace(go.Scatter(
                x=df_strat_bg["export_share_structural"], y=df_strat_bg["cagr_export"],
                mode="markers", name=t(STRINGS, "others"),
                marker=dict(color=[STRAT_COLORS.get(q,"#6B7280")
                                   for q in df_strat_bg["strategic_quadrant"]],
                            size=7, opacity=0.35, line=dict(width=0)),
                customdata=df_strat_bg[["country_name","strategic_quadrant"]],
                hovertemplate="<b>%{customdata[0]}</b><br>" + t(STRINGS, "hover_quadrant") + ": %{customdata[1]}<extra></extra>",
                showlegend=False,
            ))
            for i, cc in enumerate(_strat_all):
                rr = df_strat_all[df_strat_all["country_name"] == cc]
                if rr.empty: continue
                rr = rr.iloc[0]
                fig_strat.add_trace(go.Scatter(
                    x=[rr["export_share_structural"]], y=[rr["cagr_export"]],
                    mode="markers+text", name=cc,
                    marker=dict(color=_STRAT_PAL[i % len(_STRAT_PAL)],
                                size=18 if cc == country else 14,
                                symbol="star" if cc == country else "circle",
                                line=dict(width=1.5, color="#304870")),
                    text=[cc], textposition="top center",
                    textfont=dict(size=11, color="#EEF4FB"),
                    customdata=[[rr.get("strategic_quadrant","—"),
                                 f"{rr['export_share_structural']:.3f}",
                                 f"{rr['cagr_export']*100:.2f}%"]],
                    hovertemplate=(
                        f"<b>{cc}</b><br>"
                        + t(STRINGS, "hover_export_share") + ": %{customdata[1]}<br>"
                        + t(STRINGS, "hover_cagr") + ": %{customdata[2]}<br>"
                        + t(STRINGS, "hover_strategic_quadrant") + ": %{customdata[0]}<extra></extra>"
                    ),
                ))
            fig_strat.add_vline(x=_x_thr_s, line_dash="dash",
                                line_color="#6890B0", line_width=2)
            fig_strat.add_hline(y=0, line_dash="dash",
                                line_color="#6890B0", line_width=2)
            _sq_xlo_s = df_strat_all["export_share_structural"].min()
            _sq_xhi_s = df_strat_all["export_share_structural"].max()
            _sq_ylo_s = df_strat_all["cagr_export"].min()
            _sq_yhi_s = df_strat_all["cagr_export"].max()
            _add_quad_labels(fig_strat, _x_thr_s, 0,
                             [_sq_xlo_s, _sq_xhi_s], [_sq_ylo_s, _sq_yhi_s], [
                ("TR", t(STRINGS, "quad_stars_desc"),    "#10B981"),
                ("TL", t(STRINGS, "quad_emerging_desc"), "#60A5FA"),
                ("BR", t(STRINGS, "quad_cashcows_desc"), "#A78BFA"),
                ("BL", t(STRINGS, "quad_marginal2_desc"),"#6B7280"),
            ])
            fig_strat.update_layout(**PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_strategic_quadrant").format(country=country),
                xaxis=dict(title=t(STRINGS, "axis_export_share_pct"), autorange=True, gridcolor="#304870"),
                yaxis=dict(title=t(STRINGS, "axis_cagr_export"), autorange=True, gridcolor="#304870"),
                height=520,
            )
            st.plotly_chart(fig_strat, width='stretch')

            q_val = row_s.get("strategic_quadrant", "—")
            q_color = QUAD_COLORS.get(q_val, "#C4D5E8")
            st.markdown(f"""
            <div style="background:#1A2C48;border-left:4px solid {q_color};
                        border-radius:0 10px 10px 0;padding:.75rem 1rem;">
              <span style="color:{q_color};font-weight:600;">{country}</span>
              <span style="color:#A8C0D8;"> {t(STRINGS, "badge_strategic_quadrant")}</span>
              <span style="color:{q_color};font-weight:600;">{q_val}</span>
            </div>
            """, unsafe_allow_html=True)

        with quad_tabs[3]:
            # Meta profile positioning (F.4.6.5)
            META_COLORS = {
                "Global Leader":    "#10B981",
                "Established Power":"#60A5FA",
                "Rising Challenger":"#F59E0B",
                "Peripheral":       "#6B7280",
                "Volatile Emerging":"#EF4444",
            }
            _hl_meta = st.multiselect(
                t(STRINGS, "lbl_highlight_countries"),
                [c for c in COUNTRIES if c != country],
                default=[cc for cc in ["USA","China","Germany","Canada","Brazil","Qatar"]
                         if cc in COUNTRIES and cc != country],
                max_selections=6, key="meta_hl",
            )
            _meta_all = [country] + _hl_meta
            df_meta_all = df_structural.dropna(subset=["export_share_structural","structural_score"])
            df_meta_all["_label"] = df_meta_all["country_name"].apply(
                lambda x: x if x in _meta_all else "")
            df_meta_all["_size"] = df_meta_all["country_name"].apply(
                lambda x: 16 if x == country else 12 if x in _hl_meta else 8)
            df_meta_all["_sym"] = df_meta_all["country_name"].apply(
                lambda x: "star" if x == country else
                          "diamond" if x in _hl_meta else "circle")

            fig_meta_pos = go.Figure()
            for mp, mc in META_COLORS.items():
                df_mp = df_meta_all[df_meta_all["meta_profile"] == mp]
                if df_mp.empty:
                    continue
                fig_meta_pos.add_trace(go.Scatter(
                    x=df_mp["export_share_structural"],
                    y=df_mp["structural_score"],
                    mode="markers+text",
                    name=mp,
                    marker=dict(
                        color=mc,
                        size=df_mp["_size"],
                        opacity=[1.0 if n in _meta_all else 0.7
                                 for n in df_mp["country_name"]],
                        symbol=df_mp["_sym"],
                        line=dict(width=0.5, color="#304870"),
                    ),
                    text=df_mp["_label"],
                    textposition="top center",
                    textfont=dict(size=12, color="#EEF4FB"),
                    customdata=df_mp[["country_name","meta_profile",
                                     "export_share_structural","structural_score"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  + t(STRINGS, "hover_export_share") + ": %{customdata[2]:.3f}%<br>"
                                  + t(STRINGS, "hover_structural_score") + ": %{customdata[3]:.3f}<br>"
                                  + t(STRINGS, "hover_meta_profile") + ": %{customdata[1]}<extra></extra>",
                ))
            _p95_share_m = df_structural["export_share_structural"].quantile(0.97)
            _p95_score_m = df_structural["structural_score"].quantile(0.97)
            _p05_score_m = df_structural["structural_score"].quantile(0.03)
            # Meta profile: pure color-coded scatter, no threshold lines
            # (not a quadrant system — composite classification)
            fig_meta_pos.update_layout(**PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_meta_positioning").format(country=country),
                xaxis=dict(title=t(STRINGS, "axis_export_share_pct"), autorange=True, gridcolor="#304870"),
                yaxis=dict(title=t(STRINGS, "axis_structural_score"), autorange=True, gridcolor="#304870"),
                height=520,
            )
            # Distribution
            meta_counts = df_structural["meta_profile"].value_counts().reset_index()
            meta_counts.columns = ["Meta Profile","Count"]
            fig_metadist = go.Figure(go.Bar(
                x=meta_counts["Count"], y=meta_counts["Meta Profile"],
                orientation="h",
                marker_color=[META_COLORS.get(p,"#6B7280") for p in meta_counts["Meta Profile"]],
                text=meta_counts["Count"], textposition="outside", opacity=0.85,
            ))
            fig_metadist.update_layout(**PLOTLY_LAYOUT_COMPACT,
                paper_bgcolor="#1F3050", plot_bgcolor="#162035",
                xaxis=dict(title=t(STRINGS, "axis_countries"), gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"),
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                title=t(STRINGS, "title_meta_dist"),
                height=270, margin=dict(l=120,r=40,t=40,b=20))
            st.plotly_chart(fig_meta_pos, width='stretch')
            st.plotly_chart(fig_metadist, width='stretch')

            # Meta profile badge
            meta_val = row_s.get("meta_profile","—")
            meta_color = META_COLORS.get(meta_val, "#C4D5E8")
            st.markdown(f"""
            <div style="background:#1A2C48;border-left:4px solid {meta_color};
                        border-radius:0 10px 10px 0;padding:.75rem 1rem;">
              <span style="color:#A8C0D8;">{t(STRINGS, "badge_meta_profile").format(country=country)}</span>
              <span style="color:{meta_color};font-weight:600;">{meta_val}</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander(t(STRINGS, "expander_quad_defs")):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(t(STRINGS, "quad_defs_col1"))
            with col2:
                st.markdown(t(STRINGS, "quad_defs_col2"))


# ── TAB 5 : SCORING ──────────────────────────────────────────
with tab5:
    section_label(t(STRINGS, "sec_scoring"))

    if df_s.empty:
        st.warning(t(STRINGS, "warn_no_structural").format(country=country))
    else:
        row_s = df_s.iloc[0]

        # ── Scorecard ────────────────────────────────────────
        st.markdown(t(STRINGS, "md_profile_scorecard"))

        score_items = [
            (t(STRINGS, "score_size_profile"),        "size_profile",        "export_share_structural",  "%",   False),
            (t(STRINGS, "score_balance_profile"),     "balance_profile",     "coverage_ratio_structural", "×",  False),
            (t(STRINGS, "score_growth_profile"),      "growth_profile",      "cagr_export",               "%",  True),
            (t(STRINGS, "score_stability_profile"),   "stability_profile",   "volatility_export",         "σ",  False),
            (t(STRINGS, "score_consistency_profile"), "consistency_profile", "negfreq_export",            "",   True),
        ]

        cols_sc = st.columns(5)
        for col, (label, profile_col, value_col, unit, as_pct) in zip(cols_sc, score_items):
            profile_val = row_s.get(profile_col, "—")
            raw_val     = row_s.get(value_col, None)
            level       = LABEL_LEVELS.get(profile_col, {}).get(profile_val, 2)
            color       = LEVEL_COLORS.get(level, "#C4D5E8")
            if raw_val is None:
                val_str = "—"
            elif as_pct:
                val_str = f"{raw_val:.1%}"   # 0.4 → 40.0%
            else:
                val_str = f"{raw_val:.3f}{unit}"

            col.markdown(f"""
            <div style="background:#1F3050;border:1px solid #304870;border-top:3px solid {color};
                        border-radius:10px;padding:.75rem .8rem;text-align:center;">
              <div style="font-size:11px;color:#6890B0;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:4px;">{label}</div>
              <div style="font-size:18px;font-weight:700;color:{color};">{profile_val}</div>
              <div style="font-size:12px;color:#6890B0;margin-top:3px;">{val_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:.75rem;'></div>", unsafe_allow_html=True)

        # ── Combined score + profile ──────────────────────────
        struct_score   = row_s.get("structural_score", 0)
        struct_profile = row_s.get("structural_profile", "—")
        meta_profile   = row_s.get("meta_profile", "—")
        s_color        = PROFILE_COLORS.get(struct_profile, "#C4D5E8")

        st.markdown(f"""
        <div style="background:#1A2C48;border:1px solid #304870;border-left:4px solid {s_color};
                    border-radius:0 12px 12px 0;padding:.85rem 1.1rem;
                    display:flex;align-items:center;gap:24px;margin-bottom:1rem;">
          <div>
            <div style="font-size:11px;color:#6890B0;text-transform:uppercase;letter-spacing:.06em;">
              {t(STRINGS, "lbl_structural_score")}</div>
            <div style="font-size:26px;font-weight:700;color:{s_color};">{struct_score:.3f}</div>
          </div>
          <div>
            <div style="font-size:11px;color:#6890B0;text-transform:uppercase;letter-spacing:.06em;">
              {t(STRINGS, "lbl_structural_profile")}</div>
            <div style="font-size:18px;font-weight:600;color:{s_color};">{struct_profile}</div>
          </div>
          <div>
            <div style="font-size:11px;color:#6890B0;text-transform:uppercase;letter-spacing:.06em;">
              {t(STRINGS, "lbl_meta_profile")}</div>
            <div style="font-size:16px;font-weight:500;color:#C4D5E8;">{meta_profile}</div>
          </div>
          <div style="margin-left:auto;">
            <div style="font-size:11px;color:#6890B0;text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:.5rem;">{t(STRINGS, "lbl_quadrants")}</div>
            <div style="display:flex;flex-direction:column;gap:.3rem;">
              <div style="background:#162035;border-radius:6px;padding:.3rem .6rem;
                          display:flex;justify-content:space-between;gap:1rem;">
                <span style="font-size:11px;color:#6890B0;">{t(STRINGS, "lbl_static")}</span>
                <span style="font-size:12px;font-weight:600;
                             color:{QUAD_COLORS.get(row_s.get("static_quadrant",""),"#C4D5E8")};">
                  {row_s.get("static_quadrant","—")}</span>
              </div>
              <div style="background:#162035;border-radius:6px;padding:.3rem .6rem;
                          display:flex;justify-content:space-between;gap:1rem;">
                <span style="font-size:11px;color:#6890B0;">{t(STRINGS, "lbl_dynamic")}</span>
                <span style="font-size:12px;font-weight:600;
                             color:{QUAD_COLORS.get(row_s.get("dynamic_quadrant",""),"#C4D5E8")};">
                  {row_s.get("dynamic_quadrant","—")}</span>
              </div>
              <div style="background:#162035;border-radius:6px;padding:.3rem .6rem;
                          display:flex;justify-content:space-between;gap:1rem;">
                <span style="font-size:11px;color:#6890B0;">{t(STRINGS, "lbl_strategic")}</span>
                <span style="font-size:12px;font-weight:600;
                             color:{QUAD_COLORS.get(row_s.get("strategic_quadrant",""),"#C4D5E8")};">
                  {row_s.get("strategic_quadrant","—")}</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Radar chart ───────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            section_label(t(STRINGS, "sec_radar"))

            # Compare selector
            _opts_radar = [c for c in COUNTRIES if c != country]
            _def_radar = [c for c in ["Germany","Brazil","Canada"]
                          if c in _opts_radar]
            compare_countries = st.multiselect(
                t(STRINGS, "lbl_compare_with"),
                _opts_radar,
                default=_def_radar,
                max_selections=3,
                key="radar_compare",
            )

            all_radar = [country] + compare_countries
            radar_colors = [COLORS["export"], COLORS["import"],
                            COLORS["balance"], COLORS["coverage"]]

            fig_radar = go.Figure()
            for i, c_name in enumerate(all_radar):
                row_r = df_structural[df_structural["country_name"] == c_name]
                if row_r.empty:
                    continue
                row_r = row_r.iloc[0]
                vals = [RADAR_LEVELS[d].get(row_r.get(d, ""), 0) for d in DIMS]
                vals_closed = vals + [vals[0]]
                labels_closed = DIM_LABELS + [DIM_LABELS[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=labels_closed,
                    fill="toself",
                    name=c_name,
                    line=dict(color=radar_colors[i % len(radar_colors)], width=2),
                    fillcolor=f"rgba({','.join(str(int(int(radar_colors[i%4].lstrip('#')[j:j+2],16))) for j in (0,2,4))},0.1)",
                    opacity=0.85,
                ))

            fig_radar.update_layout(
                paper_bgcolor="#1F3050",
                plot_bgcolor="#1F3050",
                font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
                polar=dict(
                    bgcolor="#162035",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#304870", linecolor="#304870"),
                    angularaxis=dict(gridcolor="#304870", linecolor="#304870"),
                ),
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_radar, width='stretch')

        with col2:
            section_label(t(STRINGS, "sec_structural_heatmap"))

            _hm_col1, _hm_col2 = st.columns([1, 2])
            with _hm_col1:
                HM_VIEW_OPTS = ["Top performers", "Bottom performers"]
                HM_VIEW_LABELS = {
                    "Top performers": t(STRINGS, "opt_top_performers"),
                    "Bottom performers": t(STRINGS, "opt_bottom_performers"),
                }
                hm_view = st.radio(t(STRINGS, "lbl_view"), HM_VIEW_OPTS,
                                   format_func=lambda x: HM_VIEW_LABELS[x],
                                   key="heatmap_view")
                top_n = st.slider(t(STRINGS, "lbl_n_countries"), 5, 15, 8, key="heatmap_n")

            if "Top" in hm_view:
                df_top = df_structural.nlargest(top_n, "structural_score").copy()
            else:
                df_top = df_structural.nsmallest(top_n, "structural_score").copy()

            # Add selected country if not already included
            if country not in df_top["country_name"].values:
                row_c = df_structural[df_structural["country_name"] == country]
                df_top = pd.concat([df_top, row_c])

            df_top = df_top.sort_values("structural_score", ascending=True)

            # Build heatmap matrix
            hm_data = []
            for _, row_h in df_top.iterrows():
                hm_data.append([
                    LABEL_LEVELS[d].get(row_h.get(d, ""), 2)
                    for d in DIMS
                ])

            with _hm_col2:
             fig_hm = go.Figure(go.Heatmap(
                z=hm_data,
                x=DIM_LABELS,
                y=df_top["country_name"].tolist(),
                colorscale=[[0, "#10B981"], [0.33, "#60A5FA"], [0.66, "#F59E0B"], [1, "#EF4444"]],
                zmin=1, zmax=4,
                showscale=True,
                text=[[df_top.iloc[i].get(DIMS[j], "") for j in range(len(DIMS))]
                       for i in range(len(df_top))],
                texttemplate="%{text}",
                textfont=dict(size=10, color="#EEF4FB"),
                hoverongaps=False,
            ))

            # Highlight selected country
            selected_idx = df_top["country_name"].tolist().index(country) \
                if country in df_top["country_name"].tolist() else None
            if selected_idx is not None:
                fig_hm.add_shape(
                    type="rect",
                    x0=-0.5, x1=len(DIMS)-0.5,
                    y0=selected_idx-0.5, y1=selected_idx+0.5,
                    line=dict(color="#EEF4FB", width=2),
                )

            fig_hm.update_layout(
                paper_bgcolor="#1F3050",
                plot_bgcolor="#162035",
                font=dict(family="IBM Plex Sans", color="#C4D5E8", size=11),
                margin=dict(l=120, r=20, t=20, b=40),
                height=420,
            )
            st.plotly_chart(fig_hm, width='stretch')

        with st.expander(t(STRINGS, "expander_scoring_methodology")):
            st.markdown(t(STRINGS, "md_scoring_methodology"))