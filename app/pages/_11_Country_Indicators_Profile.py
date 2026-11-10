"""
Indicators Analysis — I3 · Country Indicator Profile
G · H · N — Time series · comparison · multi-score
"""

import sys
import warnings
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding
from utils.i18n import t

inject_styles()

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════
# STRINGS — i18n
# ════════════════════════════════════════════════════════════════

STRINGS = {
    "en": {
        "eyebrow": "Indicators Analysis · Page 3 of 4",
        "title": "Country Indicator Profile",
        "subtitle": "G · H · N — Time series · comparison · multi-score",
        "select_country": "Select country",
        "kpi_gdp_capita": "GDP/capita",
        "kpi_gdp_growth": "GDP Growth",
        "kpi_unemployment": "Unemployment",
        "kpi_inflation": "Inflation",
        "kpi_trade_pct_gdp": "Trade % GDP",
        "tab_time_series": "📈 G · Time Series",
        "tab_comparison": "🔄 H · Comparison",
        "tab_multiscore": "🏅 N · Multi-Score",
        "sec_country_time_series": "G · Country Time Series — All Indicators",
        "indicators_label": "Indicators",
        "view_label": "View",
        "break_threshold_slider": "Break threshold",
        "title_ts_chart": "{country} — {metric} by Indicator",
        "caption_mixed_units": (
            "💡 Indicators with different units (%, USD…) share the same axis. "
            "Switch to **Index-100** or **Z-score** for a normalized multi-indicator view."
        ),
        "sec_indicator_dynamics": "G.4 · Indicator Dynamics — CAGR & Volatility",
        "info_no_growth_country": "No growth data available for this country.",
        "trace_cagr": "CAGR",
        "trace_volatility": "Volatility",
        "title_cagr_by_indicator": "{country} — CAGR by Indicator",
        "yaxis_cagr": "CAGR",
        "yaxis_volatility": "Volatility",
        "caption_cagr_undefined": (
            "⚠️ CAGR undefined for: **{names}** — "
            "values change sign over the period (no real-valued compound growth rate). "
            "Volatility is still shown."
        ),
        "sec_zscore_heatmap": "G.5 · Z-Score Heatmap — All Indicators × Years",
        "title_zscore_heatmap": "{country} — Z-Score Heatmap (green = above world mean)",
        "year_label": "Year",
        "sec_cross_country_comparison": "H · Cross-Country Comparison",
        "indicator_label": "Indicator",
        "compare_with": "Compare with",
        "metric_label": "Metric",
        "annot_base2015": "Base 2015",
        "annot_world_mean": "World mean",
        "title_comp_chart": "{ind} — {metric}",
        "snapshot_year": "Snapshot year",
        "title_snapshot": "{metric} Snapshot — {year}",
        "sec_cagr_vs_world": "H.3 · Country vs World — CAGR per Indicator",
        "trace_world": "World",
        "title_cagr_vs_world": "CAGR — {country} vs World",
        "title_outperformance": "Outperformance vs World (CAGR delta)",
        "yaxis_cagr_delta": "CAGR Delta",
        "info_growth_unavailable": "Growth data not available.",
        "sec_multiscore": "N · Multi-Indicator Scoring",
        "warn_no_multiscore": "No multi-score data for {country}.",
        "expander_score_defs": "📐 Score definitions, formulas & weights",
        "score_defs_intro_md": """
**Double reading** — each pillar is scored twice:
- **Value Position**: weighted average of normalized indicator *values* → current structural level
- **Growth Momentum**: weighted average of normalized *growth scores* → directional trajectory

Each score is min-max normalized to **0–100** across all countries. Rank 1 = best.
""",
        "weights_economic_health_md": """
**Economic Health**
| Indicator | Weight |
|---|---|
| GDP per capita growth | +0.25 |
| Unemployment rate | −0.20 |
| Inflation (CPI) | −0.20 |
| Current account | +0.15 |
| Real interest rate | +0.10 |
| Gov. expenditure | +0.10 |
""",
        "weights_performance_md": """
**Performance**
| Indicator | Weight |
|---|---|
| GDP growth | +0.30 |
| Unemployment rate | −0.20 |
| Inflation (CPI) | −0.15 |
| Current account | +0.15 |
| Trade balance | +0.10 |
| Real interest rate | +0.10 |
""",
        "weights_competitiveness_md": """
**Competitiveness**
| Indicator | Weight |
|---|---|
| Trade % GDP | +0.30 |
| Current account | +0.25 |
| GDP per capita | +0.20 |
| Inflation (CPI) | −0.15 |
| Unemployment | −0.10 |
""",
        "quadrant_interpretation_md": """
---
**Quadrant interpretation** (scatter):
- **Top-right**: high value *and* high growth → strong & accelerating
- **Top-left**: low value, high growth → catching up
- **Bottom-right**: high value, low growth → established but slowing
- **Bottom-left**: low value, low growth → structural challenge

**Rank Divergence** — two complementary views:
- *Intra-dimension* (dumbbell chart): gap between Value Rank and Growth Rank **within** the same pillar
- *Inter-dimension* (KPI cards): spread of ranks **across** the 3 pillars (from `rank_div_value / rank_div_growth`)
""",
        "banner_value_position_title": "▸ Value Position",
        "banner_value_position_sub": "Current structural level — where the country stands today",
        "rank_hash": "Rank #{rank}",
        "banner_growth_momentum_title": "▸ Growth Momentum",
        "banner_growth_momentum_sub": "Directional score — where the country is heading",
        "radar_value_position": "Value Position",
        "radar_growth_momentum": "Growth Momentum",
        "title_score_radar": "{country} — Score Radar",
        "sec_global_ranking": "Global Ranking — All Countries",
        "highlight_countries": "Highlight countries",
        "dim_value_suffix": "{dim} · Value",
        "dim_growth_suffix": "{dim} · Growth",
        "x_axis_label": "X axis",
        "y_axis_label": "Y axis",
        "warn_select_different_axes": "Select different axes.",
        "title_scatter_axes": "{x} × {y}",
        "sec_rank_divergence": "Rank Divergence — Value vs Growth",
        "trace_value_rank": "Value Rank",
        "trace_growth_rank": "Growth Rank",
        "title_dumbbell": "Intra-dimension: Value Rank vs Growth Rank (left = better)",
        "xaxis_rank": "Rank (1 = best)",
        "spread_value_label": "Inter-Dim Spread · Value",
        "spread_value_caption": "rank-point spread across 3 pillars",
        "spread_growth_label": "Inter-Dim Spread · Growth",
        "spread_growth_caption": "rank-point spread across 3 pillars",
    },
    "fr": {
        "eyebrow": "Analyse des indicateurs · Page 3 sur 4",
        "title": "Profil pays — Indicateurs",
        "subtitle": "G · H · N — Série temporelle · comparaison · multi-score",
        "select_country": "Sélectionner un pays",
        "kpi_gdp_capita": "PIB/habitant",
        "kpi_gdp_growth": "Croissance PIB",
        "kpi_unemployment": "Chômage",
        "kpi_inflation": "Inflation",
        "kpi_trade_pct_gdp": "Commerce % PIB",
        "tab_time_series": "📈 G · Série temporelle",
        "tab_comparison": "🔄 H · Comparaison",
        "tab_multiscore": "🏅 N · Multi-score",
        "sec_country_time_series": "G · Série temporelle du pays — Tous indicateurs",
        "indicators_label": "Indicateurs",
        "view_label": "Vue",
        "break_threshold_slider": "Seuil de rupture",
        "title_ts_chart": "{country} — {metric} par indicateur",
        "caption_mixed_units": (
            "💡 Les indicateurs avec des unités différentes (%, USD…) partagent le même axe. "
            "Basculer sur **Index-100** ou **Z-score** pour une vue multi-indicateur normalisée."
        ),
        "sec_indicator_dynamics": "G.4 · Dynamique des indicateurs — CAGR & volatilité",
        "info_no_growth_country": "Aucune donnée de croissance disponible pour ce pays.",
        "trace_cagr": "CAGR",
        "trace_volatility": "Volatilité",
        "title_cagr_by_indicator": "{country} — CAGR par indicateur",
        "yaxis_cagr": "CAGR",
        "yaxis_volatility": "Volatilité",
        "caption_cagr_undefined": (
            "⚠️ CAGR non défini pour : **{names}** — "
            "les valeurs changent de signe sur la période (pas de taux de croissance composé réel). "
            "La volatilité reste affichée."
        ),
        "sec_zscore_heatmap": "G.5 · Heatmap des Z-scores — Tous indicateurs × années",
        "title_zscore_heatmap": "{country} — Heatmap des Z-scores (vert = au-dessus de la moyenne mondiale)",
        "year_label": "Année",
        "sec_cross_country_comparison": "H · Comparaison entre pays",
        "indicator_label": "Indicateur",
        "compare_with": "Comparer avec",
        "metric_label": "Indicateur",
        "annot_base2015": "Base 2015",
        "annot_world_mean": "Moyenne mondiale",
        "title_comp_chart": "{ind} — {metric}",
        "snapshot_year": "Année de référence",
        "title_snapshot": "Instantané {metric} — {year}",
        "sec_cagr_vs_world": "H.3 · Pays vs World — CAGR par indicateur",
        "trace_world": "World",
        "title_cagr_vs_world": "CAGR — {country} vs World",
        "title_outperformance": "Surperformance vs World (écart de CAGR)",
        "yaxis_cagr_delta": "Écart de CAGR",
        "info_growth_unavailable": "Données de croissance non disponibles.",
        "sec_multiscore": "N · Scoring multi-indicateurs",
        "warn_no_multiscore": "Aucune donnée de multi-score pour {country}.",
        "expander_score_defs": "📐 Définitions des scores, formules & pondérations",
        "score_defs_intro_md": """
**Double lecture** — chaque pilier est noté deux fois :
- **Position de valeur** : moyenne pondérée des *valeurs* normalisées des indicateurs → niveau structurel actuel
- **Dynamique de croissance** : moyenne pondérée des *scores de croissance* normalisés → trajectoire directionnelle

Chaque score est normalisé min-max entre **0 et 100** sur l'ensemble des pays. Rang 1 = le meilleur.
""",
        "weights_economic_health_md": """
**Santé économique**
| Indicateur | Pondération |
|---|---|
| Croissance du PIB par habitant | +0,25 |
| Taux de chômage | −0,20 |
| Inflation (IPC) | −0,20 |
| Compte courant | +0,15 |
| Taux d'intérêt réel | +0,10 |
| Dépenses publiques | +0,10 |
""",
        "weights_performance_md": """
**Performance**
| Indicateur | Pondération |
|---|---|
| Croissance du PIB | +0,30 |
| Taux de chômage | −0,20 |
| Inflation (IPC) | −0,15 |
| Compte courant | +0,15 |
| Balance commerciale | +0,10 |
| Taux d'intérêt réel | +0,10 |
""",
        "weights_competitiveness_md": """
**Compétitivité**
| Indicateur | Pondération |
|---|---|
| Commerce % PIB | +0,30 |
| Compte courant | +0,25 |
| PIB par habitant | +0,20 |
| Inflation (IPC) | −0,15 |
| Chômage | −0,10 |
""",
        "quadrant_interpretation_md": """
---
**Interprétation des quadrants** (nuage de points) :
- **Haut-droite** : valeur élevée *et* croissance élevée → solide et en accélération
- **Haut-gauche** : valeur faible, croissance élevée → en rattrapage
- **Bas-droite** : valeur élevée, croissance faible → établi mais en ralentissement
- **Bas-gauche** : valeur faible, croissance faible → défi structurel

**Divergence de rang** — deux vues complémentaires :
- *Intra-dimension* (graphique en haltère) : écart entre le rang de valeur et le rang de croissance **au sein** du même pilier
- *Inter-dimension* (cartes KPI) : dispersion des rangs **entre** les 3 piliers (d'après `rank_div_value / rank_div_growth`)
""",
        "banner_value_position_title": "▸ Position de valeur",
        "banner_value_position_sub": "Niveau structurel actuel — où se situe le pays aujourd'hui",
        "rank_hash": "Rang #{rank}",
        "banner_growth_momentum_title": "▸ Dynamique de croissance",
        "banner_growth_momentum_sub": "Score directionnel — où se dirige le pays",
        "radar_value_position": "Position de valeur",
        "radar_growth_momentum": "Dynamique de croissance",
        "title_score_radar": "{country} — Radar de scores",
        "sec_global_ranking": "Classement mondial — Tous les pays",
        "highlight_countries": "Pays à mettre en évidence",
        "dim_value_suffix": "{dim} · Valeur",
        "dim_growth_suffix": "{dim} · Croissance",
        "x_axis_label": "Axe X",
        "y_axis_label": "Axe Y",
        "warn_select_different_axes": "Sélectionner des axes différents.",
        "title_scatter_axes": "{x} × {y}",
        "sec_rank_divergence": "Divergence de rang — Valeur vs croissance",
        "trace_value_rank": "Rang de valeur",
        "trace_growth_rank": "Rang de croissance",
        "title_dumbbell": "Intra-dimension : rang de valeur vs rang de croissance (gauche = meilleur)",
        "xaxis_rank": "Rang (1 = meilleur)",
        "spread_value_label": "Dispersion inter-dim · Valeur",
        "spread_value_caption": "dispersion en rangs sur les 3 piliers",
        "spread_growth_label": "Dispersion inter-dim · Croissance",
        "spread_growth_caption": "dispersion en rangs sur les 3 piliers",
    },
}

# ════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_ind    = pd.read_csv(IND_FILES / "df_indicators.csv")
    df_multi  = pd.read_csv(IND_FILES / "multi_df.csv")
    df_cl     = pd.read_csv(IND_FILES / "cluster_df_multi.csv")
    df_growth = pd.read_csv(IND_FILES / "growth_df.csv")
    df_world  = df_ind[df_ind["entity_type"] == "ENTITY"].copy()
    df_ind    = df_ind[(df_ind["entity_type"] == "COUNTRY") &
                       df_ind["country_name"].notna()].copy()
    return df_ind, df_multi, df_cl, df_growth, df_world

df_ind, df_multi, df_cl, df_growth, df_world = load_data()

COUNTRIES  = sorted(df_ind["country_name"].dropna().unique().tolist())
INDICATORS = sorted(df_ind["indicator_name"].dropna().unique().tolist())
YEARS      = sorted(df_ind["year"].unique())
DIMS       = ["Economic Health", "Performance", "Competitiveness"]

DIM_COLORS = {
    "Economic Health": "#10B981",
    "Performance":     "#60A5FA",
    "Competitiveness": "#F59E0B",
}

CLUSTER_COLORS = {
    0: "#10B981", 1: "#60A5FA", 2: "#F59E0B", 3: "#EF4444",
    4: "#A78BFA", 5: "#F472B6", 6: "#34D399", 7: "#F87171",
}

_CL_FEATURES = ["z_cagr", "volatility", "share_under_pressure",
                 "shock_frequency", "mean_abs_z", "asymmetry_index"]

PALETTE  = ["#F59E0B", "#60A5FA", "#10B981", "#EF4444", "#A78BFA", "#F472B6"]
PALETTE2 = ["#EEF4FB", "#60A5FA", "#10B981", "#F59E0B",
            "#EF4444", "#A78BFA", "#F472B6", "#34D399"]

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

_pl = lambda **kw: {**PLOTLY_LAYOUT, **kw}

# ════════════════════════════════════════════════════════════════
# CACHED FUNCTIONS
# ════════════════════════════════════════════════════════════════

@st.cache_data
def _best_k(cache_key: str, _X: np.ndarray, k_range: tuple = (2, 8)):
    sil = {}
    for k in range(k_range[0], k_range[1] + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(_X)
        sil[k] = silhouette_score(_X, lbl)
    best = max(sil, key=sil.get)
    return sil, best

@st.cache_data
def _run_kmeans(cache_key: str, _X: np.ndarray, k: int):
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(_X)

@st.cache_data
def _hw_forecast(years: np.ndarray, values: np.ndarray, cache_key: str,
                 periods: int = 5, n_bootstrap: int = 300):
    years  = years.astype(float)
    values = values.astype(float)
    model  = ExponentialSmoothing(
        values, trend="add", seasonal=None,
        initialization_method="estimated").fit(optimized=True)
    fc_vals = model.forecast(periods)
    fc_yrs  = np.arange(int(years[-1]) + 1,
                        int(years[-1]) + 1 + periods, dtype=float)
    fitted  = model.fittedvalues
    resid   = values - fitted
    rng     = np.random.default_rng(42)
    boot    = np.array([fc_vals + rng.choice(resid, size=periods, replace=True)
                        for _ in range(n_bootstrap)])
    ci_lo   = np.percentile(boot, 2.5,  axis=0)
    ci_hi   = np.percentile(boot, 97.5, axis=0)
    ts_size = min(3, len(values) - 4)
    if ts_size >= 2:
        ev   = ExponentialSmoothing(
            values[:-ts_size], trend="add", seasonal=None,
            initialization_method="estimated").fit(optimized=True)
        yp   = ev.forecast(ts_size)
        yt   = values[-ts_size:]
        mae  = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
        yrs_ev = years[-ts_size:]
    else:
        yp = yt = yrs_ev = np.array([])
        mae = rmse = mape = float("nan")
    return dict(years=years, values=values, fc_yrs=fc_yrs, fc_vals=fc_vals,
                ci_lo=ci_lo, ci_hi=ci_hi, fitted=fitted, resid=resid,
                mae=mae, rmse=rmse, mape=mape,
                yrs_ev=yrs_ev, y_true_ev=yt, y_pred_ev=yp)

@st.cache_data
def _run_regression(target: str, predictors: tuple, cache_key: str,
                    _df: pd.DataFrame):
    all_inds = [target] + list(predictors)
    data  = _df[_df["indicator_name"].isin(all_inds)].copy()
    pivot = (data.groupby(["country_name", "indicator_name"])["value"]
             .mean().unstack("indicator_name"))
    pivot = pivot[pivot.notna().mean(axis=1) >= 0.70].fillna(pivot.median())
    if target not in pivot.columns:
        return None
    y = pivot[target].dropna()
    feat_cols = [c for c in predictors if c in pivot.columns]
    X = pivot.loc[y.index, feat_cols]
    if len(X) < 20 or len(feat_cols) < 2:
        return None
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y.values, test_size=0.25, random_state=42)
    models = {
        "Linear":            LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=0.1),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        cv = cross_val_score(m, X_sc, y.values, cv=5, scoring="r2").mean()
        results[name] = dict(R2_test=r2_score(y_te, yp), R2_cv=cv,
                             MAE=mean_absolute_error(y_te, yp),
                             RMSE=float(np.sqrt(mean_squared_error(y_te, yp))),
                             model=m)
    best = max(results, key=lambda k: results[k]["R2_cv"])
    bm   = results[best]["model"]
    if hasattr(bm, "feature_importances_"):
        imp       = bm.feature_importances_
        imp_label = "Feature Importance"
    elif hasattr(bm, "coef_"):
        imp       = bm.coef_
        imp_label = "Coefficients (standardized)"
    else:
        imp       = np.zeros(len(feat_cols))
        imp_label = "Importance"
    # Full-sample predictions for structural benchmark chart
    bm.fit(X_sc, y.values)
    y_pred_all = bm.predict(X_sc)
    return dict(results=results, best_name=best, feat_cols=feat_cols,
                importances=imp, imp_label=imp_label, n_obs=len(X),
                countries=y.index.tolist(),
                y_actual=y.values, y_pred_all=y_pred_all)

@st.cache_data
def _run_lag_regression(target: str, predictors: tuple, cache_key: str,
                        _df: pd.DataFrame):
    """
    Autoregressive distributed lag model: [target(t-1), predictors(t-1)] → target(t).
    Trained on the full country×year panel — gives genuine time-series predictions.
    """
    all_inds = list({target} | set(predictors))
    data = _df[_df["indicator_name"].isin(all_inds)].copy()
    piv  = (data.pivot_table(
                index=["country_name", "year"],
                columns="indicator_name", values="value")
            .reset_index()
            .sort_values(["country_name", "year"]))

    feat_cols = [c for c in [target] + list(predictors) if c in piv.columns]

    rows = []
    for _, grp in piv.groupby("country_name", sort=False):
        grp = grp.sort_values("year")
        for i in range(1, len(grp)):
            t_row   = grp.iloc[i]
            tm1_row = grp.iloc[i - 1]
            if pd.isna(t_row[target]):
                continue
            row = {f: tm1_row.get(f, np.nan) for f in feat_cols}
            row["_y"]           = t_row[target]
            row["_year"]        = int(t_row["year"])
            row["_country"]     = grp.iloc[i]["country_name"]
            rows.append(row)

    if len(rows) < 20:
        return None
    df_lag = pd.DataFrame(rows).dropna(subset=feat_cols + ["_y"])
    if len(df_lag) < 20:
        return None

    X = df_lag[feat_cols].values
    y = df_lag["_y"].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.25, random_state=42)

    models = {
        "Linear":            LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        cv = cross_val_score(m, X_sc, y, cv=5, scoring="r2").mean()
        results[name] = dict(R2_test=r2_score(y_te, yp), R2_cv=cv,
                             MAE=mean_absolute_error(y_te, yp),
                             RMSE=float(np.sqrt(mean_squared_error(y_te, yp))),
                             model=m)
    best = max(results, key=lambda k: results[k]["R2_cv"])
    bm   = results[best]["model"]
    bm.fit(X_sc, y)          # refit on all data for production use

    if hasattr(bm, "feature_importances_"):
        imp, imp_label = bm.feature_importances_, "Feature Importance"
    elif hasattr(bm, "coef_"):
        imp, imp_label = bm.coef_, "Coefficients (standardized)"
    else:
        imp, imp_label = np.zeros(len(feat_cols)), "Importance"

    # Historical fitted values per country (for chart overlay)
    hist_fit = {}
    for c_name, grp in df_lag.groupby("_country"):
        X_c  = scaler.transform(grp[feat_cols].values)
        hist_fit[c_name] = dict(years=grp["_year"].values,
                                actual=grp["_y"].values,
                                fitted=bm.predict(X_c))

    return dict(results=results, best_name=best, feat_cols=feat_cols,
                importances=imp, imp_label=imp_label, n_obs=len(df_lag),
                model=bm, scaler=scaler, hist_fit=hist_fit)


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "subtitle"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR + KPIs
# ════════════════════════════════════════════════════════════════

col_sel, col_kpi = st.columns([1, 3])
with col_sel:
    default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox(t(STRINGS, "select_country"), COUNTRIES,
                           index=default_idx, key="i3_country")

row_m  = df_multi[df_multi["country_name"] == country]

_KPI_FACTS = [
    ("GDP/capita",   "GDP per capita (current US$)",
     "value",      "#10B981"),
    ("GDP Growth",   "GDP (current US$)",
     "yoy_growth", "#60A5FA"),
    ("Unemployment", "Unemployment, total (% of total labor force) (modeled ILO estimate)",
     "value",      "#F59E0B"),
    ("Inflation",    "Inflation, consumer prices (annual %)",
     "value",      "#EF4444"),
    ("Trade % GDP",  "Trade (% of GDP)",
     "value",      "#A78BFA"),
]

_KPI_LABEL_KEYS = {
    "GDP/capita":   "kpi_gdp_capita",
    "GDP Growth":   "kpi_gdp_growth",
    "Unemployment": "kpi_unemployment",
    "Inflation":    "kpi_inflation",
    "Trade % GDP":  "kpi_trade_pct_gdp",
}

def _latest_val(ind_name, col="value"):
    sub = (df_ind[(df_ind["country_name"] == country) &
                  (df_ind["indicator_name"] == ind_name)]
           .dropna(subset=[col]).sort_values("year"))
    if sub.empty:
        return None, None
    row = sub.iloc[-1]
    return row[col], int(row["year"])

def _fmt_kpi(label, val, yr, col):
    disp_label = t(STRINGS, _KPI_LABEL_KEYS.get(label, label))
    if val is None:
        disp = "—"
    elif label == "GDP/capita":
        from utils.styles import format_value
        disp = format_value(val)
    elif label == "GDP Growth":
        disp = f"{val*100:+.1f}%"
    else:
        disp = f"{val:+.1f}%" if val < 0 else f"{val:.1f}%"
    yr_str = f"({yr})" if yr else ""
    return (
        f'<div class="kpi-card" style="border-top-color:{col}">'
        f'<div class="kpi-label">{disp_label}</div>'
        f'<div class="kpi-value" style="color:{col};font-size:22px;">{disp}</div>'
        f'<div style="font-size:10px;color:#6890B0;margin-top:3px;">{yr_str}</div>'
        f'</div>'
    )

with col_kpi:
    kpi_cols = st.columns(5)
    for kpi_col, (label, ind_name, data_col, color) in zip(kpi_cols, _KPI_FACTS):
        val, yr = _latest_val(ind_name, data_col)
        kpi_col.markdown(_fmt_kpi(label, val, yr, color), unsafe_allow_html=True)

thin_divider()

# default indicator used across tabs
_default_ind = next(
    (n for n in INDICATORS if "GDP per capita" in n and "current" in n),
    INDICATORS[0])

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    t(STRINGS, "tab_time_series"),
    t(STRINGS, "tab_comparison"),
    t(STRINGS, "tab_multiscore"),
])


# ── TAB 1 : G · TIME SERIES ─────────────────────────────────────────────────────
with tab1:
    section_label(t(STRINGS, "sec_country_time_series"))

    df_c = df_ind[df_ind["country_name"] == country].copy()

    inds_sel = st.multiselect(
        t(STRINGS, "indicators_label"), options=INDICATORS,
        default=[_default_ind], max_selections=6, key="ts_indicators")

    col1, col2 = st.columns([1, 3])
    with col1:
        metric_ts = st.radio(
            t(STRINGS, "view_label"), ["Value", "YoY Growth %", "Index-100", "Z-score"],
            key="ts_metric")
        if len(inds_sel) == 1:
            _brk_thr = st.slider(
                t(STRINGS, "break_threshold_slider"), min_value=0.03, max_value=0.20,
                value=0.07, step=0.01, format="%.2f", key="g_break_thr")
        else:
            _brk_thr = st.session_state.get("g_break_thr", 0.07)

    metric_col = {"Value": "value", "YoY Growth %": "yoy_growth",
                  "Index-100": "index_100", "Z-score": "z_score"}[metric_ts]
    _single = len(inds_sel) == 1

    with col2:
        fig_ts = go.Figure()
        for i, ind in enumerate(inds_sel):
            df_ci = df_c[df_c["indicator_name"] == ind].sort_values("year")
            if df_ci.empty:
                continue
            short  = ind[:35] + "…" if len(ind) > 35 else ind
            color  = PALETTE[i % len(PALETTE)]
            yvals  = df_ci[metric_col].values
            xvals  = df_ci["year"].values
            fig_ts.add_trace(go.Scatter(
                x=xvals, y=yvals, mode="lines+markers", name=short,
                line=dict(color=color, width=2), marker=dict(size=6),
            ))
            if _single and len(yvals) >= 4:
                _diff_raw   = np.diff(yvals.astype(float))
                _denom      = np.abs(yvals[:-1].astype(float)) + 1e-9
                _pct_signed = _diff_raw / _denom
                for brk_i, pct_s in enumerate(_pct_signed):
                    if abs(pct_s) > _brk_thr:
                        fig_ts.add_vline(
                            x=float(xvals[brk_i + 1]),
                            line_dash="dot", line_color="#EF4444",
                            annotation_text=f"Δ{pct_s:+.0%}",
                            annotation_font=dict(color="#EF4444", size=10),
                        )

        if metric_ts == "YoY Growth %":
            fig_ts.add_hline(y=0, line_color="#EEF4FB",
                             line_width=1, line_dash="dash")
        if metric_ts == "Index-100":
            fig_ts.add_hline(y=100, line_color="#EEF4FB",
                             line_width=1, line_dash="dash",
                             annotation_text=t(STRINGS, "annot_base2015"))
        fig_ts.update_layout(**_pl(
            title=t(STRINGS, "title_ts_chart").format(country=country, metric=metric_ts),
            yaxis_title=metric_ts, height=450))
        st.plotly_chart(fig_ts, width='stretch')
        if metric_ts == "Value" and len(inds_sel) > 1:
            st.caption(t(STRINGS, "caption_mixed_units"))

    # G.4 Indicator Dynamics
    section_label(t(STRINGS, "sec_indicator_dynamics"))
    df_grw_c = df_growth[
        (df_growth["country_name"] == country) &
        (df_growth["entity_type"] == "COUNTRY")
    ].copy()

    if df_grw_c.empty:
        st.info(t(STRINGS, "info_no_growth_country"))
    else:
        _avail = df_grw_c["indicator_name"].unique().tolist()
        _dyn   = [n for n in inds_sel if n in _avail] or _avail
        df_dyn = df_grw_c[df_grw_c["indicator_name"].isin(_dyn)].copy()
        df_dyn["_short"] = df_dyn["indicator_name"].apply(
            lambda n: n[:38] + "…" if len(n) > 38 else n)
        df_dyn = df_dyn.sort_values("cagr", ascending=False)

        _df_cagr_ok  = df_dyn.dropna(subset=["cagr"])
        _df_cagr_nan = df_dyn[df_dyn["cagr"].isna()]

        fig_dyn = go.Figure()
        if not _df_cagr_ok.empty:
            fig_dyn.add_trace(go.Bar(
                x=_df_cagr_ok["_short"], y=_df_cagr_ok["cagr"], name=t(STRINGS, "trace_cagr"),
                marker_color=["#10B981" if v >= 0 else "#EF4444"
                              for v in _df_cagr_ok["cagr"]],
                opacity=0.85,
                text=_df_cagr_ok["cagr"].apply(lambda v: f"{v:+.2%}"),
                textposition="outside",
            ))
        if "volatility" in df_dyn.columns:
            _df_vol_ok = df_dyn.dropna(subset=["volatility"])
            fig_dyn.add_trace(go.Scatter(
                x=_df_vol_ok["_short"], y=_df_vol_ok["volatility"],
                name=t(STRINGS, "trace_volatility"), mode="markers", yaxis="y2",
                marker=dict(color="#F59E0B", size=10, symbol="diamond"),
            ))
        fig_dyn.update_layout(**_pl(
            title=t(STRINGS, "title_cagr_by_indicator").format(country=country),
            yaxis_title=t(STRINGS, "yaxis_cagr"),
            yaxis2=dict(title=t(STRINGS, "yaxis_volatility"), overlaying="y", side="right",
                        showgrid=False, color="#F59E0B"),
            height=460, xaxis_tickangle=-30,
        ))
        st.plotly_chart(fig_dyn, width='stretch')
        if not _df_cagr_nan.empty:
            _nan_names = " · ".join(_df_cagr_nan["indicator_name"].tolist())
            st.caption(t(STRINGS, "caption_cagr_undefined").format(names=_nan_names))

    # G.5 Z-Score Heatmap
    section_label(t(STRINGS, "sec_zscore_heatmap"))
    df_hm = df_c.pivot_table(
        index="indicator_name", columns="year", values="z_score")
    df_hm.index = [n[:45] + "…" if len(n) > 45 else n for n in df_hm.index]
    fig_hm = go.Figure(go.Heatmap(
        z=df_hm.values,
        x=[str(y) for y in df_hm.columns],
        y=df_hm.index.tolist(),
        colorscale="RdYlGn", zmid=0,
        text=np.round(df_hm.values, 2),
        texttemplate="%{text}", textfont=dict(size=10),
        showscale=True, hoverongaps=False,
    ))
    fig_hm.update_layout(**_pl(
        title=t(STRINGS, "title_zscore_heatmap").format(country=country),
        xaxis_title=t(STRINGS, "year_label"),
        height=max(380, len(df_hm) * 32 + 80),
    ))
    st.plotly_chart(fig_hm, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · H · COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

# ── TAB 2 : H · COMPARISON ──────────────────────────────────────────────────────
with tab2:
    section_label(t(STRINGS, "sec_cross_country_comparison"))

    indicator_comp = st.selectbox(
        t(STRINGS, "indicator_label"), INDICATORS, index=0, key="comp_ind")
    peers = st.multiselect(
        t(STRINGS, "compare_with"),
        [c for c in COUNTRIES if c != country],
        default=[c for c in ["China", "Germany", "USA", "Japan"]
                 if c in COUNTRIES],
        max_selections=7, key="comp_peers",
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        comp_metric = st.radio(
            t(STRINGS, "metric_label"), ["Index-100", "Z-score", "Value"], key="comp_metric")

    metric_col_c = {"Index-100": "index_100",
                    "Z-score":   "z_score",
                    "Value":     "value"}[comp_metric]

    all_cc = [country] + peers
    df_comp = df_ind[
        (df_ind["country_name"].isin(all_cc)) &
        (df_ind["indicator_name"] == indicator_comp)
    ].copy()
    cmap_comp = {c: PALETTE2[i] for i, c in enumerate(all_cc)}

    with col2:
        fig_comp = go.Figure()
        for c in all_cc:
            df_cc = df_comp[df_comp["country_name"] == c].sort_values("year")
            if df_cc.empty:
                continue
            fig_comp.add_trace(go.Scatter(
                x=df_cc["year"], y=df_cc[metric_col_c],
                mode="lines+markers", name=c,
                line=dict(color=cmap_comp[c], width=3 if c == country else 1.5),
                marker=dict(size=7 if c == country else 5),
                opacity=1.0 if c == country else 0.75,
            ))
        if comp_metric == "Index-100":
            fig_comp.add_hline(y=100, line_color="#304870",
                               line_dash="dash", annotation_text=t(STRINGS, "annot_base2015"))
        if comp_metric == "Z-score":
            fig_comp.add_hline(y=0, line_color="#304870",
                               line_dash="dash", annotation_text=t(STRINGS, "annot_world_mean"))
        fig_comp.update_layout(**_pl(
            title=t(STRINGS, "title_comp_chart").format(ind=indicator_comp[:50], metric=comp_metric),
            yaxis_title=comp_metric, height=450))
        st.plotly_chart(fig_comp, width='stretch')

    year_snap = st.selectbox(t(STRINGS, "snapshot_year"), YEARS[::-1], index=0, key="snap_year")
    df_snap = df_ind[
        (df_ind["country_name"].isin(all_cc)) &
        (df_ind["indicator_name"] == indicator_comp) &
        (df_ind["year"] == year_snap)
    ].sort_values(metric_col_c)

    fig_snap = go.Figure(go.Bar(
        x=df_snap[metric_col_c], y=df_snap["country_name"],
        orientation="h",
        marker_color=[cmap_comp.get(c, "#6B7280")
                      for c in df_snap["country_name"]],
        opacity=0.85,
        text=df_snap[metric_col_c].apply(lambda v: f"{v:.2g}"),
        textposition="outside",
    ))
    fig_snap.update_layout(**_pl(
        title=t(STRINGS, "title_snapshot").format(metric=comp_metric, year=year_snap),
        xaxis_title=comp_metric))
    st.plotly_chart(fig_snap, width='stretch')

    # H.3 CAGR vs World
    section_label(t(STRINGS, "sec_cagr_vs_world"))
    df_grw_c2 = df_growth[
        (df_growth["country_name"] == country) &
        (df_growth["entity_type"] == "COUNTRY")
    ].copy()
    df_grw_w2 = df_growth[df_growth["entity_type"] == "ENTITY"].copy()

    if not df_grw_c2.empty and not df_grw_w2.empty:
        df_vs = df_grw_c2[["indicator_name", "cagr"]].merge(
            df_grw_w2[["indicator_name", "cagr"]].rename(
                columns={"cagr": "world_cagr"}),
            on="indicator_name", how="inner",
        ).dropna(subset=["cagr", "world_cagr"])

        df_vs["_short"]      = df_vs["indicator_name"].apply(
            lambda n: n[:38] + "…" if len(n) > 38 else n)
        df_vs["_outperform"] = df_vs["cagr"] - df_vs["world_cagr"]
        df_vs = df_vs.sort_values("_outperform", ascending=False)

        fig_cagr = go.Figure()
        fig_cagr.add_trace(go.Bar(
            name=country, x=df_vs["_short"], y=df_vs["cagr"],
            marker_color="#60A5FA", opacity=0.85))
        fig_cagr.add_trace(go.Bar(
            name=t(STRINGS, "trace_world"), x=df_vs["_short"], y=df_vs["world_cagr"],
            marker_color="#F59E0B", opacity=0.70))
        fig_cagr.update_layout(**_pl(
            title=t(STRINGS, "title_cagr_vs_world").format(country=country),
            yaxis_title=t(STRINGS, "yaxis_cagr"), barmode="group",
            height=560, xaxis_tickangle=-30,
            margin=dict(l=40, r=40, t=50, b=140)))
        st.plotly_chart(fig_cagr, width='stretch')

        fig_out = go.Figure(go.Bar(
            x=df_vs["_short"], y=df_vs["_outperform"],
            marker_color=["#10B981" if v >= 0 else "#EF4444"
                          for v in df_vs["_outperform"]],
            opacity=0.85,
            text=df_vs["_outperform"].apply(lambda v: f"{v:+.2%}"),
            textposition="outside",
        ))
        fig_out.add_hline(y=0, line_dash="dash", line_color="#EEF4FB")
        fig_out.update_layout(**_pl(
            title=t(STRINGS, "title_outperformance"),
            yaxis_title=t(STRINGS, "yaxis_cagr_delta"), height=520, xaxis_tickangle=-30,
            margin=dict(l=40, r=40, t=50, b=140)))
        st.plotly_chart(fig_out, width='stretch')
    else:
        st.info(t(STRINGS, "info_growth_unavailable"))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · N · MULTI-SCORE
# ─────────────────────────────────────────────────────────────────────────────

# ── TAB 3 : N · MULTI-SCORE ─────────────────────────────────────────────────────
with tab3:
    section_label(t(STRINGS, "sec_multiscore"))

    if row_m.empty:
        st.warning(t(STRINGS, "warn_no_multiscore").format(country=country))
    else:
        r = row_m.iloc[0]

        # ── Expander: formulas & weights ──────────────────────────────────────
        with st.expander(t(STRINGS, "expander_score_defs")):
            st.markdown(t(STRINGS, "score_defs_intro_md"))
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                st.markdown(t(STRINGS, "weights_economic_health_md"))
            with _fc2:
                st.markdown(t(STRINGS, "weights_performance_md"))
            with _fc3:
                st.markdown(t(STRINGS, "weights_competitiveness_md"))
            st.markdown(t(STRINGS, "quadrant_interpretation_md"))

        thin_divider()

        # ── Row 1: Value Position ─────────────────────────────────────────────
        st.markdown(
            '<div style="background:#0F2035;border-left:4px solid #10B981;'
            'border-radius:0 8px 8px 0;padding:.5rem 1rem;margin-bottom:.6rem;">'
            f'<span style="color:#10B981;font-weight:700;font-size:13px;'
            f'text-transform:uppercase;letter-spacing:.06em;">{t(STRINGS, "banner_value_position_title")}</span>'
            '&nbsp;&nbsp;<span style="color:#6890B0;font-size:12px;">'
            f'{t(STRINGS, "banner_value_position_sub")}</span>'
            '</div>',
            unsafe_allow_html=True)

        cols_val = st.columns(3)
        for col, dim in zip(cols_val, DIMS):
            clr    = DIM_COLORS[dim]
            v_val  = r.get(f"{dim}_value_norm",  0)
            v_rank = int(r.get(f"{dim}_value_rank",  0))
            col.markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-top:3px solid {clr};border-radius:0 0 12px 12px;
            padding:1rem;text-align:center;">
  <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.4rem;">{dim}</div>
  <div style="font-size:30px;font-weight:700;color:{clr};">{v_val:.1f}</div>
  <div style="font-size:11px;color:#6890B0;margin-top:4px;">{t(STRINGS, "rank_hash").format(rank=v_rank)}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:.75rem;'></div>", unsafe_allow_html=True)

        # ── Row 2: Growth Momentum ────────────────────────────────────────────
        st.markdown(
            '<div style="background:#0F2035;border-left:4px solid #60A5FA;'
            'border-radius:0 8px 8px 0;padding:.5rem 1rem;margin-bottom:.6rem;">'
            f'<span style="color:#60A5FA;font-weight:700;font-size:13px;'
            f'text-transform:uppercase;letter-spacing:.06em;">{t(STRINGS, "banner_growth_momentum_title")}</span>'
            '&nbsp;&nbsp;<span style="color:#6890B0;font-size:12px;">'
            f'{t(STRINGS, "banner_growth_momentum_sub")}</span>'
            '</div>',
            unsafe_allow_html=True)

        cols_gr = st.columns(3)
        for col, dim in zip(cols_gr, DIMS):
            clr     = DIM_COLORS[dim]
            v_gr    = r.get(f"{dim}_growth_norm", 0)
            gr_rank = int(r.get(f"{dim}_growth_rank", 0))
            col.markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-bottom:3px solid {clr};border-radius:12px 12px 0 0;
            padding:1rem;text-align:center;">
  <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
              letter-spacing:.07em;margin-bottom:.4rem;">{dim}</div>
  <div style="font-size:30px;font-weight:700;color:{clr};">{v_gr:.1f}</div>
  <div style="font-size:11px;color:#6890B0;margin-top:4px;">{t(STRINGS, "rank_hash").format(rank=gr_rank)}</div>
</div>""", unsafe_allow_html=True)

        thin_divider()

        # ── Radar ─────────────────────────────────────────────────────────────
        _rc1, _rc2, _rc3 = st.columns([1, 2, 1])
        with _rc2:
            fig_rad = go.Figure()
            for sfx, clr, name in [
                ("_value_norm",  "#10B981", t(STRINGS, "radar_value_position")),
                ("_growth_norm", "#60A5FA", t(STRINGS, "radar_growth_momentum")),
            ]:
                vals    = [r.get(f"{d}{sfx}", 0) for d in DIMS]
                vals_cl = vals + [vals[0]]
                lbl_cl  = DIMS + [DIMS[0]]
                rgb     = (f"{int(clr[1:3],16)},"
                           f"{int(clr[3:5],16)},"
                           f"{int(clr[5:7],16)}")
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals_cl, theta=lbl_cl, fill="toself", name=name,
                    line=dict(color=clr, width=2),
                    fillcolor=f"rgba({rgb},0.1)",
                ))
            fig_rad.update_layout(
                paper_bgcolor="#1F3050",
                font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
                polar=dict(
                    bgcolor="#162035",
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor="#304870"),
                    angularaxis=dict(gridcolor="#304870"),
                ),
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                height=360, margin=dict(l=40, r=40, t=40, b=40),
                title=dict(text=t(STRINGS, "title_score_radar").format(country=country),
                           font=dict(color="#EEF4FB")),
            )
            st.plotly_chart(fig_rad, width='stretch')

        thin_divider()

        # ── Scatter with axis selectors ───────────────────────────────────────
        section_label(t(STRINGS, "sec_global_ranking"))

        _score_opts = {
            **{t(STRINGS, "dim_value_suffix").format(dim=d):  f"{d}_value_norm"  for d in DIMS},
            **{t(STRINGS, "dim_growth_suffix").format(dim=d): f"{d}_growth_norm" for d in DIMS},
        }
        _slabels = list(_score_opts.keys())

        _sc1, _sc2, _sc3 = st.columns([3, 1.5, 1.5])
        with _sc1:
            highlight_m = st.multiselect(
                t(STRINGS, "highlight_countries"),
                [c for c in df_multi["country_name"].tolist() if c != country],
                default=[c for c in ["China", "Germany", "USA"]
                         if c in df_multi["country_name"].values],
                max_selections=5, key="multi_hl",
            )
        with _sc2:
            _x_label = st.selectbox(t(STRINGS, "x_axis_label"), _slabels, index=0, key="n_x_axis")
        with _sc3:
            _y_label = st.selectbox(t(STRINGS, "y_axis_label"), _slabels, index=3, key="n_y_axis")

        _xc       = _score_opts[_x_label]
        _yc       = _score_opts[_y_label]
        _other    = [l for l in _slabels if l not in (_x_label, _y_label)]
        _zc       = _score_opts[_other[0]] if _other else _xc
        _z_label  = _other[0] if _other else _x_label

        if _x_label == _y_label:
            st.warning(t(STRINGS, "warn_select_different_axes"))
        else:
            df_m_all = df_multi.copy()
            _hl_set  = set([country] + highlight_m)
            df_bg_m  = df_m_all[~df_m_all["country_name"].isin(_hl_set)]

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=df_bg_m[_xc], y=df_bg_m[_yc], mode="markers",
                marker=dict(color="#304870", size=6, opacity=0.5),
                customdata=df_bg_m[["country_name", _zc]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{_x_label}: %{{x:.1f}}<br>"
                    f"{_y_label}: %{{y:.1f}}<br>"
                    f"{_z_label}: %{{customdata[1]:.1f}}"
                    "<extra></extra>"),
                showlegend=False,
            ))
            for c_name in [country] + highlight_m:
                df_hh = df_m_all[df_m_all["country_name"] == c_name]
                if df_hh.empty:
                    continue
                rr  = df_hh.iloc[0]
                clr = "#EEF4FB" if c_name == country else \
                      PALETTE[highlight_m.index(c_name) % len(PALETTE)]
                fig_sc.add_trace(go.Scatter(
                    x=[rr[_xc]], y=[rr[_yc]],
                    mode="markers+text", name=c_name,
                    marker=dict(
                        color=clr,
                        size=14 if c_name == country else 11,
                        symbol="star" if c_name == country else "circle",
                        line=dict(width=1.5, color="#304870")),
                    text=[c_name], textposition="top center",
                    textfont=dict(size=11, color="#EEF4FB"),
                ))
            fig_sc.update_layout(**_pl(
                title=t(STRINGS, "title_scatter_axes").format(x=_x_label, y=_y_label),
                xaxis_title=_x_label, yaxis_title=_y_label, height=420))
            st.plotly_chart(fig_sc, width='stretch')

        thin_divider()

        # ── Dumbbell: Rank Divergence ─────────────────────────────────────────
        section_label(t(STRINGS, "sec_rank_divergence"))

        fig_db   = go.Figure()
        _db_dims = list(reversed(DIMS))
        for dim in _db_dims:
            _vr = int(r.get(f"{dim}_value_rank",  0))
            _gr = int(r.get(f"{dim}_growth_rank", 0))
            _lc = ("#10B981" if _gr < _vr else
                   "#EF4444" if _gr > _vr else "#6B7280")
            fig_db.add_trace(go.Scatter(
                x=[_vr, _gr], y=[dim, dim], mode="lines",
                line=dict(color=_lc, width=2.5),
                showlegend=False, hoverinfo="skip",
            ))

        _vr_list = [int(r.get(f"{d}_value_rank",  0)) for d in _db_dims]
        _gr_list = [int(r.get(f"{d}_growth_rank", 0)) for d in _db_dims]

        fig_db.add_trace(go.Scatter(
            x=_vr_list, y=_db_dims, mode="markers+text",
            name=t(STRINGS, "trace_value_rank"),
            marker=dict(color="#10B981", size=14, symbol="circle",
                        line=dict(width=1, color="#EEF4FB")),
            text=[f"#{v}" for v in _vr_list],
            textposition="top center",
            textfont=dict(size=10, color="#EEF4FB"),
        ))
        fig_db.add_trace(go.Scatter(
            x=_gr_list, y=_db_dims, mode="markers+text",
            name=t(STRINGS, "trace_growth_rank"),
            marker=dict(color="#60A5FA", size=14, symbol="diamond",
                        line=dict(width=1, color="#EEF4FB")),
            text=[f"#{v}" for v in _gr_list],
            textposition="bottom center",
            textfont=dict(size=10, color="#EEF4FB"),
        ))
        fig_db.update_layout(**PLOTLY_LAYOUT)
        fig_db.update_layout(
            title=t(STRINGS, "title_dumbbell"),
            xaxis=dict(title=t(STRINGS, "xaxis_rank"), autorange="reversed",
                       dtick=10, gridcolor="#304870", linecolor="#304870"),
            yaxis=dict(gridcolor="#304870", linecolor="#304870"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25),
            height=420, margin=dict(l=40, r=40, t=50, b=80),
        )
        st.plotly_chart(fig_db, width='stretch')

        _dk1, _dk2 = st.columns(2)
        rd_val = int(r.get("rank_div_value",  0))
        rd_gr  = int(r.get("rank_div_growth", 0))
        _dk1.markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-top:3px solid #10B981;border-radius:0 0 12px 12px;
            padding:1rem;text-align:center;">
  <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
              letter-spacing:.06em;margin-bottom:.3rem;">
    {t(STRINGS, "spread_value_label")}</div>
  <div style="font-size:28px;font-weight:700;color:#10B981;">{rd_val}</div>
  <div style="font-size:11px;color:#6890B0;">{t(STRINGS, "spread_value_caption")}</div>
</div>""", unsafe_allow_html=True)
        _dk2.markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-top:3px solid #60A5FA;border-radius:0 0 12px 12px;
            padding:1rem;text-align:center;">
  <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
              letter-spacing:.06em;margin-bottom:.3rem;">
    {t(STRINGS, "spread_growth_label")}</div>
  <div style="font-size:28px;font-weight:700;color:#60A5FA;">{rd_gr}</div>
  <div style="font-size:11px;color:#6890B0;">{t(STRINGS, "spread_growth_caption")}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · L · CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
