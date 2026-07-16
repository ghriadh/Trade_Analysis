"""
Indicators Analysis — I4 · Advanced Analysis
Growth Dynamics · Correlations · Convergence · Clustering · Forecasting
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
import plotly.express as px
import numpy as np
from scipy.stats import linregress
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

inject_styles()

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ════════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df_ind    = pd.read_csv(IND_FILES / "df_indicators.csv")
    df_cv     = pd.read_csv(IND_FILES / "df_cv.csv")
    df_gini   = pd.read_csv(IND_FILES / "df_gini.csv")
    df_growth = pd.read_csv(IND_FILES / "growth_df.csv")
    df_multi  = pd.read_csv(IND_FILES / "multi_df.csv")
    df_cl     = pd.read_csv(IND_FILES / "cluster_df_multi.csv")
    df_gdp_cl = pd.read_csv(IND_FILES / "GDP_per_capita_USD_cluster_df.csv")
    df_entity = df_ind[df_ind["entity_type"] == "ENTITY"].copy()
    df_ind    = df_ind[(df_ind["entity_type"] == "COUNTRY") &
                       df_ind["country_name"].notna()].copy()
    return df_ind, df_cv, df_gini, df_growth, df_multi, df_cl, df_gdp_cl, df_entity

df_ind, df_cv, df_gini, df_growth, df_multi, df_cl, df_gdp_cl, df_entity = load_data()

INDICATORS = sorted(df_ind["indicator_name"].dropna().unique().tolist())
COUNTRIES  = sorted(df_ind["country_name"].dropna().unique().tolist())
YEARS      = sorted(df_ind["year"].unique())
YEAR_MAX   = int(YEARS[-1])

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

# ════════════════════════════════════════════════════════════════════════════════
# CACHED FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

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
    bm.fit(X_sc, y.values)
    y_pred_all = bm.predict(X_sc)
    if hasattr(bm, "feature_importances_"):
        imp, imp_label = bm.feature_importances_, "Feature Importance"
    elif hasattr(bm, "coef_"):
        imp, imp_label = bm.coef_, "Coefficients (standardized)"
    else:
        imp, imp_label = np.zeros(len(feat_cols)), "Importance"
    return dict(results=results, best_name=best, feat_cols=feat_cols,
                importances=imp, imp_label=imp_label, n_obs=len(X),
                countries=y.index.tolist(),
                y_actual=y.values, y_pred_all=y_pred_all)

@st.cache_data
def _run_lag_regression(target: str, predictors: tuple, cache_key: str,
                        _df: pd.DataFrame):
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
            row["_y"]       = t_row[target]
            row["_year"]    = int(t_row["year"])
            row["_country"] = grp.iloc[i]["country_name"]
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
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42)
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
    bm.fit(X_sc, y)
    if hasattr(bm, "feature_importances_"):
        imp, imp_label = bm.feature_importances_, "Feature Importance"
    elif hasattr(bm, "coef_"):
        imp, imp_label = bm.coef_, "Coefficients (standardized)"
    else:
        imp, imp_label = np.zeros(len(feat_cols)), "Importance"
    hist_fit = {}
    for c_name, grp in df_lag.groupby("_country"):
        X_c  = scaler.transform(grp[feat_cols].values)
        hist_fit[c_name] = dict(years=grp["_year"].values,
                                actual=grp["_y"].values,
                                fitted=bm.predict(X_c))
    return dict(results=results, best_name=best, feat_cols=feat_cols,
                importances=imp, imp_label=imp_label, n_obs=len(df_lag),
                model=bm, scaler=scaler, hist_fit=hist_fit)

# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">Indicators Analysis · Page 4 of 4</div>',
            unsafe_allow_html=True)
st.title("Advanced Analysis")
st.caption("Growth Dynamics · Correlations · Convergence · Clustering · Forecasting")
thin_divider()

# ════════════════════════════════════════════════════════════════════════════════
# SELECTORS
# ════════════════════════════════════════════════════════════════════════════════

_sel1, _sel2 = st.columns(2)
with _sel1:
    default_c = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox("Country", COUNTRIES,
                           index=default_c, key="adv_country")

with _sel2:
    default_idx = next((i for i, n in enumerate(INDICATORS)
                        if "GDP per capita" in n and "current" in n), 0)
    indicator = st.selectbox("Indicator", INDICATORS,
                             index=default_idx, key="adv_indicator")

# Filtered data for the selected indicator
df_i    = df_ind[df_ind["indicator_name"] == indicator].copy()
df_c_i  = df_cv[df_cv["indicator_name"] == indicator].sort_values("year")
df_g_i  = df_gini[df_gini["indicator_name"] == indicator].sort_values("year")
df_gr_i = df_growth[df_growth["indicator_name"] == indicator].copy()
unit    = df_i["indicator_unit"].iloc[0] if not df_i.empty else ""

# Alias used inside forecasting tab
_default_ind = indicator

thin_divider()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f680 Growth Dynamics",
    "\U0001f517 Correlations",
    "\U0001f4d0 Convergence",
    "\U0001f5c2️ L · Clustering",
    "\U0001f52e M · Forecasting",
])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 : GROWTH DYNAMICS
# ──────────────────────────────────────────────────────────────────────
# ── TAB 3 : GROWTH DYNAMICS ──────────────────────────────────
with tab1:
    # ── Mode selector — centered card ─────────────────────────
    st.markdown("""
<style>
div[data-testid="stRadio"] > div {
    background: #1A2E48;
    border: 1px solid #3B82F6;
    border-radius: 14px;
    padding: 0.75rem 1.6rem 0.65rem;
    display: inline-flex;
    gap: 2rem;
}
div[data-testid="stRadio"] label span p {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #C4D5E8 !important;
}
</style>
""", unsafe_allow_html=True)

    _, _col_mode, _ = st.columns([1, 1.6, 1])
    with _col_mode:
        _gr_mode = st.radio(
            "Analysis Mode",
            ["🏳️ Country Focus", "📊 Indicator Focus"],
            horizontal=True, key="gr_mode",
            label_visibility="collapsed",
        )
    thin_divider()

    # ══ MODE A : Indicator Focus ══════════════════════════════
    if _gr_mode == "📊 Indicator Focus":
        section_label("I · Growth Score & Dynamics")

        if df_gr_i.empty:
            st.info("No growth data for this indicator.")
        else:
            # ── A.1 · CAGR × Volatility scatter ───────────────
            _DEFAULT_HL = ["USA", "Japan", "Qatar", "Nigeria",
                           "China", "Canada", "Brazil", "Germany"]
            _default_hl_g = [country] + [c for c in _DEFAULT_HL if c != country]
            highlight_g = st.multiselect(
                "Highlight countries",
                options=sorted(df_gr_i["country_name"].tolist()),
                default=[c for c in _default_hl_g
                         if c in df_gr_i["country_name"].values][:8],
                max_selections=8,
                key="growth_hl",
            )

            df_gr_plot = df_gr_i.copy()
            _hl_set_g = set(highlight_g)

            median_cagr = df_gr_plot["cagr"].median()
            median_vol  = df_gr_plot["volatility"].median()

            df_gr_plot["quadrant"] = df_gr_plot.apply(
                lambda r: "Dynamic & Stable"    if r["cagr"] >= median_cagr and r["volatility"] <= median_vol
                else "Dynamic & Volatile"  if r["cagr"] >= median_cagr
                else "Stagnant & Stable"   if r["volatility"] <= median_vol
                else "Stagnant & Volatile",
                axis=1,
            )

            qcolors = {
                "Dynamic & Stable":    "#10B981",
                "Dynamic & Volatile":  "#F59E0B",
                "Stagnant & Stable":   "#60A5FA",
                "Stagnant & Volatile": "#EF4444",
            }

            fig_gs = go.Figure()
            for q, color in qcolors.items():
                df_q = df_gr_plot[df_gr_plot["quadrant"] == q]
                if df_q.empty:
                    continue
                fig_gs.add_trace(go.Scatter(
                    x=df_q["cagr"] * 100,
                    y=df_q["volatility"],
                    mode="markers",
                    name=q,
                    marker=dict(color=color, size=9, opacity=0.7,
                                line=dict(width=0.5, color="#304870")),
                    customdata=df_q[["country_name","growth_score"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "CAGR: %{x:.1f}%<br>"
                                  "Volatility: %{y:.2f}<br>"
                                  "Score: %{customdata[1]:.3f}<extra></extra>",
                ))

            # Highlighted countries overlay — rings + visible labels
            _df_hl_g = df_gr_plot[df_gr_plot["country_name"].isin(_hl_set_g)]
            if not _df_hl_g.empty:
                fig_gs.add_trace(go.Scatter(
                    x=_df_hl_g["cagr"] * 100,
                    y=_df_hl_g["volatility"],
                    mode="markers+text",
                    showlegend=False,
                    marker=dict(color="rgba(0,0,0,0)", size=15,
                                line=dict(width=2, color="#FFFFFF")),
                    text=_df_hl_g["country_name"],
                    textposition="top center",
                    textfont=dict(size=13, color="#FFFFFF",
                                  family="IBM Plex Sans"),
                    customdata=_df_hl_g[["country_name","growth_score"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "CAGR: %{x:.1f}%<br>"
                                  "Volatility: %{y:.2f}<br>"
                                  "Score: %{customdata[1]:.3f}<extra></extra>",
                ))

            fig_gs.add_vline(x=median_cagr * 100, line_dash="dash",
                             line_color="#6B8FAF", line_width=1.2)
            fig_gs.add_hline(y=median_vol, line_dash="dash",
                             line_color="#6B8FAF", line_width=1.2)
            fig_gs.update_layout(**PLOTLY_LAYOUT,
                title="CAGR × Volatility — Growth Matrix",
                xaxis_title="CAGR (%)",
                yaxis_title="Volatility",
                height=500,
            )
            st.plotly_chart(fig_gs, width='stretch')

            thin_divider()

            # ── A.2 · Growth Score distribution ───────────────
            section_label("A.2 · Growth Score Distribution")
            fig_gsd = go.Figure()
            fig_gsd.add_trace(go.Histogram(
                x=df_gr_i["growth_score"],
                nbinsx=25,
                marker_color="#A78BFA",
                opacity=0.8,
                name="Growth Score",
            ))
            fig_gsd.add_vline(
                x=df_gr_i["growth_score"].median(),
                line_dash="dash", line_color="#EEF4FB",
                annotation_text=f"Median: {df_gr_i['growth_score'].median():.3f}",
            )
            fig_gsd.update_layout(**PLOTLY_LAYOUT,
                title="Growth Score Distribution",
                xaxis_title="Growth Score",
                yaxis_title="Count",
                height=380,
            )
            st.plotly_chart(fig_gsd, width='stretch')

        # GDP per capita clustering
        is_gdp = "GDP per capita" in indicator
        if is_gdp and not df_gdp_cl.empty:
            section_label("GDP per Capita — Country Clusters")
            _default_hl_cl = [country] + [c for c in _DEFAULT_HL if c != country]
            highlight_cl = st.multiselect(
                "Highlight",
                options=sorted(df_gdp_cl["country_name"].tolist()),
                default=[c for c in _default_hl_cl
                         if c in df_gdp_cl["country_name"].values][:8],
                max_selections=8,
                key="gdp_hl",
            )
            df_cl = df_gdp_cl.copy()
            _hl_set_cl = set(highlight_cl)
            df_cl["_size"] = (df_cl["growth_score"].rank() /
                              len(df_cl) * 25 + 6).clip(6, 31)

            fig_cl = go.Figure()
            for cl_id in sorted(df_cl["cluster_id"].unique()):
                df_q = df_cl[df_cl["cluster_id"] == cl_id]
                fig_cl.add_trace(go.Scatter(
                    x=df_q["cagr"] * 100,
                    y=df_q["volatility"],
                    mode="markers",
                    name=f"Cluster {cl_id}",
                    marker=dict(
                        color=CLUSTER_COLORS.get(cl_id, "#6B7280"),
                        size=df_q["_size"], opacity=0.7,
                        line=dict(width=0.5, color="#304870"),
                    ),
                    customdata=df_q[["country_name","growth_score"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "CAGR: %{x:.1f}%<br>"
                                  "Volatility: %{y:.2f}<br>"
                                  "Score: %{customdata[1]:.3f}<extra></extra>",
                ))

            # Highlighted overlay — rings + labels
            _df_hl_cl = df_cl[df_cl["country_name"].isin(_hl_set_cl)]
            if not _df_hl_cl.empty:
                fig_cl.add_trace(go.Scatter(
                    x=_df_hl_cl["cagr"] * 100,
                    y=_df_hl_cl["volatility"],
                    mode="markers+text",
                    showlegend=False,
                    marker=dict(color="rgba(0,0,0,0)", size=15,
                                line=dict(width=2, color="#FFFFFF")),
                    text=_df_hl_cl["country_name"],
                    textposition="top center",
                    textfont=dict(size=13, color="#FFFFFF",
                                  family="IBM Plex Sans"),
                    customdata=_df_hl_cl[["country_name","growth_score"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "CAGR: %{x:.1f}%<br>"
                                  "Volatility: %{y:.2f}<br>"
                                  "Score: %{customdata[1]:.3f}<extra></extra>",
                ))
            fig_cl.update_layout(**PLOTLY_LAYOUT,
                title="GDP per Capita — KMeans Clusters (CAGR × Volatility)",
                xaxis_title="CAGR (%)",
                yaxis_title="Volatility",
                height=500,
            )
            st.plotly_chart(fig_cl, width='stretch')

    # ══ MODE B : Country Focus ════════════════════════════════
    else:
        # Remove stale keys left by the old in-tab selector code
        for _stale in ("gr_b_ind", "gr_b_ind_sync", "gr_country"):
            st.session_state.pop(_stale, None)

        df_gr_c = df_growth[
            (df_growth["country_name"] == country) &
            (df_growth["entity_type"] == "COUNTRY")
        ].copy()

        if df_gr_c.empty:
            st.info(f"No growth data for {country}.")
        else:
            # ── Per-indicator KPIs ─────────────────────────────
            _ind_short = indicator[:45] + "…" if len(indicator) > 45 else indicator
            section_label(f"Growth Profile — {country} × {_ind_short}")
            _row_ki = df_gr_c[df_gr_c["indicator_name"] == indicator]

            if not _row_ki.empty:
                _ki = _row_ki.iloc[0]
                k1, k2, k3, k4, k5, k6 = st.columns(6)

                # Spacer div keeps all cards the same height
                _spacer = '<div style="font-size:10px;visibility:hidden;margin-top:4px;">·</div>'

                def _kpi_full(col, label, value, color=""):
                    cls = f"kpi-card {color}".strip()
                    vcls = f"kpi-value {color}".strip()
                    col.markdown(
                        f'<div class="{cls}">'
                        f'<div class="kpi-label">{label}</div>'
                        f'<div class="{vcls}">{value}</div>'
                        f'{_spacer}</div>',
                        unsafe_allow_html=True)

                _gs  = _ki.get("growth_score")
                _gsc = "teal" if pd.notna(_gs) and _gs > 0 else ""
                _kpi_full(k1, "Growth Score",
                    f"{_gs:+.3f}" if pd.notna(_gs) else "—", _gsc)

                _ca  = _ki.get("cagr")
                _cac = "teal" if pd.notna(_ca) and _ca > 0 else ""
                _kpi_full(k2, "CAGR",
                    f"{_ca*100:+.2f}%" if pd.notna(_ca) else "—", _cac)

                _vo  = _ki.get("volatility")
                _kpi_full(k3, "Volatility",
                    f"{_vo:.3f}" if pd.notna(_vo) else "—", "")

                # Shock Freq — sub-line with colored pos/neg counts
                _sf  = _ki.get("shock_frequency")
                _nps = int(_ki.get("n_positive_shocks", 0) or 0)
                _nns = int(_ki.get("n_negative_shocks", 0) or 0)
                _sf_val = f"{_sf:.2f}" if pd.notna(_sf) else "—"
                k4.markdown(
                    f'<div class="kpi-card amber">'
                    f'<div class="kpi-label">Shock Freq.</div>'
                    f'<div class="kpi-value amber">{_sf_val}</div>'
                    f'<div style="font-size:10px;margin-top:4px;">'
                    f'<span style="color:#34D399;">+{_nps}</span>'
                    f'<span style="color:#94A3B8;"> pos &nbsp;/&nbsp; </span>'
                    f'<span style="color:#F87171;">−{_nns}</span>'
                    f'<span style="color:#94A3B8;"> neg</span>'
                    f'</div></div>',
                    unsafe_allow_html=True)

                _sd  = _ki.get("share_dynamic")
                _kpi_full(k5, "Dynamic Phase",
                    f"{_sd*100:.0f}%" if pd.notna(_sd) else "—", "teal")

                _su  = _ki.get("share_under_pressure")
                _kpi_full(k6, "Pressure Phase",
                    f"{_su*100:.0f}%" if pd.notna(_su) else "—", "")

            else:
                st.info(
                    f"No growth score data for **{country}** × **{indicator}**. "
                    "This indicator may have too few data points for CAGR computation, "
                    "or may not be present in the growth dataset for this country."
                )

            with st.expander("📖 Growth Score & Metrics — Methodology",
                             expanded=False):
                st.markdown("""
**Growth Score formula:**
`Score = 0.50 × z_CAGR − 0.20 × z_Volatility − 0.15 × z_Under_Pressure − 0.15 × z_Shock_Freq`

| Metric | Definition | Weight |
|---|---|---|
| **CAGR** | Compound Annual Growth Rate over full period | 0.50 ✅ higher = better |
| **Volatility** | Std dev of YoY growth — measures consistency | 0.20 ❌ lower = better |
| **Dynamic Phase** | % of years with YoY growth ≥ Q60 of global distribution | — |
| **Pressure Phase** | % of years with YoY growth < Q25 (contraction/stagnation) | 0.15 ❌ lower = better |
| **Shock Freq.** | Nb of years where |z_growth| ≥ 2 (extreme events) / total years | 0.15 ❌ lower = better |

All z-score components normalized vs all countries × indicators. Higher score = stronger & more stable growth.
""")

            thin_divider()

            # ── B.2 · Growth Score per Indicator ──────────────
            section_label("B.2 · Growth Score by Indicator")
            _df_b2 = (df_gr_c.dropna(subset=["growth_score"])
                      .sort_values("growth_score", ascending=True))
            if not _df_b2.empty:
                _ind_b2 = [n[:42]+"…" if len(n)>42 else n
                           for n in _df_b2["indicator_name"]]
                _clr_b2 = ["#10B981" if v >= 0 else "#EF4444"
                            for v in _df_b2["growth_score"]]
                fig_b2 = go.Figure()
                fig_b2.add_trace(go.Bar(
                    x=_df_b2["growth_score"], y=_ind_b2,
                    orientation="h",
                    marker_color=_clr_b2, opacity=0.85,
                    text=[f"{v:+.3f}" for v in _df_b2["growth_score"]],
                    textposition="outside", textfont=dict(size=10),
                ))
                fig_b2.add_vline(x=0, line_color="#EEF4FB", line_width=1)
                fig_b2.update_layout(**{**PLOTLY_LAYOUT,
                    "title": f"Growth Score — {country} (all indicators)",
                    "xaxis": dict(title="Growth Score",
                                  gridcolor="#304870", linecolor="#304870"),
                    "margin": dict(l=260, r=80, t=50, b=40),
                    "height": max(380, len(_df_b2)*35 + 80),
                })
                st.plotly_chart(fig_b2, width='stretch')

            thin_divider()

            # ── B.3 · CAGR × Volatility — full width ──────────
            section_label("B.3 · CAGR × Volatility by Indicator")
            # Reset multiselect when the primary indicator changes
            if st.session_state.get("gr_b3_prev_ind") != indicator:
                st.session_state.pop("gr_b3_inds", None)
                st.session_state["gr_b3_prev_ind"] = indicator
            _inds_b3 = st.multiselect(
                "Indicators to display",
                options=INDICATORS,
                default=[indicator],
                key="gr_b3_inds",
            )
            _df_b3 = (df_gr_c[df_gr_c["indicator_name"].isin(_inds_b3)]
                      .dropna(subset=["cagr","volatility"])
                      if _inds_b3 else pd.DataFrame())
            if not _df_b3.empty:
                _med_c = df_growth["cagr"].median()
                _med_v = df_growth["volatility"].median()
                _sc_min = _df_b3["growth_score"].min()
                _sc_rng = max(_df_b3["growth_score"].max() - _sc_min, 1e-6)

                def _scol(s):
                    t = float(np.clip((s - _sc_min) / _sc_rng, 0, 1))
                    r = int(239*(1-t) + 16*t)
                    g = int(68*(1-t) + 185*t)
                    b = int(68*(1-t) + 129*t)
                    return f"rgb({r},{g},{b})"

                _clr_b3 = [_scol(s) for s in _df_b3["growth_score"].fillna(_sc_min)]
                _sz_b3  = (_df_b3["shock_frequency"].fillna(0)*200+8).clip(8,32)
                _lbl_b3 = [n[:20]+"…" if len(n)>20 else n
                           for n in _df_b3["indicator_name"]]
                fig_b3 = go.Figure()
                fig_b3.add_trace(go.Scatter(
                    x=_df_b3["cagr"]*100, y=_df_b3["volatility"],
                    mode="markers+text",
                    marker=dict(color=_clr_b3, size=_sz_b3, opacity=0.85,
                                line=dict(width=1, color="#304870")),
                    text=_lbl_b3,
                    textposition="top center",
                    textfont=dict(size=9, color="#EEF4FB"),
                    customdata=_df_b3[["indicator_name","growth_score",
                                       "shock_frequency"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "CAGR: %{x:+.2f}%<br>"
                                  "Volatility: %{y:.3f}<br>"
                                  "Score: %{customdata[1]:.3f}<br>"
                                  "Shock freq: %{customdata[2]:.2f}"
                                  "<extra></extra>",
                    showlegend=False,
                ))
                # Compute padded axis range ensuring all 4 quadrants are visible
                _xv = _df_b3["cagr"] * 100
                _yv = _df_b3["volatility"]
                _mx = _med_c * 100
                _xpad = max((_xv - _mx).abs().max(), abs(_mx), 1) * 0.4
                _ypad = max((_yv - _med_v).abs().max(), _med_v, 0.01) * 0.4
                _xmin = min(_xv.min(), _mx) - _xpad
                _xmax = max(_xv.max(), _mx) + _xpad
                _ymin = max(0.0, min(_yv.min(), _med_v) - _ypad)
                _ymax = max(_yv.max(), _med_v) + _ypad

                fig_b3.add_vline(x=_mx, line_dash="dash",
                                 line_color="#6B8FAF", line_width=1.2)
                fig_b3.add_hline(y=_med_v, line_dash="dash",
                                 line_color="#6B8FAF", line_width=1.2)
                # Labels at 25% into each quadrant from the median (near center)
                _f = 0.25
                for _qt, _qx, _qy, _qc in [
                    ("Stagnant & Volatile", _mx - (_mx-_xmin)*_f, _med_v + (_ymax-_med_v)*_f, "#EF4444"),
                    ("Dynamic & Volatile",  _mx + (_xmax-_mx)*_f, _med_v + (_ymax-_med_v)*_f, "#F59E0B"),
                    ("Stagnant & Stable",   _mx - (_mx-_xmin)*_f, _med_v - (_med_v-_ymin)*_f, "#60A5FA"),
                    ("Dynamic & Stable",    _mx + (_xmax-_mx)*_f, _med_v - (_med_v-_ymin)*_f, "#10B981"),
                ]:
                    fig_b3.add_annotation(
                        x=_qx, y=_qy, xref="x", yref="y",
                        text=_qt, showarrow=False,
                        font=dict(size=9, color=_qc),
                        bgcolor="#1F3050", borderpad=2,
                    )
                fig_b3.update_layout(**{**PLOTLY_LAYOUT,
                    "title": f"CAGR × Volatility — {country}",
                    "xaxis": dict(title="CAGR (%)", range=[_xmin, _xmax],
                                  gridcolor="#304870", linecolor="#304870"),
                    "yaxis": dict(title="Volatility", range=[_ymin, _ymax],
                                  gridcolor="#304870", linecolor="#304870"),
                    "height": 460,
                })
                st.plotly_chart(fig_b3, width='stretch')
                st.caption("Bubble size = shock frequency · "
                           "Color: green = high growth score, red = low")
            else:
                st.info(
                    "No CAGR / volatility data for the selected indicator(s). "
                    "Try selecting a different indicator from the list above."
                )

            thin_divider()

            # ── B.4 + B.5 ─────────────────────────────────────
            col_b4, col_b5 = st.columns(2)

            with col_b4:
                section_label("B.4 · Regime Distribution")
                _df_b4 = df_gr_c.dropna(
                    subset=["share_dynamic","share_moderate",
                            "share_under_pressure"]
                ).sort_values("share_dynamic", ascending=True)
                _ind_b4 = [n[:32]+"…" if len(n)>32 else n
                           for n in _df_b4["indicator_name"]]
                if not _df_b4.empty:
                    fig_b4 = go.Figure()
                    fig_b4.add_trace(go.Bar(
                        x=_df_b4["share_under_pressure"],
                        y=_ind_b4, orientation="h",
                        name="Pressure",
                        marker_color="#EF4444", opacity=0.85,
                        text=[f"{v:.0%}" for v in _df_b4["share_under_pressure"]],
                        textposition="inside",
                        textfont=dict(size=9, color="#EEF4FB"),
                        insidetextanchor="middle",
                    ))
                    fig_b4.add_trace(go.Bar(
                        x=_df_b4["share_moderate"],
                        y=_ind_b4, orientation="h",
                        name="Moderate",
                        marker_color="#60A5FA", opacity=0.8,
                        text=[f"{v:.0%}" for v in _df_b4["share_moderate"]],
                        textposition="inside",
                        textfont=dict(size=9, color="#EEF4FB"),
                        insidetextanchor="middle",
                    ))
                    fig_b4.add_trace(go.Bar(
                        x=_df_b4["share_dynamic"],
                        y=_ind_b4, orientation="h",
                        name="Dynamic",
                        marker_color="#10B981", opacity=0.85,
                        text=[f"{v:.0%}" for v in _df_b4["share_dynamic"]],
                        textposition="inside",
                        textfont=dict(size=9, color="#EEF4FB"),
                        insidetextanchor="middle",
                    ))
                    fig_b4.update_layout(**{**PLOTLY_LAYOUT,
                        "title": f"Regime Distribution — {country}",
                        "barmode": "stack",
                        "xaxis": dict(title="Share of years",
                                      tickformat=".0%",
                                      gridcolor="#304870",
                                      linecolor="#304870"),
                        "margin": dict(l=220, r=40, t=50, b=40),
                        "height": max(380, len(_df_b4)*35 + 80),
                    })
                    st.plotly_chart(fig_b4, width='stretch')

            with col_b5:
                section_label("B.5 · Shock Profile")
                _df_b5 = (df_gr_c.dropna(subset=["shock_frequency"])
                          .sort_values("shock_frequency", ascending=True))
                _ind_b5 = [n[:32]+"…" if len(n)>32 else n
                           for n in _df_b5["indicator_name"]]
                if not _df_b5.empty:
                    fig_b5 = go.Figure()
                    fig_b5.add_trace(go.Bar(
                        x=_df_b5["n_positive_shocks"],
                        y=_ind_b5, orientation="h",
                        name="Positive shocks",
                        marker_color="#10B981", opacity=0.85,
                        text=[str(int(v)) for v in _df_b5["n_positive_shocks"]],
                        textposition="inside",
                        textfont=dict(size=9, color="#EEF4FB"),
                        insidetextanchor="middle",
                    ))
                    fig_b5.add_trace(go.Bar(
                        x=-_df_b5["n_negative_shocks"],
                        y=_ind_b5, orientation="h",
                        name="Negative shocks",
                        marker_color="#EF4444", opacity=0.85,
                        text=[str(int(v)) for v in _df_b5["n_negative_shocks"]],
                        textposition="inside",
                        textfont=dict(size=9, color="#EEF4FB"),
                        insidetextanchor="middle",
                    ))
                    fig_b5.add_vline(x=0, line_color="#EEF4FB", line_width=1)
                    fig_b5.update_layout(**{**PLOTLY_LAYOUT,
                        "title": f"Shock Balance — {country}",
                        "barmode": "overlay",
                        "xaxis": dict(title="Number of shocks",
                                      gridcolor="#304870",
                                      linecolor="#304870"),
                        "margin": dict(l=220, r=40, t=50, b=40),
                        "height": max(380, len(_df_b5)*35 + 80),
                    })
                    st.plotly_chart(fig_b5, width='stretch')

            thin_divider()

            # ── B.6 · Radar ───────────────────────────────────
            _ind_short_b6 = indicator[:45]+"…" if len(indicator)>45 else indicator
            section_label(f"B.6 · Growth Radar — {_ind_short_b6}")
            _row_r = df_gr_c[df_gr_c["indicator_name"] == indicator]
            if _row_r.empty:
                st.info(f"No data for {country} — {indicator}")
            else:
                _r = _row_r.iloc[0]

                def _pct(val, col, invert=False):
                    arr = df_growth[col].dropna()
                    p = float((arr <= val).mean())
                    return (1 - p) if invert else p

                _rv = [
                    _pct(_r["cagr"],             "cagr"),
                    _pct(_r["volatility"],       "volatility", True),
                    float(_r.get("share_dynamic", 0) or 0),
                    1 - float(_r.get("share_under_pressure", 0) or 0),
                    _pct(_r["shock_balance"],    "shock_balance"),
                ]
                _rl = ["CAGR", "Low Volatility",
                       "Dynamic Phase", "Stability", "Shock Balance"]
                _, _col_rad, _ = st.columns([1, 2, 1])
                with _col_rad:
                    fig_b6 = go.Figure()
                    fig_b6.add_trace(go.Scatterpolar(
                        r=_rv + [_rv[0]], theta=_rl + [_rl[0]],
                        fill="toself",
                        line=dict(color="#60A5FA", width=2),
                        fillcolor="rgba(96,165,250,0.15)",
                        name=country,
                    ))
                    fig_b6.update_layout(
                        paper_bgcolor="#1F3050",
                        font=dict(family="IBM Plex Sans",
                                  color="#C4D5E8", size=12),
                        polar=dict(
                            bgcolor="#162035",
                            radialaxis=dict(visible=True, range=[0,1],
                                            gridcolor="#304870"),
                            angularaxis=dict(gridcolor="#304870"),
                        ),
                        legend=dict(bgcolor="#1F3050",
                                    bordercolor="#304870"),
                        height=420,
                        margin=dict(l=50, r=50, t=60, b=50),
                        title=dict(
                            text=f"{country} — {indicator[:35]}"
                                 f"{'…' if len(indicator)>35 else ''}",
                            font=dict(color="#EEF4FB", size=13),
                        ),
                    )
                    st.plotly_chart(fig_b6, width='stretch')
                    st.caption("Values = percentile vs all countries · "
                               "1 = best, 0 = worst")




# ──────────────────────────────────────────────────────────────────────
# TAB 2 : CORRELATIONS
# ──────────────────────────────────────────────────────────────────────
# ── TAB 6 : CORRELATIONS ─────────────────────────────────────────
with tab2:
    section_label("J · Correlations & Structure")

    _DEF_CNTRIES = ["USA", "China", "Germany", "France", "India", "Brazil",
                    "Japan", "United Kingdom", "South Korea", "Canada"]
    _avail_c = sorted(df_i["country_name"].dropna().unique().tolist())
    _def_c6  = [c for c in _DEF_CNTRIES if c in _avail_c]

    # ── J.1 · Cross-Country Correlation ──────────────────────────
    section_label("J.1 · Cross-Country Correlation")
    with st.expander("📖 What is Cross-Country Correlation?", expanded=False):
        st.markdown("""
**Concept:** For the selected indicator, this measures how **synchronized** countries are in their evolution over time.

A high correlation between two countries means they tend to grow and contract in the same years — a signal of structural linkage, common exposure to global shocks, or trade integration.

| Metric option | What it measures |
|---|---|
| **YoY Growth** | Cycle synchronization — do countries accelerate/decelerate together? *(most meaningful)* |
| **Absolute Value** | Level co-movement — do countries follow the same absolute trajectory? |
| **Index 100** | Relative dynamic — do countries grow at a similar cumulative pace? |

**How to read the heatmap:**
- **Red** = strong positive correlation (countries move together)
- **Blue** = strong negative correlation (when one grows, the other contracts)
- **White** = no correlation (independent dynamics)
- **Rolling correlation** shows whether the synchronization is stable or has evolved over time — a declining line signals decoupling.
""")

    _j1col1, _j1col2 = st.columns([2, 1])
    with _j1col1:
        _j1_countries = st.multiselect(
            "Countries", _avail_c, default=_def_c6,
            max_selections=20, key="j1_countries",
        )
    with _j1col2:
        _j1_metric = st.selectbox(
            "Metric", ["yoy_growth", "value", "index_100"],
            format_func=lambda x: {"yoy_growth": "YoY Growth",
                                   "value": "Absolute Value",
                                   "index_100": "Index 100"}[x],
            key="j1_metric",
        )

    if _j1_countries and len(_j1_countries) >= 2:
        _df_j1 = df_i[df_i["country_name"].isin(_j1_countries)].copy()
        if _j1_metric == "index_100":
            _base = (_df_j1.groupby("country_name")
                     .apply(lambda g: g.sort_values("year")["value"].iloc[0])
                     .rename("base"))
            _df_j1 = _df_j1.merge(_base, on="country_name")
            _df_j1["index_100"] = _df_j1["value"] / _df_j1["base"] * 100
        _pivot_j1 = _df_j1.pivot_table(
            index="year", columns="country_name", values=_j1_metric
        )
        _corr_j1 = _pivot_j1.corr(method="pearson")

        _n = len(_corr_j1)
        fig_j1 = go.Figure(go.Heatmap(
            z=_corr_j1.values,
            x=_corr_j1.columns.tolist(),
            y=_corr_j1.index.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(_corr_j1.values, 2),
            texttemplate="%{text:.2f}", textfont=dict(size=9),
            colorbar=dict(title="r"), hoverongaps=False,
        ))
        fig_j1.update_layout(**{**PLOTLY_LAYOUT,
            "title": f"Cross-Country Correlation — {indicator[:50]}",
            "height": max(420, _n * 38 + 100),
            "margin": dict(l=120, r=40, t=50, b=120),
            "xaxis": dict(tickangle=-45, gridcolor="#304870"),
            "yaxis": dict(gridcolor="#304870"),
        })
        st.plotly_chart(fig_j1, width='stretch')

        _pairs = []
        for i, c1 in enumerate(_corr_j1.columns):
            for c2 in _corr_j1.columns[i+1:]:
                _pairs.append({"Country A": c1, "Country B": c2,
                               "r": round(_corr_j1.loc[c1, c2], 3)})
        _df_pairs = (pd.DataFrame(_pairs)
                     .sort_values("r", key=abs, ascending=False)
                     .head(8).reset_index(drop=True))
        if not _df_pairs.empty:
            st.caption("Top correlated pairs")
            st.dataframe(
                _df_pairs.style
                    .background_gradient(subset=["r"], cmap="RdBu", vmin=-1, vmax=1)
                    .format({"r": "{:+.3f}"}),
                width='stretch', hide_index=True,
            )

        thin_divider()
        section_label("J.1b · Rolling Correlation")
        _rc1, _rc2 = st.columns([2, 1])
        with _rc1:
            _j1_pairs_opts = [f"{r['Country A']} × {r['Country B']}"
                              for _, r in _df_pairs.head(5).iterrows()]
            _j1_pair_sel = st.selectbox("Pair", _j1_pairs_opts, key="j1_pair")
        with _rc2:
            _j1_window = st.slider("Rolling window (years)", 2, 5, 3, key="j1_win")

        _p_a, _p_b = _j1_pair_sel.split(" × ")
        _s_a = _pivot_j1[_p_a].dropna() if _p_a in _pivot_j1 else None
        _s_b = _pivot_j1[_p_b].dropna() if _p_b in _pivot_j1 else None
        if _s_a is not None and _s_b is not None:
            _common = _s_a.index.intersection(_s_b.index)
            _roll_corr = (_s_a.loc[_common].rolling(_j1_window)
                          .corr(_s_b.loc[_common]))
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=_common, y=_roll_corr, mode="lines+markers",
                line=dict(color="#60A5FA", width=2.5), marker=dict(size=7),
                name=f"r (window={_j1_window}y)",
            ))
            fig_rc.add_hline(y=0, line_dash="dash",
                             line_color="#EEF4FB", line_width=1)
            fig_rc.update_layout(**{**PLOTLY_LAYOUT,
                "title": f"Rolling Correlation — {_p_a} × {_p_b}",
                "yaxis": dict(title="Pearson r", range=[-1.05, 1.05],
                              gridcolor="#304870", linecolor="#304870"),
                "height": 340,
            })
            st.plotly_chart(fig_rc, width='stretch')
    else:
        st.info("Select at least 2 countries.")

    thin_divider()

    # ── J.2 · Inter-Indicator Correlation ────────────────────────
    section_label("J.2 · Inter-Indicator Correlation")
    with st.expander("📖 What is Inter-Indicator Correlation?", expanded=False):
        st.markdown("""
**Concept:** Instead of comparing countries on one indicator, this flips the question: **do economic indicators move together across countries?**

Each country becomes one observation; each indicator a variable. We compute the mean YoY growth per country × indicator, then measure pairwise correlation.

A strong correlation between two indicators means that countries which grow fast on one tend to grow fast (or slow) on the other — revealing structural economic linkages.

**How to read the heatmap:**
- **Red** = when one indicator grows fast in a country, the other tends to as well (positive link)
- **Blue** = inverse relationship (one grows when the other contracts)
- **White** = independent dynamics

**Examples to look for:**
- GDP ↔ Trade % of GDP: richer countries tend to be more open (positive)
- Inflation ↔ Real interest rate: central banks raise rates when inflation spikes (negative)
- GDP ↔ External debt: ambiguous — can be positive (borrowing to grow) or negative

**Use the selector** to focus on 2–6 indicators and see their relationships more clearly.
""")

    _all_inds_j2 = sorted(df_ind["indicator_name"].dropna().unique().tolist())
    _j2_sel = st.multiselect(
        "Indicators to include", _all_inds_j2,
        default=_all_inds_j2, key="j2_inds",
    )

    # Build pivot: country × indicator (mean YoY growth)
    _pivot_j2_full = (df_ind.groupby(["country_name","indicator_name"])["yoy_growth"]
                      .mean().reset_index()
                      .pivot(index="country_name", columns="indicator_name",
                             values="yoy_growth"))

    if _j2_sel and len(_j2_sel) >= 2:
        _cols_j2 = [c for c in _j2_sel if c in _pivot_j2_full.columns]
        _pivot_j2 = _pivot_j2_full[_cols_j2].dropna(how="any")
        _corr_j2 = _pivot_j2.corr(method="pearson")
        _short = {n: n[:28]+"…" if len(n)>28 else n for n in _corr_j2.columns}
        _corr_j2_disp = _corr_j2.rename(columns=_short, index=_short)

        fig_j2 = go.Figure(go.Heatmap(
            z=_corr_j2_disp.values,
            x=_corr_j2_disp.columns.tolist(),
            y=_corr_j2_disp.index.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(_corr_j2_disp.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=9 if len(_cols_j2) <= 8 else 7),
            colorbar=dict(title="r"), hoverongaps=False,
        ))
        _nj2 = len(_corr_j2_disp)
        fig_j2.update_layout(**{**PLOTLY_LAYOUT,
            "title": "Inter-Indicator Correlation — Mean YoY Growth",
            "height": max(420, _nj2 * 44 + 100),
            "margin": dict(l=230, r=40, t=50, b=230),
            "xaxis": dict(tickangle=-45, gridcolor="#304870"),
            "yaxis": dict(gridcolor="#304870"),
        })
        st.plotly_chart(fig_j2, width='stretch')

        _sp = []
        for i, c1 in enumerate(_corr_j2.columns):
            for c2 in _corr_j2.columns[i+1:]:
                r = _corr_j2.loc[c1, c2]
                if abs(r) >= 0.4:
                    _sp.append({"Indicator A": c1, "Indicator B": c2,
                                "r": round(r, 3)})
        if _sp:
            _df_sp = (pd.DataFrame(_sp).sort_values("r", key=abs, ascending=False)
                      .head(10).reset_index(drop=True))
            st.caption("Strong pairs (|r| ≥ 0.4)")
            st.dataframe(
                _df_sp.style
                    .background_gradient(subset=["r"], cmap="RdBu", vmin=-1, vmax=1)
                    .format({"r": "{:+.3f}"}),
                width='stretch', hide_index=True,
            )
    else:
        st.info("Select at least 2 indicators.")

    thin_divider()

    # ── J.3 · PCA ────────────────────────────────────────────────
    section_label("J.3 · PCA — Country Structure")
    with st.expander("📖 What is PCA and how to read this chart?", expanded=False):
        st.markdown("""
**Concept:** Principal Component Analysis (PCA) reduces the 16 economic indicators to 2 main axes (PC1 and PC2) that capture the most structural variation across countries.

**Input:** Mean z-score per country × indicator (normalized level — positive = above world average, negative = below). This separates countries by their **structural economic profile**, not just their growth rate.

**How to read the scatter (country positions):**
- Each point = one country, positioned according to its profile across all indicators simultaneously
- **Countries close together** share similar economic structures
- **Countries far apart** are structurally different
- PC1 typically separates **development level** (rich vs poor); PC2 often separates **economic model** (open trade vs closed, stable vs volatile)
- Hover over any point to see the country name

**How to read the Loadings chart (below the scatter):**
- Each bar = one indicator's contribution to PC1 or PC2
- A large positive bar on PC1 means: countries scoring high on PC1 tend to have high values for that indicator
- A large negative bar means the opposite
- This lets you **interpret what each axis represents economically**
""")

    _j3_hl = st.multiselect(
        "Highlight countries", _avail_c, default=_def_c6[:6],
        max_selections=10, key="j3_hl",
    )

    # PCA input: country × indicator mean z_score (level-based, not growth)
    # fillna(0) = world average for missing indicators (e.g. External debt
    # not reported for developed countries) — keeps all countries in PCA
    _pca_pivot = (df_ind[df_ind["entity_type"] == "COUNTRY"]
                  .groupby(["country_name","indicator_name"])["z_score"]
                  .mean().reset_index()
                  .pivot(index="country_name", columns="indicator_name",
                         values="z_score")
                  .fillna(0))

    if len(_pca_pivot) >= 5:
        _X_pca = StandardScaler().fit_transform(_pca_pivot.values)
        _pca_model = SkPCA(n_components=min(4, _pca_pivot.shape[1]))
        _pca_scores = _pca_model.fit_transform(_X_pca)
        _var = _pca_model.explained_variance_ratio_
        _loadings = pd.DataFrame(
            _pca_model.components_.T,
            index=_pca_pivot.columns,
            columns=[f"PC{i+1}" for i in range(_pca_model.n_components_)],
        )
        _scores_df = pd.DataFrame(
            _pca_scores[:, :2], columns=["PC1","PC2"],
            index=_pca_pivot.index,
        )
        _scores_df["highlight"] = _scores_df.index.isin(_j3_hl)

        # Scatter — country positions
        fig_j3 = go.Figure()
        _bg = _scores_df[~_scores_df["highlight"]]
        fig_j3.add_trace(go.Scatter(
            x=_bg["PC1"], y=_bg["PC2"], mode="markers",
            marker=dict(color="#304870", size=6, opacity=0.55),
            hovertext=_bg.index,
            hovertemplate="%{hovertext}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            name="Other",
        ))
        _hl = _scores_df[_scores_df["highlight"]]
        _hl_colors = ["#60A5FA","#F59E0B","#10B981","#EF4444",
                      "#A78BFA","#F472B6","#34D399","#FB923C",
                      "#38BDF8","#FBBF24"]
        for i, (cname, row) in enumerate(_hl.iterrows()):
            fig_j3.add_trace(go.Scatter(
                x=[row["PC1"]], y=[row["PC2"]], mode="markers+text",
                marker=dict(color=_hl_colors[i % len(_hl_colors)], size=13,
                            opacity=0.95, line=dict(width=1, color="#EEF4FB")),
                text=[cname], textposition="top center",
                textfont=dict(size=10, color="#EEF4FB"),
                name=cname,
                hovertemplate=f"{cname}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
            ))
        fig_j3.update_layout(**{**PLOTLY_LAYOUT,
            "title": (f"Country Structure — PC1 ({_var[0]*100:.1f}%) × "
                      f"PC2 ({_var[1]*100:.1f}%)"),
            "xaxis": dict(title=f"PC1 ({_var[0]*100:.1f}% variance)",
                          gridcolor="#304870", linecolor="#304870",
                          zeroline=True, zerolinecolor="#4E6D8C"),
            "yaxis": dict(title=f"PC2 ({_var[1]*100:.1f}% variance)",
                          gridcolor="#304870", linecolor="#304870",
                          zeroline=True, zerolinecolor="#4E6D8C"),
            "height": 520, "showlegend": False,
        })
        st.plotly_chart(fig_j3, width='stretch')

        thin_divider()

        # Loadings bar chart — PC1 and PC2 side by side
        section_label("J.3b · Indicator Contributions (Loadings)")
        _load_disp = _loadings[["PC1","PC2"]].copy()
        _load_disp.index = [n[:32]+"…" if len(n)>32 else n
                            for n in _load_disp.index]
        _load_disp = _load_disp.sort_values("PC1", ascending=True)

        fig_load = go.Figure()
        fig_load.add_trace(go.Bar(
            y=_load_disp.index, x=_load_disp["PC1"],
            orientation="h", name="PC1",
            marker_color=["#10B981" if v >= 0 else "#EF4444"
                          for v in _load_disp["PC1"]],
            opacity=0.85,
            text=[f"{v:+.2f}" for v in _load_disp["PC1"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_load.add_vline(x=0, line_color="#EEF4FB", line_width=1)
        fig_load.update_layout(**{**PLOTLY_LAYOUT,
            "title": f"PC1 Loadings ({_var[0]*100:.1f}% variance) — indicator contribution",
            "xaxis": dict(title="Loading", gridcolor="#304870", linecolor="#304870"),
            "margin": dict(l=240, r=80, t=50, b=40),
            "height": max(380, len(_load_disp)*35 + 80),
        })
        st.plotly_chart(fig_load, width='stretch')

        _load_pc2 = _loadings[["PC1","PC2"]].copy()
        _load_pc2.index = [n[:32]+"…" if len(n)>32 else n
                           for n in _load_pc2.index]
        _load_pc2 = _load_pc2.sort_values("PC2", ascending=True)
        fig_load2 = go.Figure()
        fig_load2.add_trace(go.Bar(
            y=_load_pc2.index, x=_load_pc2["PC2"],
            orientation="h", name="PC2",
            marker_color=["#60A5FA" if v >= 0 else "#F59E0B"
                          for v in _load_pc2["PC2"]],
            opacity=0.85,
            text=[f"{v:+.2f}" for v in _load_pc2["PC2"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_load2.add_vline(x=0, line_color="#EEF4FB", line_width=1)
        fig_load2.update_layout(**{**PLOTLY_LAYOUT,
            "title": f"PC2 Loadings ({_var[1]*100:.1f}% variance) — indicator contribution",
            "xaxis": dict(title="Loading", gridcolor="#304870", linecolor="#304870"),
            "margin": dict(l=240, r=80, t=50, b=40),
            "height": max(380, len(_load_pc2)*35 + 80),
        })
        st.plotly_chart(fig_load2, width='stretch')

        # Variance explained
        thin_divider()
        _npc = min(4, len(_var))
        fig_var = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(_npc)],
            y=[v*100 for v in _var[:_npc]],
            marker_color=["#60A5FA","#F59E0B","#10B981","#A78BFA"][:_npc],
            text=[f"{v*100:.1f}%" for v in _var[:_npc]],
            textposition="outside",
        ))
        fig_var.add_hline(y=sum(_var[:2])*100, line_dash="dash",
                          line_color="#EEF4FB", line_width=1,
                          annotation_text=f"PC1+PC2 = {sum(_var[:2])*100:.1f}%")
        fig_var.update_layout(**{**PLOTLY_LAYOUT,
            "title": "PCA — Variance Explained per Component",
            "yaxis": dict(title="%", gridcolor="#304870", linecolor="#304870"),
            "height": 280,
        })
        _, _vc, _ = st.columns([1, 2, 1])
        with _vc:
            st.plotly_chart(fig_var, width='stretch')
    else:
        st.info("Not enough data to run PCA.")




# ──────────────────────────────────────────────────────────────────────
# TAB 3 : CONVERGENCE
# ──────────────────────────────────────────────────────────────────────
# ── TAB 7 : CONVERGENCE ──────────────────────────────────────────
with tab3:
    section_label("K · Convergence / Divergence")

    _K_PRESETS = {
        "Developed": ["USA","Germany","France","Japan","United Kingdom",
                      "Canada","Australia","Netherlands","Sweden","Switzerland"],
        "Emerging":  ["China","India","Brazil","South Africa","Mexico",
                      "Indonesia","Turkey","Argentina","Thailand","Malaysia"],
        "Africa":    ["Nigeria","Ethiopia","Kenya","Ghana","Tanzania",
                      "Senegal","Morocco","Tunisia","Côte d'Ivoire","Rwanda"],
    }
    _avail_k = sorted(df_i["country_name"].dropna().unique().tolist())

    _kc1, _kc2 = st.columns(2)
    with _kc1:
        _grp_a_name = st.text_input("Group A label", "Developed", key="k_ga_name")
        _grp_a = st.multiselect(
            "Group A countries", _avail_k,
            default=[c for c in _K_PRESETS["Developed"] if c in _avail_k],
            key="k_ga",
        )
    with _kc2:
        _grp_b_name = st.text_input("Group B label", "Emerging", key="k_gb_name")
        _grp_b = st.multiselect(
            "Group B countries", _avail_k,
            default=[c for c in _K_PRESETS["Emerging"] if c in _avail_k],
            key="k_gb",
        )

    _all_k = list(set(_grp_a + _grp_b))
    if len(_all_k) < 4:
        st.info("Select at least 2 countries in each group.")
    else:
        thin_divider()

        # ── K.1 · Sigma-Convergence ──────────────────────────────
        section_label("K.1 · Sigma-Convergence — Intra-Group Dispersion")
        with st.expander("📖 What is Sigma-Convergence?", expanded=False):
            st.markdown("""
**Concept:** Sigma-convergence measures whether countries within a group are becoming *more similar* over time on this indicator.

It does so by computing the **standard deviation (dispersion)** of normalized values across countries, for each year.

| Dispersion trend | Interpretation |
|---|---|
| **Decreasing** ↓ | Countries are converging — differences shrink over time |
| **Increasing** ↑ | Countries are diverging — differences grow over time |
| **Flat** → | No clear trend — group remains structurally stable |
""")


        _df_k = df_i[df_i["country_name"].isin(_all_k)].copy()
        _df_k["group"] = _df_k["country_name"].apply(
            lambda c: _grp_a_name if c in _grp_a else _grp_b_name
        )
        # Use z_score if available, else normalize value
        if "z_score" in _df_k.columns and _df_k["z_score"].notna().sum() > 0:
            _sigma_col = "z_score"
        else:
            _mu = _df_k["value"].mean()
            _sd = _df_k["value"].std()
            _df_k["z_norm"] = (_df_k["value"] - _mu) / (_sd if _sd > 0 else 1)
            _sigma_col = "z_norm"

        _sigma = (_df_k.groupby(["year","group"])[_sigma_col]
                  .std().reset_index().rename(columns={_sigma_col: "dispersion"}))

        fig_k1 = go.Figure()
        _k_colors = {"Developed": "#60A5FA", "Emerging": "#F59E0B",
                     _grp_a_name: "#60A5FA", _grp_b_name: "#F59E0B"}
        for grp in [_grp_a_name, _grp_b_name]:
            _s = _sigma[_sigma["group"] == grp]
            _slope_v = linregress(_s["year"], _s["dispersion"]).slope if len(_s) > 2 else 0
            _trend = "converging ↓" if _slope_v < 0 else "diverging ↑"
            fig_k1.add_trace(go.Scatter(
                x=_s["year"], y=_s["dispersion"],
                mode="lines+markers",
                name=f"{grp} ({_trend})",
                line=dict(color=_k_colors.get(grp, "#A78BFA"), width=2.5),
                marker=dict(size=7),
            ))
        fig_k1.update_layout(**{**PLOTLY_LAYOUT,
            "title": f"Intra-Group Dispersion — {indicator[:45]}",
            "yaxis": dict(title="Std Dev (z-scores)",
                          gridcolor="#304870", linecolor="#304870"),
            "height": 380,
        })
        st.plotly_chart(fig_k1, width='stretch')
        st.caption("Decreasing dispersion = convergence within the group")

        thin_divider()

        # Inter-group mean gap
        section_label("K.1b · Inter-Group Gap")
        _means = (_df_k.groupby(["year","group"])[_sigma_col]
                  .mean().unstack("group").reset_index())
        if _grp_a_name in _means.columns and _grp_b_name in _means.columns:
            _means["gap"] = (_means[_grp_a_name] - _means[_grp_b_name]).abs()
            _gap0 = _means["gap"].iloc[0]
            _gap1 = _means["gap"].iloc[-1]
            _gap_chg = (_gap1 - _gap0) / _gap0 * 100 if _gap0 != 0 else 0

            fig_k1b = go.Figure()
            for grp, col in [(_grp_a_name, "#60A5FA"), (_grp_b_name, "#F59E0B")]:
                if grp in _means.columns:
                    fig_k1b.add_trace(go.Scatter(
                        x=_means["year"], y=_means[grp],
                        mode="lines+markers", name=grp,
                        line=dict(color=col, width=2.5), marker=dict(size=7),
                    ))
            fig_k1b.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text=f"Gap change: {_gap_chg:+.1f}%"
                     f" → {'narrowing ↓' if _gap_chg < 0 else 'widening ↑'}",
                showarrow=False, align="center",
                xanchor="center", yanchor="middle",
                font=dict(size=11, color="#EEF4FB"),
                bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=4,
            )
            fig_k1b.update_layout(**{**PLOTLY_LAYOUT,
                "title": "Inter-Group Mean Gap",
                "yaxis": dict(title="Mean z-score",
                              gridcolor="#304870", linecolor="#304870"),
                "height": 340,
            })
            st.plotly_chart(fig_k1b, width='stretch')

        with st.expander("📊 How to read the charts", expanded=False):
            st.markdown("""
- **Y-axis** = dispersion (std dev of z-scores). Lower = more homogeneous group.
- **Each line** = one group. A falling line = convergence within that group.
- **K.1 Intra** answers: *"Are countries within the same group becoming more alike?"*
- **K.1b Inter** answers: *"Is the gap between the two groups shrinking?"*
  - Converging groups = lines getting closer together; shrinking gap annotation.
""")

        thin_divider()

        # ── K.2 · Beta-Convergence ───────────────────────────────
        section_label("K.2 · Beta-Convergence — Initial Level vs Growth")
        with st.expander("📖 What is Beta-Convergence?", expanded=False):
            st.markdown("""
**Concept:** Beta-convergence tests the *catch-up hypothesis* — do countries that started at a lower level grow faster than those that started higher?

**Method:** A linear regression is fitted:
`Annualized Growth (CAGR) = α + β × Initial Level`

| β coefficient | Interpretation |
|---|---|
| **β < 0, p < 0.05** | ✅ Confirmed convergence — lower starters grow faster (catch-up) |
| **β < 0, p ≥ 0.05** | 🟡 Convergence tendency — but not statistically significant |
| **β > 0** | 🔴 Divergence — higher starters grow even faster (rich get richer) |

**Difference with Sigma:**
- Sigma asks: *"Are countries getting more similar?"* (dispersion over time)
- Beta asks: *"Do laggards grow faster?"* (level vs growth rate relationship)
- Both can coexist or contradict — sigma convergence without beta convergence is possible.
""")


        # initial level = earliest year z-score / value per country
        _yr_min = int(df_i["year"].min())
        _df_init = (df_i[df_i["year"] == _yr_min][["country_name", _sigma_col if _sigma_col == "z_score" else "value"]]
                    .rename(columns={_sigma_col if _sigma_col == "z_score" else "value": "initial"}))
        _df_beta = (df_gr_i[["country_name","cagr"]]
                    .merge(_df_init, on="country_name")
                    .dropna())
        _df_beta = _df_beta[_df_beta["country_name"].isin(_all_k)].copy()
        _df_beta["group"] = _df_beta["country_name"].apply(
            lambda c: _grp_a_name if c in _grp_a else _grp_b_name
        )

        if len(_df_beta) >= 5:
            _slope, _intercept, _r, _p, _se = linregress(
                _df_beta["initial"], _df_beta["cagr"] * 100
            )
            _x_line = np.linspace(_df_beta["initial"].min(),
                                  _df_beta["initial"].max(), 50)
            _y_line = _slope * _x_line + _intercept

            fig_k2 = go.Figure()
            for grp, col in [(_grp_a_name, "#60A5FA"), (_grp_b_name, "#F59E0B")]:
                _sub = _df_beta[_df_beta["group"] == grp]
                fig_k2.add_trace(go.Scatter(
                    x=_sub["initial"], y=_sub["cagr"] * 100,
                    mode="markers+text",
                    marker=dict(color=col, size=9, opacity=0.85,
                                line=dict(width=0.5, color="#304870")),
                    text=_sub["country_name"],
                    textposition="top center",
                    textfont=dict(size=9, color="#EEF4FB"),
                    name=grp,
                    hovertemplate="%{text}<br>Initial: %{x:.2f}<br>"
                                  "CAGR: %{y:.2f}%<extra></extra>",
                ))
            # Regression line
            _sig = "✅ significant" if _p < 0.05 else "⚠ not significant"
            _conv = "β<0 → convergence" if _slope < 0 else "β>0 → divergence"
            fig_k2.add_trace(go.Scatter(
                x=_x_line, y=_y_line, mode="lines",
                line=dict(color="#EEF4FB", width=1.5, dash="dash"),
                name=f"β = {_slope:+.4f} ({_sig})",
            ))
            fig_k2.add_annotation(
                x=0.98, y=0.98, xref="paper", yref="paper",
                text=(f"β = {_slope:+.4f}  |  p = {_p:.3f}  |  R² = {_r**2:.3f}<br>"
                      f"{_conv}  ·  {_sig}"),
                showarrow=False, align="right",
                xanchor="right", yanchor="top",
                font=dict(size=11, color="#EEF4FB"),
                bgcolor="#1F3050", bordercolor="#304870", borderwidth=1, borderpad=5,
            )
            fig_k2.update_layout(**{**PLOTLY_LAYOUT,
                "title": f"Beta-Convergence — {indicator[:45]}",
                "xaxis": dict(title=f"Initial level ({_yr_min})",
                              gridcolor="#304870", linecolor="#304870"),
                "yaxis": dict(title="Annualized CAGR (%)",
                              gridcolor="#304870", linecolor="#304870"),
                "height": 480,
            })
            st.plotly_chart(fig_k2, width='stretch')
        else:
            st.info("Not enough data for beta-convergence analysis.")

        with st.expander("📊 How to read the chart", expanded=False):
            st.markdown("""
- **X-axis** = initial level (z-score of the first available year). Negative = started below average; positive = above average.
- **Y-axis** = annualized CAGR over the full period (%).
- **Dashed line** = regression fit. A downward slope (β < 0) = catch-up dynamic.
- **Annotation box** = β value, p-value (significance), R² (fit quality).
- **Colors** = groups (Group A vs Group B).
""")

        thin_divider()

        # ── K.3 · Convergence Summary ────────────────────────────
        section_label("K.3 · Convergence Summary")

        _sum_rows = []
        for grp, countries in [(_grp_a_name, _grp_a), (_grp_b_name, _grp_b)]:
            _sg = _sigma[_sigma["group"] == grp]
            if len(_sg) > 2:
                _sl = linregress(_sg["year"], _sg["dispersion"]).slope
                _sum_rows.append({
                    "Level": "K.1 Sigma Intra",
                    "Group": grp,
                    "Result": "Converging ↓" if _sl < 0 else "Diverging ↑",
                    "Value": f"slope = {_sl:+.4f}",
                    "Signal": "🟢" if _sl < 0 else "🔴",
                })
        if _grp_a_name in _means.columns and _grp_b_name in _means.columns:
            _sum_rows.append({
                "Level": "K.1b Sigma Inter",
                "Group": f"{_grp_a_name} vs {_grp_b_name}",
                "Result": "Gap narrowing ↓" if _gap_chg < 0 else "Gap widening ↑",
                "Value": f"Δ = {_gap_chg:+.1f}%",
                "Signal": "🟢" if _gap_chg < 0 else "🔴",
            })
        if len(_df_beta) >= 5:
            _sum_rows.append({
                "Level": "K.2 Beta",
                "Group": "All selected",
                "Result": ("Convergence ↓" if _slope < 0 else "Divergence ↑")
                          + (f" ({'p<0.05' if _p < 0.05 else 'p≥0.05'})"),
                "Value": f"β = {_slope:+.4f}  R² = {_r**2:.3f}",
                "Signal": ("🟢" if _slope < 0 and _p < 0.05
                           else "🟡" if _slope < 0
                           else "🔴"),
            })
        if _sum_rows:
            _df_sum = pd.DataFrame(_sum_rows)
            st.dataframe(_df_sum, width='stretch', hide_index=True)

# ──────────────────────────────────────────────────────────────────────
# TAB 4 : L · CLUSTERING
# ──────────────────────────────────────────────────────────────────────
with tab4:
    section_label("L · Country Clustering")

    st.markdown(
        '<div style="background:#0D2137;border:1px solid #304870;'
        'border-left:4px solid #10B981;border-radius:0 12px 12px 0;'
        'padding:.65rem 1rem;margin-bottom:1rem;">'
        '<span style="color:#A8C0D8;font-size:13px;">'
        '<strong style="color:#10B981;">Dynamic clustering</strong> — '
        'KMeans fitted live on the selected feature set. '
        'Silhouette score selects optimal k.</span></div>',
        unsafe_allow_html=True)

    _cl_mode = st.radio(
        "Clustering mode",
        ["📊 Single Indicator", "🔀 Multi-Indicator", "🌐 PCA Space"],
        horizontal=True, key="cl_mode",
    )

    if _cl_mode == "📊 Single Indicator":
        _cl_ind = st.selectbox("Indicator", INDICATORS, key="cl_ind")
        _df_cl_input = df_growth[
            (df_growth["indicator_name"] == _cl_ind) &
            (df_growth["entity_type"] == "COUNTRY")
        ].dropna(subset=_CL_FEATURES).reset_index(drop=True)
        _feat_cols = _CL_FEATURES
        _pca_mode  = False

    elif _cl_mode == "🔀 Multi-Indicator":
        _multi_inds = st.multiselect(
            "Indicators (≥ 2)", INDICATORS,
            default=INDICATORS[:5], key="cl_multi_inds")
        if len(_multi_inds) < 2:
            st.warning("Select at least 2 indicators.")
            st.stop()
        _df_cl_raw = df_growth[
            (df_growth["indicator_name"].isin(_multi_inds)) &
            (df_growth["entity_type"] == "COUNTRY")
        ]
        _df_cl_input = (
            _df_cl_raw.groupby("country_name")[_CL_FEATURES]
            .mean().dropna().reset_index()
        )
        _df_cl_input["indicator_name"] = "multi"
        _feat_cols = _CL_FEATURES
        _pca_mode  = False

    else:
        _n_pc = st.slider("Number of PCs", 2, 5, 3, key="cl_n_pc")
        _piv  = (df_ind.pivot_table(
            index="country_name", columns="indicator_name",
            values="z_score", aggfunc="mean").fillna(0))
        _X_pca_raw  = StandardScaler().fit_transform(_piv.values)
        _pca_obj    = SkPCA(n_components=_n_pc)
        _pca_scores = _pca_obj.fit_transform(_X_pca_raw)
        _pc_cols    = [f"PC{i+1}" for i in range(_n_pc)]
        _df_cl_input = pd.DataFrame(
            _pca_scores, columns=_pc_cols, index=_piv.index,
        ).reset_index().rename(columns={"index": "country_name"})
        _df_cl_input["indicator_name"] = "pca"
        _feat_cols = _pc_cols
        _pca_mode  = True
        st.caption(" · ".join(
            f"PC{i+1}: {v:.1%}"
            for i, v in enumerate(_pca_obj.explained_variance_ratio_)))

    if _df_cl_input.empty:
        st.warning("No data for the selected configuration.")
        st.stop()

    _X_cl   = StandardScaler().fit_transform(_df_cl_input[_feat_cols].values)
    _X_hash = f"{_cl_mode}_{len(_df_cl_input)}_{_feat_cols}"

    _sil_scores, _best_k_sil = _best_k(_X_hash, _X_cl)

    _ctl1, _ctl2, _ctl3 = st.columns([2, 1, 1])
    with _ctl1:
        fig_sil = go.Figure(go.Bar(
            x=list(_sil_scores.keys()),
            y=list(_sil_scores.values()),
            marker_color=["#10B981" if k == _best_k_sil else "#304870"
                          for k in _sil_scores],
            text=[f"{v:.3f}" for v in _sil_scores.values()],
            textposition="outside",
        ))
        fig_sil.update_layout(**_pl(
            title="Silhouette Score by k",
            xaxis_title="k", yaxis_title="Silhouette",
            height=220, margin=dict(l=20, r=20, t=40, b=20)))
        st.plotly_chart(fig_sil, width='stretch')
    with _ctl2:
        st.markdown(kpi("Best k", str(_best_k_sil), "teal"),
                    unsafe_allow_html=True)
    with _ctl3:
        _k_choice = st.slider("k", 2, 8, _best_k_sil, key="cl_k")

    _labels = _run_kmeans(_X_hash, _X_cl, _k_choice)
    _df_cl_input = _df_cl_input.copy()
    _df_cl_input["cluster_id"] = _labels

    _cl_hl_opts    = [c for c in _df_cl_input["country_name"].tolist()
                      if c != country]
    _cl_hl_default = [c for c in ["China", "Germany", "USA"]
                      if c in _cl_hl_opts]
    _cl_hl = st.multiselect(
        "Highlight countries", _cl_hl_opts,
        default=_cl_hl_default, max_selections=5, key="cl_hl")

    _df_cl_input["_label"] = _df_cl_input["country_name"].apply(
        lambda x: x if x in _cl_hl else "")

    _cl_id_ser   = _df_cl_input.loc[
        _df_cl_input["country_name"] == country, "cluster_id"]
    _cl_id_val   = int(_cl_id_ser.iloc[0]) if not _cl_id_ser.empty else 0
    _cl_color_lv = CLUSTER_COLORS.get(_cl_id_val, "#6B7280")

    # ── L.1 Scatter ──────────────────────────────────────────────────────────
    section_label("L.1 · Cluster Scatter")
    _x_feat = _feat_cols[0]
    _y_feat = _feat_cols[1] if len(_feat_cols) > 1 else _feat_cols[0]

    fig_l1 = go.Figure()
    for cid in sorted(_df_cl_input["cluster_id"].unique()):
        df_q  = _df_cl_input[_df_cl_input["cluster_id"] == cid]
        _clr  = CLUSTER_COLORS.get(int(cid), "#6B7280")
        fig_l1.add_trace(go.Scatter(
            x=df_q[_x_feat], y=df_q[_y_feat],
            mode="markers+text", name=f"Cluster {cid}",
            marker=dict(color=_clr, size=8, opacity=0.6,
                        line=dict(width=0.5, color="#304870")),
            text=df_q["_label"],
            textposition="top center",
            textfont=dict(size=10, color="#EEF4FB"),
            customdata=df_q[["country_name"]],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
        ))

    _sel_row = _df_cl_input[_df_cl_input["country_name"] == country]
    if not _sel_row.empty:
        sc = _sel_row.iloc[0]
        fig_l1.add_trace(go.Scatter(
            x=[sc[_x_feat]], y=[sc[_y_feat]],
            mode="markers+text",
            marker=dict(color=_cl_color_lv, size=18, symbol="star",
                        line=dict(width=2, color="#EEF4FB")),
            text=[country], textposition="top center",
            textfont=dict(size=12, color="#EEF4FB"),
            showlegend=False, name=country,
        ))
    fig_l1.update_layout(**_pl(
        title=f"Clusters — {_x_feat} × {_y_feat}",
        xaxis_title=_x_feat, yaxis_title=_y_feat, height=480))
    st.plotly_chart(fig_l1, width='stretch')

    thin_divider()

    # ── L.2 + L.3 ────────────────────────────────────────────────────────────
    _radar_cols = [c for c in _feat_cols if c in _df_cl_input.columns]
    _lc1, _lc2  = st.columns(2)

    with _lc1:
        section_label("L.2 · Cluster Profile Radar")
        if _radar_cols:
            _cl_grp = (_df_cl_input.groupby("cluster_id")[_radar_cols]
                       .mean().reset_index())
            fig_l2 = go.Figure()
            for _, rw in _cl_grp.iterrows():
                cid  = int(rw["cluster_id"])
                clr  = CLUSTER_COLORS.get(cid, "#6B7280")
                vals = [rw[c] for c in _radar_cols]
                mn   = _cl_grp[_radar_cols].min()
                mx   = _cl_grp[_radar_cols].max()
                vn   = [(v - mn[c]) / (mx[c] - mn[c] + 1e-9)
                        for v, c in zip(vals, _radar_cols)]
                vn_cl  = vn + [vn[0]]
                lbl_cl = _radar_cols + [_radar_cols[0]]
                fig_l2.add_trace(go.Scatterpolar(
                    r=vn_cl, theta=lbl_cl, fill="toself",
                    name=f"Cluster {cid}",
                    line=dict(color=clr,
                              width=2.5 if cid == _cl_id_val else 1.5),
                    fillcolor=(f"rgba({int(clr[1:3],16)},"
                               f"{int(clr[3:5],16)},"
                               f"{int(clr[5:7],16)},"
                               f"{'0.3' if cid == _cl_id_val else '0.07'})"),
                ))
            fig_l2.update_layout(
                paper_bgcolor="#1F3050",
                font=dict(family="IBM Plex Sans", color="#C4D5E8", size=12),
                polar=dict(bgcolor="#162035",
                           radialaxis=dict(visible=True, range=[0, 1],
                                           gridcolor="#304870"),
                           angularaxis=dict(gridcolor="#304870")),
                legend=dict(bgcolor="#1F3050", bordercolor="#304870"),
                height=360, margin=dict(l=40, r=40, t=40, b=40),
                title=dict(text=f"Cluster {_cl_id_val} highlighted",
                           font=dict(color=_cl_color_lv)),
            )
            st.plotly_chart(fig_l2, width='stretch')

    with _lc2:
        section_label("L.3 · Feature Heatmap by Cluster")
        if _radar_cols:
            _hm_data = (_df_cl_input.groupby("cluster_id")[_radar_cols].mean())
            fig_l3 = go.Figure(go.Heatmap(
                z=_hm_data.values, x=_radar_cols,
                y=[f"Cluster {i}" for i in _hm_data.index],
                colorscale="RdYlGn", zmid=0,
                text=np.round(_hm_data.values, 2),
                texttemplate="%{text}", textfont=dict(size=10),
                showscale=True,
            ))
            fig_l3.update_layout(**_pl(
                title="Mean Feature Values by Cluster",
                height=280, margin=dict(l=40, r=40, t=40, b=60),
                xaxis_tickangle=-20))
            st.plotly_chart(fig_l3, width='stretch')

    # ── L.4 Interpretation ────────────────────────────────────────────────────
    section_label("L.4 · Cluster Interpretation")

    def _auto_name(row_dict):
        if row_dict.get("z_cagr", 0) > 0.5 and row_dict.get("volatility", 1) < 0.4:
            return "🟢 Stable Growth"
        if row_dict.get("z_cagr", 0) > 0.5:
            return "🟡 High Growth / Volatile"
        if row_dict.get("z_cagr", 0) < -0.5 and row_dict.get("volatility", 1) >= 0.4:
            return "🔴 Declining / Turbulent"
        if row_dict.get("z_cagr", 0) < -0.5:
            return "🟠 Stable Decline"
        if row_dict.get("mean_abs_z", 0) > 1.0:
            return "🔵 Strong Outlier"
        return "⚪ Mixed Profile"

    if _radar_cols:
        _l4_grp = _df_cl_input.groupby("cluster_id")[_radar_cols].mean()
        if _pca_mode:
            _pc_rows = []
            for cid, row_pc in _l4_grp.iterrows():
                _pc_rows.append({
                    "Cluster":     f"Cluster {int(cid)}",
                    "Dominant PC": row_pc.idxmax(),
                    **{c: f"{row_pc[c]:.3f}" for c in _radar_cols},
                })
            st.dataframe(pd.DataFrame(_pc_rows),
                         hide_index=True, width='stretch')
        else:
            _l4_cols = st.columns(min(_k_choice, 4))
            for i, (cid, row_l4) in enumerate(_l4_grp.iterrows()):
                _name = _auto_name(row_l4.to_dict())
                _clr  = CLUSTER_COLORS.get(int(cid), "#6B7280")
                _cnt  = len(_df_cl_input[_df_cl_input["cluster_id"] == cid])
                _l4_cols[i % len(_l4_cols)].markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-left:4px solid {_clr};border-radius:0 8px 8px 0;
            padding:.6rem .9rem;margin-bottom:.5rem;">
  <div style="font-size:12px;font-weight:700;color:{_clr};">
    Cluster {int(cid)}</div>
  <div style="font-size:13px;color:#EEF4FB;margin:.2rem 0;">{_name}</div>
  <div style="font-size:10px;color:#6890B0;">{_cnt} countries</div>
</div>""", unsafe_allow_html=True)

    # ── L.5 Members ──────────────────────────────────────────────────────────
    section_label(f"L.5 · Cluster {_cl_id_val} — Members")
    _members = (_df_cl_input[_df_cl_input["cluster_id"] == _cl_id_val]
                ["country_name"].sort_values().tolist())
    st.markdown(
        "<div style='font-size:13px;color:#A8C0D8;line-height:2;'>"
        + " · ".join(
            f"<strong style='color:#EEF4FB;'>{m}</strong>"
            if m == country else m
            for m in _members)
        + "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · M · FORECASTING
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# TAB 5 : M · FORECASTING
# ──────────────────────────────────────────────────────────────────────
with tab5:
    section_label("M · Forecasting — Country Indicator Projections")

    _m_tabs = st.tabs(["🔮 M.2 · Holt-Winters", "📐 M.3 · Regression"])

    # ── M.2 HOLT-WINTERS ─────────────────────────────────────────────────────
    with _m_tabs[0]:
        section_label("M.2 · Holt-Winters Exponential Smoothing")

        with st.expander("📖 What is Holt-Winters?"):
            st.markdown("""
**Holt-Winters (double exponential smoothing)** models a time series as two components:
- **Level** — the current baseline value, updated at each step
- **Trend** — the direction and speed of change (additive here: constant slope)

The model learns two smoothing weights (α for level, β for trend) automatically from the data.

**Reading the chart:**
| Element | Meaning |
|---|---|
| Blue line | Observed historical values |
| Purple dotted | Model fit on historical data |
| Orange dashed | Forecast (projected forward) |
| Shaded band | 95% confidence interval — built from 300 bootstrap resamples of residuals |
| Red × | Holdout prediction — the model was retrained on data minus the last 3 years, then tested on those years |

**When to trust the forecast:** narrow CI + low MAPE (< 10%) = reliable signal. Wide CI or MAPE > 20% means high uncertainty — treat as indicative, not predictive.
""")

        _hw_c1, _hw_c2 = st.columns([1, 3])
        with _hw_c1:
            _hw_ind = st.selectbox(
                "Indicator", INDICATORS, key="hw_ind",
                index=INDICATORS.index(_default_ind)
                      if _default_ind in INDICATORS else 0)
            _hw_h = st.slider("Horizon (years)", 1, 10, 5, key="hw_horizon")
            _hw_metric = st.radio(
                "Series", ["Value", "Index-100", "Z-score"], key="hw_metric")
            _hw_mc = {"Value": "value",
                      "Index-100": "index_100",
                      "Z-score": "z_score"}[_hw_metric]

        df_hw_raw = df_ind[
            (df_ind["country_name"] == country) &
            (df_ind["indicator_name"] == _hw_ind)
        ].sort_values("year")
        _hw_valid = df_hw_raw[_hw_mc].dropna()

        if len(_hw_valid) < 5:
            st.warning(f"Not enough data ({len(_hw_valid)} points — need ≥ 5).")
        else:
            _hw_yrs = df_hw_raw["year"].values[-len(_hw_valid):].astype(float)
            _hw_val = _hw_valid.values
            _hw_key = f"hw_{country}_{_hw_ind}_{_hw_metric}_{_hw_h}"
            _hw     = _hw_forecast(_hw_yrs, _hw_val, _hw_key, periods=_hw_h)

            with _hw_c2:
                fig_hw = go.Figure()
                fig_hw.add_trace(go.Scatter(
                    x=_hw["years"], y=_hw["values"],
                    mode="lines+markers", name="Historical",
                    line=dict(color="#60A5FA", width=2),
                    marker=dict(size=5),
                ))
                fig_hw.add_trace(go.Scatter(
                    x=_hw["years"], y=_hw["fitted"],
                    mode="lines", name="Fitted",
                    line=dict(color="#A78BFA", width=1.5, dash="dot"),
                ))
                fig_hw.add_trace(go.Scatter(
                    x=np.concatenate([_hw["fc_yrs"], _hw["fc_yrs"][::-1]]),
                    y=np.concatenate([_hw["ci_hi"], _hw["ci_lo"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(245,158,11,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% CI",
                ))
                fig_hw.add_trace(go.Scatter(
                    x=_hw["fc_yrs"], y=_hw["fc_vals"],
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#F59E0B", width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                ))
                if len(_hw["yrs_ev"]) > 0:
                    fig_hw.add_trace(go.Scatter(
                        x=_hw["yrs_ev"], y=_hw["y_pred_ev"],
                        mode="markers", name="Holdout pred",
                        marker=dict(color="#EF4444", size=9, symbol="x",
                                    line=dict(width=2)),
                    ))
                fig_hw.add_vline(
                    x=float(_hw["years"][-1]),
                    line_dash="dot", line_color="#EEF4FB",
                    annotation_text="Forecast →",
                    annotation_font=dict(color="#EEF4FB", size=10),
                )
                fig_hw.update_layout(**_pl(
                    title=f"{country} — {_hw_ind[:45]} ({_hw_metric})",
                    yaxis_title=_hw_metric, height=430))
                st.plotly_chart(fig_hw, width='stretch')

            # Eval KPIs
            _ev1, _ev2, _ev3 = st.columns(3)
            _ev1.markdown(kpi("MAE",
                f"{_hw['mae']:.3f}" if not np.isnan(_hw['mae']) else "—",
                "teal"), unsafe_allow_html=True)
            _ev2.markdown(kpi("RMSE",
                f"{_hw['rmse']:.3f}" if not np.isnan(_hw['rmse']) else "—",
                "blue"), unsafe_allow_html=True)
            _ev3.markdown(kpi("MAPE",
                f"{_hw['mape']:.1f}%" if not np.isnan(_hw['mape']) else "—",
                "amber"), unsafe_allow_html=True)

            with st.expander("📊 How to read these accuracy metrics"):
                st.markdown("""
| Metric | What it measures | Good range |
|---|---|---|
| **MAE** | Average absolute error in the holdout period — in the same unit as the series | < 5–10% of the series mean |
| **RMSE** | Like MAE but penalises large errors more — always ≥ MAE | Close to MAE = no large outlier errors |
| **MAPE** | Error as a % of the actual value — unit-free, comparable across indicators | < 10% = good · 10–20% = acceptable · > 20% = high uncertainty |

*Metrics are computed on the last 3 years withheld from training (holdout set).*
""")

            thin_divider()

            # Residuals
            section_label("Residuals")
            with st.expander("📖 What are residuals?"):
                st.markdown("""
A **residual** = actual value − fitted value.

- **Flat around zero** → the model captures the trend well; errors are random noise.
- **Systematic pattern** (U-shape, drift) → the model is missing a structural component (e.g. a shock, a cycle).
- **Outlier spike** → a specific year had an anomaly the model couldn't anticipate (crisis, policy change).

For forecasting, random residuals are desirable — they mean the remaining error is not predictable.
""")
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=_hw["years"], y=_hw["resid"],
                mode="lines+markers", name="Residual",
                line=dict(color="#F59E0B", width=1.5),
                marker=dict(size=5),
            ))
            fig_res.add_hline(y=0, line_dash="dash",
                              line_color="#EEF4FB", line_width=1)
            fig_res.update_layout(**_pl(
                title="Residuals (Historical − Fitted)",
                yaxis_title="Residual", height=240,
                margin=dict(l=40, r=40, t=40, b=30)))
            st.plotly_chart(fig_res, width='stretch')

            thin_divider()

            # Scenarios
            section_label("Scenarios")
            with st.expander("📖 What does the shock slider do?"):
                st.markdown("""
The **shock** applies a uniform percentage shift to the baseline forecast — it simulates a structural break that the model cannot foresee:

- **Positive shock (+%)**: e.g. a major investment programme, commodity price surge, reform dividend
- **Negative shock (−%)**: e.g. financial crisis, sanctions, sudden trade collapse

The shaded CI band stays anchored to the baseline; the scenario line shows where the indicator would land if the shock materialises at the start of the forecast horizon.

This is a *what-if* tool — not a probabilistic scenario from the model itself.
""")
            _sc_a, _sc_b = st.columns([1, 3])
            with _sc_a:
                _shock = st.slider(
                    "Shock (%)", -30, 30, 0, step=5, key="hw_shock")
            _sc_fc = _hw["fc_vals"] * (1 + _shock / 100)
            with _sc_b:
                # Context: last 3 historical years + forecast
                _ctx_n   = min(3, len(_hw["years"]))
                _ctx_yrs = _hw["years"][-_ctx_n:]
                _ctx_val = _hw["values"][-_ctx_n:]

                fig_sc = go.Figure()
                # Historical context
                fig_sc.add_trace(go.Scatter(
                    x=_ctx_yrs, y=_ctx_val,
                    mode="lines+markers", name="Historical (last 3 yrs)",
                    line=dict(color="#60A5FA", width=2),
                    marker=dict(size=6),
                ))
                # CI band
                fig_sc.add_trace(go.Scatter(
                    x=np.concatenate([_hw["fc_yrs"], _hw["fc_yrs"][::-1]]),
                    y=np.concatenate([_hw["ci_hi"], _hw["ci_lo"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(245,158,11,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% CI (baseline)",
                ))
                # Baseline forecast
                fig_sc.add_trace(go.Scatter(
                    x=_hw["fc_yrs"], y=_hw["fc_vals"],
                    mode="lines+markers", name="Baseline forecast",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                ))
                # Scenario line (only if shock != 0)
                if _shock != 0:
                    _sc_color = "#10B981" if _shock > 0 else "#EF4444"
                    fig_sc.add_trace(go.Scatter(
                        x=_hw["fc_yrs"], y=_sc_fc,
                        mode="lines+markers",
                        name=f"Scenario ({_shock:+d}%)",
                        line=dict(color=_sc_color, width=2.5),
                        marker=dict(size=7, symbol="circle"),
                    ))
                fig_sc.add_vline(
                    x=float(_hw["years"][-1]),
                    line_dash="dot", line_color="#EEF4FB",
                    annotation_text="Forecast →",
                    annotation_font=dict(color="#EEF4FB", size=10),
                )
                fig_sc.update_layout(**_pl(
                    title=f"Scenario — {_shock:+d}% shock applied",
                    yaxis_title=_hw_metric, height=360,
                    margin=dict(l=40, r=40, t=50, b=40),
                ))
                st.plotly_chart(fig_sc, width='stretch')

            thin_divider()

            # Multi-country comparison
            section_label("Multi-Country Comparison")
            _hw_peers = st.multiselect(
                "Compare with",
                [c for c in COUNTRIES if c != country],
                default=[c for c in ["Germany", "China"] if c in COUNTRIES],
                max_selections=4, key="hw_peers",
            )
            if _hw_peers:
                fig_hwp = go.Figure()
                for i, peer in enumerate([country] + _hw_peers):
                    df_peer = df_ind[
                        (df_ind["country_name"] == peer) &
                        (df_ind["indicator_name"] == _hw_ind)
                    ].sort_values("year")
                    _pv = df_peer[_hw_mc].dropna()
                    if len(_pv) < 5:
                        continue
                    _px  = df_peer["year"].values[-len(_pv):].astype(float)
                    _pfc = _hw_forecast(
                        _px, _pv.values,
                        f"hw_{peer}_{_hw_ind}_{_hw_metric}_{_hw_h}",
                        periods=_hw_h)
                    _pc = PALETTE2[i % len(PALETTE2)]
                    _lw = 2.5 if peer == country else 1.5
                    fig_hwp.add_trace(go.Scatter(
                        x=_pfc["years"], y=_pfc["values"],
                        mode="lines", name=peer,
                        line=dict(color=_pc, width=_lw),
                    ))
                    fig_hwp.add_trace(go.Scatter(
                        x=_pfc["fc_yrs"], y=_pfc["fc_vals"],
                        mode="lines", name=f"{peer} (fc)",
                        line=dict(color=_pc, width=_lw, dash="dash"),
                        showlegend=False,
                    ))
                fig_hwp.add_vline(
                    x=float(_hw["years"][-1]),
                    line_dash="dot", line_color="#EEF4FB")
                fig_hwp.update_layout(**_pl(
                    title=f"Multi-Country Forecast — {_hw_ind[:45]}",
                    yaxis_title=_hw_metric, height=400))
                st.plotly_chart(fig_hwp, width='stretch')

    # ── M.3 REGRESSION ───────────────────────────────────────────────────────
    with _m_tabs[1]:
        section_label("M.3 · Lag Regression — Time-Series Forecast")

        with st.expander("📖 How does the regression forecast work?"):
            st.markdown("""
**Method: Autoregressive Distributed Lag (ADL)**

The model is trained on the full panel of countries × historical years.
For each country-year pair (t), the features are **all indicator values at t−1** (one year lag),
and the target is the **indicator value at t**.

This means the model learns: *"given where all indicators were last year, where should the target be this year?"*

**Multi-step forecast (chaining):**
- **Year n+1**: uses last known values of all indicators as input → predicts target
- **Year n+2**: uses the predicted target at n+1 as its own lag; other predictors held constant at last known value
- **Year n+3**: same logic, chained forward

**Why hold other predictors constant?**
This is a conservative assumption — it isolates the target's own dynamics.
In practice, other indicators will also evolve, so the forecast should be read as a baseline trajectory.

**Structural Benchmark (expander below)** is a separate cross-sectional model
that answers a different question: *"where should this country be, given how similar countries perform?"*
""")

        _rg_c1, _rg_c2 = st.columns([1, 2])
        with _rg_c1:
            _rg_target = st.selectbox(
                "Target indicator", INDICATORS, key="rg_target",
                index=INDICATORS.index(_default_ind)
                      if _default_ind in INDICATORS else 0)
            _rg_preds = st.multiselect(
                "Predictor indicators",
                [i for i in INDICATORS if i != _rg_target],
                default=[i for i in INDICATORS if i != _rg_target][:4],
                max_selections=8, key="rg_preds",
            )
            _rg_horizon = st.slider("Forecast horizon (years)", 1, 8, 3,
                                    key="rg_horizon")

        if len(_rg_preds) < 1:
            st.warning("Select at least 1 predictor.")
        else:
            _lag_cache = f"lag_{_rg_target}_{str(sorted(_rg_preds))}"
            _lag_res   = _run_lag_regression(
                _rg_target, tuple(_rg_preds), _lag_cache, df_ind)

            if _lag_res is None:
                st.warning("Not enough data (need ≥ 20 country-year observations).")
            else:
                # ── Model quality ─────────────────────────────────────────────
                with _rg_c2:
                    _r2_df = pd.DataFrame([
                        {"Model": k, "R² Test": v["R2_test"], "R² CV": v["R2_cv"]}
                        for k, v in _lag_res["results"].items()
                    ]).sort_values("R² CV", ascending=False)
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Bar(
                        name="R² Test", x=_r2_df["Model"], y=_r2_df["R² Test"],
                        marker_color="#60A5FA", opacity=0.85))
                    fig_r2.add_trace(go.Bar(
                        name="R² CV (5-fold)", x=_r2_df["Model"], y=_r2_df["R² CV"],
                        marker_color="#10B981", opacity=0.85))
                    fig_r2.update_layout(**_pl(
                        title=f"Model Comparison — R² · {_rg_target[:38]}",
                        yaxis_title="R²", barmode="group", height=300))
                    st.plotly_chart(fig_r2, width='stretch')

                _best = _lag_res["best_name"]
                _bres = _lag_res["results"][_best]
                _bc1, _bc2, _bc3, _bc4 = st.columns(4)
                _bc1.markdown(kpi("Best Model", _best[:12], "teal"),
                              unsafe_allow_html=True)
                _bc2.markdown(kpi("R² CV", f"{_bres['R2_cv']:.3f}", "blue"),
                              unsafe_allow_html=True)
                _bc3.markdown(kpi("MAE", f"{_bres['MAE']:.3f}", "amber"),
                              unsafe_allow_html=True)
                _bc4.markdown(kpi("Obs (panel)", str(_lag_res["n_obs"]), ""),
                              unsafe_allow_html=True)

                thin_divider()

                # ── Feature Importance ────────────────────────────────────────
                section_label(f"{_lag_res['imp_label']} — {_best}")
                _imp_sorted = sorted(
                    zip(_lag_res["feat_cols"], _lag_res["importances"]),
                    key=lambda x: abs(x[1]), reverse=True)
                _imp_lbl = [n[:40] + "…" if len(n) > 40 else n
                            for n, _ in _imp_sorted]
                _imp_val = [v for _, v in _imp_sorted]

                fig_imp = go.Figure(go.Bar(
                    x=_imp_val, y=_imp_lbl, orientation="h",
                    marker_color=["#10B981" if v >= 0 else "#EF4444"
                                  for v in _imp_val],
                    opacity=0.85,
                ))
                fig_imp.update_layout(**_pl(
                    title=f"{_lag_res['imp_label']} — {_best} (lag features at t−1)",
                    xaxis_title=_lag_res["imp_label"],
                    height=max(260, len(_imp_lbl) * 30 + 80),
                    margin=dict(l=220, r=40, t=50, b=40)))
                st.plotly_chart(fig_imp, width='stretch')

                thin_divider()

                # ── Forecast for selected country ─────────────────────────────
                section_label(f"Forecast — {country} · {_rg_target[:40]}")

                # Get most recent non-null value for each feature (lag inputs)
                _latest_vals      = {}
                _latest_val_years = {}
                for feat in _lag_res["feat_cols"]:
                    _s = (df_ind[
                            (df_ind["country_name"] == country) &
                            (df_ind["indicator_name"] == feat)
                          ]
                          .sort_values("year")
                          .dropna(subset=["value"]))      # skip NaN years
                    if not _s.empty:
                        _latest_vals[feat]      = float(_s["value"].iloc[-1])
                        _latest_val_years[feat] = int(_s["year"].iloc[-1])

                _missing_feats = [f for f in _lag_res["feat_cols"]
                                  if f not in _latest_vals]

                if _missing_feats:
                    st.warning(
                        f"No data at all for {len(_missing_feats)} feature(s) "
                        f"— cannot generate forecast:\n\n"
                        + "\n".join(f"- {f}" for f in _missing_feats)
                    )
                    st.info(
                        "Tip: remove these indicators from the predictor list, "
                        "or choose a country with complete data."
                    )
                else:
                    # Show which year each feature was last observed
                    _stale = {f: y for f, y in _latest_val_years.items()
                              if y < max(_latest_val_years.values())}
                    if _stale:
                        st.caption(
                            "⚠ Some features use older data (held constant from "
                            "their last known year): "
                            + ", ".join(f"**{f}** ({y})"
                                        for f, y in sorted(
                                            _stale.items(), key=lambda x: x[1]))
                        )

                    # Get last known year for the target
                    _hist_target = df_ind[
                        (df_ind["country_name"] == country) &
                        (df_ind["indicator_name"] == _rg_target)
                    ].sort_values("year")
                    _last_year = int(_hist_target["year"].iloc[-1]) \
                                 if not _hist_target.empty else 2023

                    # Chain predictions
                    _X_curr  = np.array([_latest_vals[f]
                                         for f in _lag_res["feat_cols"]])
                    _fc_yrs  = []
                    _fc_vals = []
                    _target_feat_idx = (_lag_res["feat_cols"].index(_rg_target)
                                        if _rg_target in _lag_res["feat_cols"]
                                        else None)

                    for step in range(_rg_horizon):
                        _X_sc_step = _lag_res["scaler"].transform([_X_curr])
                        _y_step    = float(_lag_res["model"].predict(_X_sc_step)[0])
                        _fc_yrs.append(_last_year + step + 1)
                        _fc_vals.append(_y_step)
                        # Update lag: only the target column is updated; others constant
                        if _target_feat_idx is not None:
                            _X_curr[_target_feat_idx] = _y_step

                    # Historical series for this country
                    _hist_yrs = _hist_target["year"].values
                    _hist_val = _hist_target["value"].values

                    # Historical fitted values (from lag model)
                    _hist_fit_c = _lag_res["hist_fit"].get(country, {})

                    fig_fc = go.Figure()
                    # Historical
                    fig_fc.add_trace(go.Scatter(
                        x=_hist_yrs, y=_hist_val,
                        mode="lines+markers", name="Historical",
                        line=dict(color="#60A5FA", width=2),
                        marker=dict(size=5),
                    ))
                    # Model fitted on historical
                    if _hist_fit_c:
                        fig_fc.add_trace(go.Scatter(
                            x=_hist_fit_c["years"], y=_hist_fit_c["fitted"],
                            mode="lines", name="Model fit (historical)",
                            line=dict(color="#A78BFA", width=1.5, dash="dot"),
                        ))
                    # Forecast
                    # Bridge: connect last historical point to first forecast
                    _bridge_x = [float(_hist_yrs[-1])] + _fc_yrs
                    _bridge_y = [float(_hist_val[-1])] + _fc_vals
                    fig_fc.add_trace(go.Scatter(
                        x=_bridge_x, y=_bridge_y,
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#F59E0B", width=2.5, dash="dash"),
                        marker=dict(size=8, symbol="diamond"),
                    ))
                    fig_fc.add_vline(
                        x=float(_hist_yrs[-1]),
                        line_dash="dot", line_color="#EEF4FB",
                        annotation_text="Forecast →",
                        annotation_font=dict(color="#EEF4FB", size=10),
                    )
                    fig_fc.update_layout(**_pl(
                        title=f"{country} — {_rg_target[:45]} · Lag Regression Forecast",
                        yaxis_title="Value", height=420))
                    st.plotly_chart(fig_fc, width='stretch')

                    # Forecast table
                    st.dataframe(
                        pd.DataFrame({
                            "Year":     _fc_yrs,
                            "Predicted": [f"{v:.3f}" for v in _fc_vals],
                        }),
                        hide_index=True, width='content')

                thin_divider()

                # ── Structural Benchmark (cross-sectional, demoted) ───────────
                with st.expander("📊 Structural Benchmark (cross-country model)"):
                    st.markdown(
                        "This model is trained on **country averages** (not time series). "
                        "It answers: *where should this country be, given how similar countries perform?* "
                        "It is not a time forecast.")

                    _rg_cache = f"reg_{_rg_target}_{str(sorted(_rg_preds))}"
                    _rg_res   = _run_regression(
                        _rg_target, tuple(_rg_preds), _rg_cache, df_ind)

                    if _rg_res is None:
                        st.info("Not enough cross-country data.")
                    else:
                        _countries_rg = _rg_res["countries"]
                        _y_act        = _rg_res["y_actual"]
                        _y_pred_cs    = _rg_res["y_pred_all"]
                        _cidx         = (_countries_rg.index(country)
                                         if country in _countries_rg else None)

                        fig_bench = go.Figure()
                        _dmin = min(float(np.min(_y_act)), float(np.min(_y_pred_cs)))
                        _dmax = max(float(np.max(_y_act)), float(np.max(_y_pred_cs)))
                        fig_bench.add_trace(go.Scatter(
                            x=[_dmin, _dmax], y=[_dmin, _dmax],
                            mode="lines", name="Perfect fit",
                            line=dict(color="#6B7280", width=1, dash="dot")))
                        _bg = [i for i, c in enumerate(_countries_rg) if c != country]
                        fig_bench.add_trace(go.Scatter(
                            x=[_y_act[i] for i in _bg],
                            y=[_y_pred_cs[i] for i in _bg],
                            mode="markers", name="Other countries",
                            marker=dict(color="#304870", size=6, opacity=0.6),
                            customdata=[[_countries_rg[i]] for i in _bg],
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "Actual: %{x:.2f}<br>Benchmark: %{y:.2f}"
                                "<extra></extra>")))
                        if _cidx is not None:
                            fig_bench.add_trace(go.Scatter(
                                x=[float(_y_act[_cidx])],
                                y=[float(_y_pred_cs[_cidx])],
                                mode="markers+text", name=country,
                                marker=dict(color="#EEF4FB", size=16,
                                            symbol="star",
                                            line=dict(width=2, color="#F59E0B")),
                                text=[country], textposition="top center",
                                textfont=dict(size=12, color="#EEF4FB")))
                        fig_bench.update_layout(**_pl(
                            title=f"Structural Benchmark — {_rg_target[:40]}",
                            xaxis_title="Actual (country mean)",
                            yaxis_title="Benchmark estimate", height=380))
                        st.plotly_chart(fig_bench, width='stretch')

                        if _cidx is not None:
                            _av = float(_y_act[_cidx])
                            _pv = float(_y_pred_cs[_cidx])
                            _gp = _pv - _av
                            _gp_pct = _gp / (abs(_av) + 1e-9) * 100
                            _gl = (f"+{_gp_pct:.1f}% — model expects more"
                                   if _gp > 0 else
                                   f"{_gp_pct:.1f}% — outperforms peers")
                            _bk1, _bk2, _bk3 = st.columns(3)
                            _bk1.markdown(kpi("Actual", f"{_av:.2f}", "blue"),
                                          unsafe_allow_html=True)
                            _bk2.markdown(kpi("Benchmark", f"{_pv:.2f}", "amber"),
                                          unsafe_allow_html=True)
                            _bk3.markdown(kpi("Gap", _gl[:20], "teal"),
                                          unsafe_allow_html=True)

                with st.expander("All model metrics (lag model)"):
                    st.dataframe(pd.DataFrame([
                        {"Model":   name,
                         "R² Test": f"{v['R2_test']:.3f}",
                         "R² CV":   f"{v['R2_cv']:.3f}",
                         "MAE":     f"{v['MAE']:.3f}",
                         "RMSE":    f"{v['RMSE']:.3f}"}
                        for name, v in _lag_res["results"].items()
                    ]), hide_index=True, width='stretch')

