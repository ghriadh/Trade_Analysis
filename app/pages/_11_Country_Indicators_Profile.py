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

inject_styles()

IND_FILES = ROOT / "Analysis" / "Analysis_files" / "Indicators"

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

st.markdown('<div class="page-eyebrow">Indicators Analysis · Page 3 of 4</div>',
            unsafe_allow_html=True)
st.title("Country Indicator Profile")
st.caption("G · H · N — Time series · comparison · multi-score")
thin_divider()

# ════════════════════════════════════════════════════════════════
# COUNTRY SELECTOR + KPIs
# ════════════════════════════════════════════════════════════════

col_sel, col_kpi = st.columns([1, 3])
with col_sel:
    default_idx = COUNTRIES.index("France") if "France" in COUNTRIES else 0
    country = st.selectbox("Select country", COUNTRIES,
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

def _latest_val(ind_name, col="value"):
    sub = (df_ind[(df_ind["country_name"] == country) &
                  (df_ind["indicator_name"] == ind_name)]
           .dropna(subset=[col]).sort_values("year"))
    if sub.empty:
        return None, None
    row = sub.iloc[-1]
    return row[col], int(row["year"])

def _fmt_kpi(label, val, yr, col):
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
        f'<div class="kpi-label">{label}</div>'
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
    "📈 G · Time Series",
    "🔄 H · Comparison",
    "🏅 N · Multi-Score",
])


# ── TAB 1 : G · TIME SERIES ─────────────────────────────────────────────────────
with tab1:
    section_label("G · Country Time Series — All Indicators")

    df_c = df_ind[df_ind["country_name"] == country].copy()

    inds_sel = st.multiselect(
        "Indicators", options=INDICATORS,
        default=[_default_ind], max_selections=6, key="ts_indicators")

    col1, col2 = st.columns([1, 3])
    with col1:
        metric_ts = st.radio(
            "View", ["Value", "YoY Growth %", "Index-100", "Z-score"],
            key="ts_metric")
        if len(inds_sel) == 1:
            _brk_thr = st.slider(
                "Break threshold", min_value=0.03, max_value=0.20,
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
                             annotation_text="2015 = 100")
        fig_ts.update_layout(**_pl(
            title=f"{country} — {metric_ts} by Indicator",
            yaxis_title=metric_ts, height=450))
        st.plotly_chart(fig_ts, width='stretch')
        if metric_ts == "Value" and len(inds_sel) > 1:
            st.caption(
                "💡 Indicators with different units (%, USD…) share the same axis. "
                "Switch to **Index-100** or **Z-score** for a normalized multi-indicator view."
            )

    # G.4 Indicator Dynamics
    section_label("G.4 · Indicator Dynamics — CAGR & Volatility")
    df_grw_c = df_growth[
        (df_growth["country_name"] == country) &
        (df_growth["entity_type"] == "COUNTRY")
    ].copy()

    if df_grw_c.empty:
        st.info("No growth data available for this country.")
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
                x=_df_cagr_ok["_short"], y=_df_cagr_ok["cagr"], name="CAGR",
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
                name="Volatility", mode="markers", yaxis="y2",
                marker=dict(color="#F59E0B", size=10, symbol="diamond"),
            ))
        fig_dyn.update_layout(**_pl(
            title=f"{country} — CAGR by Indicator",
            yaxis_title="CAGR",
            yaxis2=dict(title="Volatility", overlaying="y", side="right",
                        showgrid=False, color="#F59E0B"),
            height=460, xaxis_tickangle=-30,
        ))
        st.plotly_chart(fig_dyn, width='stretch')
        if not _df_cagr_nan.empty:
            _nan_names = " · ".join(_df_cagr_nan["indicator_name"].tolist())
            st.caption(
                f"⚠️ CAGR undefined for: **{_nan_names}** — "
                "values change sign over the period (no real-valued compound growth rate). "
                "Volatility is still shown."
            )

    # G.5 Z-Score Heatmap
    section_label("G.5 · Z-Score Heatmap — All Indicators × Years")
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
        title=f"{country} — Z-Score Heatmap (green = above world mean)",
        xaxis_title="Year",
        height=max(380, len(df_hm) * 32 + 80),
    ))
    st.plotly_chart(fig_hm, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · H · COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

# ── TAB 2 : H · COMPARISON ──────────────────────────────────────────────────────
with tab2:
    section_label("H · Cross-Country Comparison")

    indicator_comp = st.selectbox(
        "Indicator", INDICATORS, index=0, key="comp_ind")
    peers = st.multiselect(
        "Compare with",
        [c for c in COUNTRIES if c != country],
        default=[c for c in ["China", "Germany", "USA", "Japan"]
                 if c in COUNTRIES],
        max_selections=7, key="comp_peers",
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        comp_metric = st.radio(
            "Metric", ["Index-100", "Z-score", "Value"], key="comp_metric")

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
                               line_dash="dash", annotation_text="Base 2015")
        if comp_metric == "Z-score":
            fig_comp.add_hline(y=0, line_color="#304870",
                               line_dash="dash", annotation_text="World mean")
        fig_comp.update_layout(**_pl(
            title=f"{indicator_comp[:50]} — {comp_metric}",
            yaxis_title=comp_metric, height=450))
        st.plotly_chart(fig_comp, width='stretch')

    year_snap = st.selectbox("Snapshot year", YEARS[::-1], index=0, key="snap_year")
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
        title=f"{comp_metric} Snapshot — {year_snap}",
        xaxis_title=comp_metric))
    st.plotly_chart(fig_snap, width='stretch')

    # H.3 CAGR vs World
    section_label("H.3 · Country vs World — CAGR per Indicator")
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
            name="World", x=df_vs["_short"], y=df_vs["world_cagr"],
            marker_color="#F59E0B", opacity=0.70))
        fig_cagr.update_layout(**_pl(
            title=f"CAGR — {country} vs World",
            yaxis_title="CAGR", barmode="group",
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
            title="Outperformance vs World (CAGR delta)",
            yaxis_title="CAGR Delta", height=520, xaxis_tickangle=-30,
            margin=dict(l=40, r=40, t=50, b=140)))
        st.plotly_chart(fig_out, width='stretch')
    else:
        st.info("Growth data not available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · N · MULTI-SCORE
# ─────────────────────────────────────────────────────────────────────────────

# ── TAB 3 : N · MULTI-SCORE ─────────────────────────────────────────────────────
with tab3:
    section_label("N · Multi-Indicator Scoring")

    if row_m.empty:
        st.warning(f"No multi-score data for {country}.")
    else:
        r = row_m.iloc[0]

        # ── Expander: formulas & weights ──────────────────────────────────────
        with st.expander("📐 Score definitions, formulas & weights"):
            st.markdown("""
**Double reading** — each pillar is scored twice:
- **Value Position**: weighted average of normalized indicator *values* → current structural level
- **Growth Momentum**: weighted average of normalized *growth scores* → directional trajectory

Each score is min-max normalized to **0–100** across all countries. Rank 1 = best.
""")
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                st.markdown("""
**Economic Health**
| Indicator | Weight |
|---|---|
| GDP per capita growth | +0.25 |
| Unemployment rate | −0.20 |
| Inflation (CPI) | −0.20 |
| Current account | +0.15 |
| Real interest rate | +0.10 |
| Gov. expenditure | +0.10 |
""")
            with _fc2:
                st.markdown("""
**Performance**
| Indicator | Weight |
|---|---|
| GDP growth | +0.30 |
| Unemployment rate | −0.20 |
| Inflation (CPI) | −0.15 |
| Current account | +0.15 |
| Trade balance | +0.10 |
| Real interest rate | +0.10 |
""")
            with _fc3:
                st.markdown("""
**Competitiveness**
| Indicator | Weight |
|---|---|
| Trade % GDP | +0.30 |
| Current account | +0.25 |
| GDP per capita | +0.20 |
| Inflation (CPI) | −0.15 |
| Unemployment | −0.10 |
""")
            st.markdown("""
---
**Quadrant interpretation** (scatter):
- **Top-right**: high value *and* high growth → strong & accelerating
- **Top-left**: low value, high growth → catching up
- **Bottom-right**: high value, low growth → established but slowing
- **Bottom-left**: low value, low growth → structural challenge

**Rank Divergence** — two complementary views:
- *Intra-dimension* (dumbbell chart): gap between Value Rank and Growth Rank **within** the same pillar
- *Inter-dimension* (KPI cards): spread of ranks **across** the 3 pillars (from `rank_div_value / rank_div_growth`)
""")

        thin_divider()

        # ── Row 1: Value Position ─────────────────────────────────────────────
        st.markdown(
            '<div style="background:#0F2035;border-left:4px solid #10B981;'
            'border-radius:0 8px 8px 0;padding:.5rem 1rem;margin-bottom:.6rem;">'
            '<span style="color:#10B981;font-weight:700;font-size:13px;'
            'text-transform:uppercase;letter-spacing:.06em;">▸ Value Position</span>'
            '&nbsp;&nbsp;<span style="color:#6890B0;font-size:12px;">'
            'Current structural level — where the country stands today</span>'
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
  <div style="font-size:11px;color:#6890B0;margin-top:4px;">Rank #{v_rank}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:.75rem;'></div>", unsafe_allow_html=True)

        # ── Row 2: Growth Momentum ────────────────────────────────────────────
        st.markdown(
            '<div style="background:#0F2035;border-left:4px solid #60A5FA;'
            'border-radius:0 8px 8px 0;padding:.5rem 1rem;margin-bottom:.6rem;">'
            '<span style="color:#60A5FA;font-weight:700;font-size:13px;'
            'text-transform:uppercase;letter-spacing:.06em;">▸ Growth Momentum</span>'
            '&nbsp;&nbsp;<span style="color:#6890B0;font-size:12px;">'
            'Directional score — where the country is heading</span>'
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
  <div style="font-size:11px;color:#6890B0;margin-top:4px;">Rank #{gr_rank}</div>
</div>""", unsafe_allow_html=True)

        thin_divider()

        # ── Radar ─────────────────────────────────────────────────────────────
        _rc1, _rc2, _rc3 = st.columns([1, 2, 1])
        with _rc2:
            fig_rad = go.Figure()
            for sfx, clr, name in [
                ("_value_norm",  "#10B981", "Value Position"),
                ("_growth_norm", "#60A5FA", "Growth Momentum"),
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
                title=dict(text=f"{country} — Score Radar",
                           font=dict(color="#EEF4FB")),
            )
            st.plotly_chart(fig_rad, width='stretch')

        thin_divider()

        # ── Scatter with axis selectors ───────────────────────────────────────
        section_label("Global Ranking — All Countries")

        _score_opts = {
            **{f"{d} · Value":  f"{d}_value_norm"  for d in DIMS},
            **{f"{d} · Growth": f"{d}_growth_norm" for d in DIMS},
        }
        _slabels = list(_score_opts.keys())

        _sc1, _sc2, _sc3 = st.columns([3, 1.5, 1.5])
        with _sc1:
            highlight_m = st.multiselect(
                "Highlight countries",
                [c for c in df_multi["country_name"].tolist() if c != country],
                default=[c for c in ["China", "Germany", "USA"]
                         if c in df_multi["country_name"].values],
                max_selections=5, key="multi_hl",
            )
        with _sc2:
            _x_label = st.selectbox("X axis", _slabels, index=0, key="n_x_axis")
        with _sc3:
            _y_label = st.selectbox("Y axis", _slabels, index=3, key="n_y_axis")

        _xc       = _score_opts[_x_label]
        _yc       = _score_opts[_y_label]
        _other    = [l for l in _slabels if l not in (_x_label, _y_label)]
        _zc       = _score_opts[_other[0]] if _other else _xc
        _z_label  = _other[0] if _other else _x_label

        if _x_label == _y_label:
            st.warning("Select different axes.")
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
                title=f"{_x_label} × {_y_label}",
                xaxis_title=_x_label, yaxis_title=_y_label, height=420))
            st.plotly_chart(fig_sc, width='stretch')

        thin_divider()

        # ── Dumbbell: Rank Divergence ─────────────────────────────────────────
        section_label("Rank Divergence — Value vs Growth")

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
            name="Value Rank",
            marker=dict(color="#10B981", size=14, symbol="circle",
                        line=dict(width=1, color="#EEF4FB")),
            text=[f"#{v}" for v in _vr_list],
            textposition="top center",
            textfont=dict(size=10, color="#EEF4FB"),
        ))
        fig_db.add_trace(go.Scatter(
            x=_gr_list, y=_db_dims, mode="markers+text",
            name="Growth Rank",
            marker=dict(color="#60A5FA", size=14, symbol="diamond",
                        line=dict(width=1, color="#EEF4FB")),
            text=[f"#{v}" for v in _gr_list],
            textposition="bottom center",
            textfont=dict(size=10, color="#EEF4FB"),
        ))
        fig_db.update_layout(**PLOTLY_LAYOUT)
        fig_db.update_layout(
            title="Intra-dimension: Value Rank vs Growth Rank (left = better)",
            xaxis=dict(title="Rank (1 = best)", autorange="reversed",
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
    Inter-Dim Spread · Value</div>
  <div style="font-size:28px;font-weight:700;color:#10B981;">{rd_val}</div>
  <div style="font-size:11px;color:#6890B0;">rank-point spread across 3 pillars</div>
</div>""", unsafe_allow_html=True)
        _dk2.markdown(f"""
<div style="background:#1F3050;border:1px solid #304870;
            border-top:3px solid #60A5FA;border-radius:0 0 12px 12px;
            padding:1rem;text-align:center;">
  <div style="font-size:10px;color:#6890B0;text-transform:uppercase;
              letter-spacing:.06em;margin-bottom:.3rem;">
    Inter-Dim Spread · Growth</div>
  <div style="font-size:28px;font-weight:700;color:#60A5FA;">{rd_gr}</div>
  <div style="font-size:11px;color:#6890B0;">rank-point spread across 3 pillars</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · L · CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
