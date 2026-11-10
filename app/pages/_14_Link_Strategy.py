"""
Trade × Indicators — Link · Strategy & Conclusions
Sections L · M · N — Fingerprint, Prediction, Direction 2
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
from scipy.stats import spearmanr
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, fmt_usd
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Link · Trade × Indicators · Page 2 of 2",
        "title": "Strategy & Conclusions",
        "caption": (
            "Sections L · M · N — Macro fingerprint by profile, prediction attempt, "
            "and reversal of the direction of analysis"
        ),
        "tab_l": "L · Macro Fingerprint",
        "tab_m": "M · Prediction Attempt",
        "tab_n": "N · Direction 2",
        "tab_p": "P · Target Country",
        "tab_syn": "Summary",

        "l_label": "L · Macro Fingerprint by Trade Profile",
        "l_intro": (
            "Idea: does each trade profile have a <em>characteristic macro fingerprint</em>? "
            "Indicators normalized as percentile (0=worst, 1=best in dataset). "
            "For Inflation and Unemployment: inverted percentile (lower = better)."
        ),
        "l_rationale": (
            '<strong style="color:#34D399;">Rationale for the 6 indicators</strong>: '
            'GDP/capita and CA% are the two dominant [E] signals (sections G–K). '
            'Inflation, Unemployment, Real Rate complete the cyclical macro picture. '
            'Exports/GDP (indicator [T]) is included only to visualize '
            'trade openness — not to establish causality.'
        ),
        "radar_gdp_cap": "GDP/cap",
        "radar_cur_acct": "Cur. Acct",
        "radar_inflation": "Inflation",
        "radar_unemployment": "Unemployment",
        "radar_real_rate": "Real Rate",
        "radar_exports_gdp": "Exports/GDP",
        "profile_dim_label": "Profile dimension",
        "profile_combined": "Combined Profile",
        "profile_meta": "Meta Profile",
        "profile_structural": "Structural Profile",
        "percentile_hover": "<b>%{theta}</b><br>Percentile = %{r:.2f}<extra></extra>",
        "l_headline_out_of": "out of 5",
        "l_headline_title": "macro indicator discriminates trade profiles",
        "l_headline_body": (
            '<strong style="color:#34D399;">GDP / capita</strong> — clear and '
            'significant differences between profiles (KW p&nbsp;&lt;&nbsp;0.001).<br>'
            '<span style="color:#6890B0;">Inflation · Unemployment · Real Rate · CA%</span> — '
            'weak or non-significant differences between profiles.<br>'
            '<em style="color:#7090B0;">Across these 6 macro indicators and 10 years of data: '
            'the dominant signal is wealth. With more sectoral indicators '
            'or longer data, other dimensions could emerge.</em>'
        ),
        "l_how_to_read_expander": "How to read the chart — visualization limits",
        "l_how_to_read_html": """
        <div style='font-size:13px;color:#A8C0D8;line-height:1.8;'>
        <strong>Axis</strong> = normalized percentile of the indicator (0 = worst country in dataset, 1 = best).<br>
        <strong>Larger area</strong> = macro profile globally better on these 6 dimensions.<br>
        <strong>Inflation & Unemployment</strong>: inverted percentile — a country with low inflation/unemployment appears in a high position.<br>
        <br>
        <strong style='color:#F59E0B;'>This is not truly a radar in the classical sense.</strong><br>
        A real radar assumes <em>independent</em> axes: each dimension brings distinct information
        and profiles form polygons with truly different shapes.
        Here, GDP/capita co-determines most other indicators (wealthy countries tend
        to have better CA%, controlled inflation, lower unemployment).
        Result: all profiles look like concentric versions of the same polygon —
        not distinct multi-dimensional fingerprints, just a <em>wealth scale</em> displayed on 6 correlated axes.<br>
        <br>
        <strong>Scope of results</strong>: these 6 general macro indicators over 10 years capture
        only a fraction of a country's economic context. With sectoral indicators,
        trade policies, or a longer time window, other
        discriminating dimensions could emerge.
        </div>
        """,
        "l_finding": (
            "<strong>Visible but mono-axial fingerprint</strong>: Global Champions / Strong have "
            "a higher percentile profile, but mainly on GDP/capita and CA%. "
            "Differences on Inflation, Real Rate and Exports/GDP are small — "
            "profiles do not distinguish themselves on these dimensions. "
            "Conclusion valid for this dataset: 6 general macro indicators, 10 years, ~170 countries. "
            "It does not necessarily extend to better-instrumented contexts."
        ),

        "m_label": "M · Can trade performance be predicted from macro fundamentals?",
        "m_question_label": "Question M",
        "m_question_body": (
            "Do the 16 macro indicators <strong>combined</strong> allow prediction of "
            "trade performance (combined score, export share, structural score) "
            "out-of-sample (cross-validation)?"
        ),
        "methodology_label": "Methodology",
        "methodology_html": (
            '<div>① <strong style="color:#EEF4FB;">Features</strong>: 16 macro indicators (median value 2015–2024 per country)</div>'
            '<div>② <strong style="color:#EEF4FB;">Targets</strong>: trade variables (combined score, export share, structural score, CAGR export...)</div>'
            '<div>③ <strong style="color:#EEF4FB;">Model</strong>: Random Forest Regressor (300 trees, max_features=√n)</div>'
            '<div>④ <strong style="color:#EEF4FB;">Evaluation</strong>: R² in-sample vs R² cross-validation (Leave-One-Out / K-Fold)</div>'
            '<div>⑤ <strong style="color:#EEF4FB;">Dataset</strong>: ~170–183 countries, limited n</div>'
        ),
        "key_results_label": "Key Results",
        "res_combined_score": "Combined Score",
        "res_structural_score": "Structural Score",
        "res_export_share": "Export Share",
        "res_cagr_export": "CAGR Export",
        "res_volatility": "Volatility",
        "r2_cv_legend": "R² = in-sample (train) | CV = cross-validation out-of-sample",
        "m_overfit_label": "R² Train vs OLS vs CV — Massive Overfitting",
        "trace_r2_train": "R² Train RF (in-sample)",
        "trace_r2_ols": "R² OLS (linear, reference)",
        "trace_r2_cv": "R² CV RF (out-of-sample)",
        "m_overfit_chart_title": "Overfitting: R² Train RF high, OLS moderate, CV RF near-zero",
        "m_overfit_finding": (
            "<strong>OLS as a linear benchmark</strong>: a simple linear model "
            "(OLS regression on 16 indicators) achieves R² = 0.12–0.28 — well below "
            "RF train (0.45–0.91) but at the same level as RF CV (0.05–0.11). "
            "This means the RF does not learn useful non-linear structure: "
            "it memorizes in-sample noise without capturing more signal than OLS out-of-sample."
        ),
        "m_fi_label": "M · Feature Importance — What the RF Learned",
        "m_fi_intro": (
            "Even while overfitting, the RF reveals "
            "which dimensions it relies on. For level variables (Combined Score, "
            "Structural Score), [T] and [S] indicators dominate — the RF learns a "
            "tautological relationship. For CAGR, no indicator stands out — "
            "the RF is unable to find a signal."
        ),
        "m_fi_selectbox_label": "Target trade variable — see which macro indicators were used",
        "m_fi_chart_title": (
            "Feature Importance RF — {sel_fi}<br>"
            "<span style='font-size:11px;color:#6890B0;'>"
            "[T] red · [S] amber · [E] green</span>"
        ),
        "m_fi_x": "Importance (RF)",
        "m_fi_finding_intro": "<strong>RF learns mainly from [{flag}]</strong>: the most important indicator is <em>{lbl}</em> (importance={val:.3f}). ",
        "m_fi_finding_ts": (
            "[T] or [S] dominant → the RF exploits a tautological or size relationship, "
            "not a causal economic signal. Consistent with near-zero CV R²: "
            "the learned relationship is mechanical, not generalizable."
        ),
        "m_fi_finding_e": (
            "[E] dominant → real economic signal, but insufficient to generalize "
            "(n=~170 too small for 16 non-linear features)."
        ),
        "m_resid_label": "M · Residual Analysis — Export Share vs GDP/capita (OLS)",
        "m_resid_intro": (
            "Bivariate OLS residuals: Export/GDP ~ GDP/capita. Countries above the line "
            "export <em>more</em> than their wealth level predicts — "
            "countries below, <em>less</em>."
        ),
        "m_resid_chart_title": "OLS Residuals: Export/GDP ~ GDP/capita",
        "gdp_pc_usd_axis": "GDP per capita (USD)",
        "residual_axis": "Residual (pp — deviation vs OLS)",
        "reference_label": "Reference",
        "m_resid_pos_finding": "<strong>Positive outliers (over-exporters)</strong>: {items}. These countries export significantly more than their GDP/capita predicts — export-led development model or niche positioning.",
        "m_resid_neg_finding": "<strong>Negative outliers (under-exporters)</strong>: {items}. These countries export less than their wealth level predicts — large domestic economies or import dependency.",
        "interpretation_label": "Interpretation",
        "m_interp_finding1": (
            "<strong>Structural overfitting</strong>: the RF memorises in-sample patterns "
            "but does not generalise. R² train high (0.72–0.91) vs CV near-zero "
            "(0.05–0.11) — the gap is massive. Main cause: n≈170 countries for "
            "16 features — insufficient ratio for a robust non-linear model."
        ),
        "m_interp_finding2": (
            "<strong>For dynamics (CAGR, volatility)</strong>: negative CV R² "
            "— the model performs worse than a constant. Dynamic trade performance "
            "is not predictable from macro fundamentals with this data."
        ),
        "m_interp_finding3": (
            "<strong>What this reveals</strong>: the macro → trade relationship is "
            "<em>descriptive and structural</em>, not predictive. Knowing a country's "
            "16 indicators is not enough to predict its future trade performance "
            "out-of-sample. This is not a failure — it is an important analytical result "
            "that guides the conclusions."
        ),
        "m_interp_finding4": (
            "<strong>What would be needed</strong>: longitudinal panel data (country × time), "
            "sectoral indicators, trade policies, bilateral agreements — "
            "dimensions not captured by the 16 World Bank macro indicators."
        ),

        "n_label": "N · Direction 2 — Trade Structure → Macro Fundamentals",
        "n_question_label": "Question N",
        "n_question_body": (
            "Direction 1 (G→M): <em>macro fundamentals explain trade structure</em>.<br>"
            "Direction 2 (N): <em>can trade structure express a macro indicator?</em><br>"
            "Can a combination of trade variables reconstruct a macro level?"
        ),
        "n_summary_label": "Summary Table — Strongest Signals Direction 2",
        "n_table_html": (
            '<table style="width:100%;border-collapse:collapse;background:#1A2840;'
            'border-radius:8px;overflow:hidden;">'
            '<thead><tr>'
            '<th style="{th}">Macro Indicator</th>'
            '<th style="{th}">Most Correlated Trade Variable</th>'
            '<th style="{th}">ρ</th>'
            '<th style="{th}">Sig.</th>'
            '</tr></thead><tbody>'
            '<tr><td style="{td}color:#A8C0D8;">GDP / capita</td>'
            '<td style="{td}color:#EEF4FB;">Exports / capita</td>'
            '<td style="{td}font-weight:700;color:#10B981;">+0.71</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0.001</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;">Unemployment %</td>'
            '<td style="{td}color:#EEF4FB;">Balance / GDP</td>'
            '<td style="{td}font-weight:700;color:#EF4444;">−0.19</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0.05</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;">Inflation %</td>'
            '<td style="{td}color:#EEF4FB;">Exports / capita</td>'
            '<td style="{td}font-weight:700;color:#EF4444;">−0.18</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0.05</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;border:none;">Current Account % GDP</td>'
            '<td style="{td}color:#EEF4FB;border:none;">Balance / GDP</td>'
            '<td style="{td}font-weight:700;color:#10B981;border:none;">+0.58</td>'
            '<td style="{td}color:#10B981;border:none;">✓ p&lt;0.001</td></tr>'
            '</tbody></table>'
        ),
        "n2_label": "N.2 · Trade → GDP per capita",
        "n2_finding": (
            "<strong>Strongest signal (Direction 2)</strong>: Exports/capita → GDP/capita "
            "(ρ = +0.71, p&lt;0.001, n≈171). A country's trade structure — notably its export "
            "level per capita — is strongly correlated with its standard of living. "
            "But causality is bidirectional: rich countries export more <em>and</em> "
            "countries that export more grow richer."
        ),
        "n2a_chart_title": "Exports/capita → GDP/capita",
        "export_pc_usd": "Exports per capita (USD)",
        "gdp_pc_usd": "GDP per capita (USD)",
        "other_countries_label": "Other countries",
        "n2b_chart_title": "Trade Intensity → GDP/capita",
        "trade_intensity_xm_gdp": "Trade Intensity (exp+imp)/GDP",
        "n2_china_finding": (
            "<strong>China case</strong>: outlier on the 'volume' dimension — massive total "
            "exports but moderate GDP/capita (~12k USD). Trade intensity ≈ 0.37, intermediate "
            "position. Illustrates that absolute export volume does not imply an equivalent "
            "standard of living: the distribution of value created remains unequal."
        ),

        "n3_label": "N.3 · Trade → Unemployment",
        "n3_chart_title": "Trade Balance / GDP → Unemployment",
        "balance_gdp_axis": "Balance / GDP  (exp−imp)/GDP",
        "unemployment_axis": "Unemployment (% of labour force)",
        "n3_finding1": (
            "<strong>Weak but consistent signal</strong>: ρ ≈ −0.19 (p&lt;0.05). "
            "A trade surplus is slightly associated with lower unemployment — "
            "a logical relationship (net exports = net domestic jobs)."
        ),
        "n3_finding2": (
            "<strong>Germany / Japan cluster</strong>: high trade surplus "
            "+ low unemployment (~3–5%). Most theory-consistent configuration."
        ),
        "n3_finding3": (
            "<strong>Brazil anomaly</strong>: positive trade surplus but "
            "high unemployment (~11%). Illustrates that sectoral structure (commodities) "
            "can decouple trade balance from domestic employment."
        ),
        "n3_finding4": (
            "<strong>Asymmetric signal vs Direction 1</strong>: unemployment was "
            "not a significant discriminant of trade profiles (Section L). "
            "Direction 2 reveals a complementary logic: the net balance partially "
            "reflects labour absorption capacity."
        ),

        "n4_label": "N.4 · Trade → Inflation",
        "n4_chart_title": "Exports/capita → Inflation",
        "inflation_annual_axis": "Inflation (% annual)",
        "n4_finding1": (
            "<strong>Weak negative signal</strong>: countries with high per-capita "
            "exports tend towards lower inflation — consistent with the supply of "
            "goods available on the domestic market and the competitiveness "
            "of exporting countries' currencies."
        ),
        "n4_finding2": (
            "<strong>Direction 1 vs 2 asymmetry</strong>: in Direction 1, "
            "inflation was not a significant discriminant of trade profiles. "
            "Direction 2 reveals a weak but real relationship: major "
            "exporters better maintain price stability."
        ),
        "n4_finding3": (
            "<strong>High variance</strong>: countries with low exports show "
            "inflation from 1% to 30%+. Trade structure explains a minor fraction "
            "of inflation variance — other factors (monetary policy, "
            "energy prices, cycles) dominate."
        ),

        "n5_label": "N.5 · Inverse-L Result — A Single Discriminating Indicator",
        "n5_box": (
            '<strong style="color:#A78BFA;">Section L Reminder</strong>: in Direction 1, '
            'only GDP/capita significantly discriminates trade profiles. '
            'Unemployment, inflation, interest rates: no robust cross-profile signal.'
        ),
        "n5_finding1": (
            "<strong>The same result in reverse</strong>: Direction 2 confirms that "
            "GDP/capita is the only macro indicator strongly correlated with trade variables "
            "(ρ=+0.71 for exports/capita). Reversing the direction — 'trade → macro' — "
            "yields exactly the same hierarchy: GDP/capita dominates, other "
            "indicators remain marginal (|ρ| &lt; 0.20)."
        ),
        "n5_finding2": (
            "<strong>Interpretation</strong>: it is not that trade 'causes' wealth, "
            "nor that wealth 'causes' trade. GDP/capita and trade structure are two "
            "manifestations of the same phenomenon: the level of economic development. "
            "Both variables are co-determined — the chosen analytical direction "
            "does not alter the statistical signal."
        ),

        "n6_label": "N.6 · Inverse-M Result — Prediction Fails in Both Directions",
        "n6_finding_intro": (
            "<strong>Section M</strong> showed that macro indicators cannot "
            "predict trade performance out-of-sample (CV R² near-zero or negative). "
            "<strong>Section N.6</strong> reverses the question: can a macro outcome "
            "be predicted from trade variables?"
        ),
        "n6_trace_train": "R² Train (in-sample)",
        "n6_trace_cv": "R² CV (out-of-sample)",
        "n6_gdp_cap": "GDP/capita",
        "n6_chart_title": "Direction 2 — R² Train vs CV (Trade → Macro)",
        "n6_finding1": (
            "<strong>Same pattern as Direction 1</strong>: high R² train (0.38–0.82), "
            "near-zero or negative CV. The RF memorises the data but does not generalise "
            "in either direction."
        ),
        "n6_finding2": (
            "<strong>Direction 2 even more fragile</strong>: for Inflation, CV = −0.49. "
            "Using trade variables to predict an unknown country's inflation "
            "would give a worse result than a constant prediction at the mean."
        ),

        "n_conclusion_label": "Direction 2 Conclusion",
        "n_conclusion1": (
            "<strong>Symmetric correlations, asymmetric interpretation</strong>: "
            "ρ values are identical in both directions by construction. What changes is "
            "the question: 'trade → macro' asks whether a country's structure can reconstruct "
            "its macro level. The answer (CV R² ≈ 0) is no out-of-sample."
        ),
        "n_conclusion2": (
            "<strong>Additional signals from N</strong>: balance/GDP → unemployment and "
            "export/capita → inflation are two relationships that the Direction 1 analysis "
            "did not highlight. They are weak (|ρ| &lt; 0.20) but economically "
            "consistent."
        ),
        "n_conclusion3": (
            "<strong>Directional conclusion</strong>: the data better supports "
            "the macro → trade direction than trade → macro. Economic fundamentals "
            "structure trade position; trade structure does not predict "
            "macro outcomes. The relationship is real but descriptive, not predictive."
        ),

        "p_label": "P · Target Country — Trade × Macro Diagnostic",
        "country_label": "Country",
        "scorecard_label": "Scorecard — {country}",
        "chip_combined_score": "Combined Score",
        "chip_structural_score": "Structural Score",
        "chip_f_norm_rank": "F Norm Rank",
        "chip_g_norm_rank": "G Norm Rank",
        "chip_combined_profile": "Combined Profile",
        "chip_meta_profile": "Meta Profile",
        "bm_export_share": "Export Share",
        "bm_import_share": "Import Share",
        "bm_coverage_ratio": "Coverage Ratio",
        "bm_cagr_export": "CAGR Export",
        "bm_export_volatility": "Export Volatility",
        "bm_balance_gdp": "Balance/GDP",
        "median_label": "Median",
        "global_fallback": "global",
        "trade_metrics_vs_median_title": "Trade metrics vs group median ({profile})",
        "mac_gdp_cap": "GDP/capita",
        "mac_current_account": "Current Account %",
        "mac_inflation": "Inflation %",
        "mac_unemployment": "Unemployment %",
        "mac_real_rate": "Real Rate %",
        "mac_expense_gdp": "Expense % GDP",
        "group_fallback": "group",
        "vs_group_median": "▲ / ▼ vs group median <strong style=\"color:#6890B0;\">{group}</strong>",
        "na_suffix": "n/a",
        "radar_vs_peers_label": "Radar — {country} vs group « {profile} »",
        "percentile_hover2": "<b>%{theta}</b><br>Percentile=%{r:.2f}<extra></extra>",
        "position_trend_label": "Position relative to the structural GDP/capita trend",
        "descriptive_reading_box": (
            "Descriptive reading — not a prediction. The OLS line (GDP/capita → combined_score) "
            "shows where countries with this wealth level generally sit. "
            "The residual says whether this country is above or below that structural trend, "
            "not what it should be."
        ),
        "structural_residual_label": "Structural residual",
        "structural_residual_body": (
            'Observed score: <strong style="color:{rc};">{ccs:.2f}</strong> · '
            'Trend for this GDP/capita: <strong>{trend:.2f}</strong><br>'
            '<strong style="color:{rc};">{country} is {dir} '
            'the structural trend</strong> '
            '(residual = {resid:+.2f}, i.e. {rpct:.0f}%)<br>'
            '<span style="font-size:12px;color:#7090B0;">'
            'GDP/capita: ${cg:,.0f} — '
            'ρ(GDP/cap → combined_score) = +0.33 [E]</span>'
        ),
        "above_trend": "above",
        "below_trend": "below",
        "surplus_word": "in surplus",
        "deficit_word": "in deficit",
        "ca_pct_finding": (
            "<strong>CA% = {ca:+.1f}%</strong> ({cad} vs median {cam:+.1f}%) "
            "— second dominant [E] signal (η²=0.241–0.250 sections I/J). {coherence}"
        ),
        "ca_coherent": "Coherent signal: current surplus and trade overperformance.",
        "ca_mixed": (
            "Mixed signal: the current balance contradicts the position relative to "
            "GDP/capita — idiosyncratic factors play a role."
        ),
        "scatter_explorer_label": "Scatter Explorer — Explore any pair",
        "tc_combined_score": "Combined Score",
        "tc_structural_score": "Structural Score",
        "tc_export_share": "Export Share",
        "tc_coverage_ratio": "Coverage Ratio",
        "tc_cagr_export": "CAGR Export",
        "tc_volatility_export": "Export Volatility",
        "tc_balance_gdp": "Balance/GDP",
        "tc_export_pc": "Export/Capita",
        "mc_gdp_capita": "GDP/Capita",
        "mc_ca_gdp": "CA % GDP",
        "mc_inflation": "Inflation %",
        "mc_unemployment": "Unemployment %",
        "mc_real_rate": "Real Rate %",
        "mc_expense_gdp": "Expense % GDP",
        "trade_var_y_label": "Trade variable (Y)",
        "macro_ind_x_label": "Macro indicator (X)",
        "how_to_read_expander": "How to read this chart",
        "how_to_read_scatter_html": """
                <div style='font-size:13px;color:#A8C0D8;line-height:1.9;'>
                <strong style='color:#EEF4FB;'>ρ (Spearman) and the ✓ / ✗</strong><br>
                ρ measures the strength of the monotonic relationship across all ~170 countries.
                ✓ (p&lt;0.05) = the overall trend is statistically real.
                ✗ = even the overall trend is noise — there is no signal between these two variables.<br>
                <em>Caution: ✓ does not mean an individual country's position is predictable —
                the CV R² remains near-zero (Section M).</em><br><br>
                <strong style='color:#EEF4FB;'>OLS line (dashed blue)</strong><br>
                Linear approximation of the central trend. It is not a prediction model —
                it is a visual reference to see a country's deviation.<br><br>
                <strong style='color:#EEF4FB;'>Country position (gold star) relative to the line</strong><br>
                <span style='color:#10B981;'>Above</span> = this country performs better on Y than its X value would suggest
                (positive residual — idiosyncratic advantage).<br>
                <span style='color:#EF4444;'>Below</span> = this country performs worse than the trend
                (negative residual — limiting factors not captured by X).<br><br>
                <strong style='color:#EEF4FB;'>Point color</strong><br>
                Combined trade profile — lets you see whether profile clusters
                group together or scatter on the chosen pair.
                </div>
                """,

        "syn_label": "Summary — What the analysis reveals",
        "syn_scope_html": (
            '<span style="font-size:11px;font-weight:700;color:#4A6890;text-transform:uppercase;'
            'letter-spacing:.07em;">Scope</span>'
            '<span style="font-size:13px;color:#A8C0D8;">~<strong style="color:#EEF4FB;">170</strong> countries</span>'
            '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">16</strong> World Bank macro indicators</span>'
            '<span style="font-size:13px;color:#A8C0D8;">Median <strong style="color:#EEF4FB;">2015–2024</strong></span>'
            '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">2</strong> analysis directions</span>'
            '<span style="font-size:12px;color:#F59E0B;font-style:italic;font-weight:600;">'
            '⚠ conclusions valid for this scope only</span>'
        ),
        "syn_dir1_title": "Direction 1 · G → M",
        "syn_dir1_sub": "Macro fundamentals → Trade structure",
        "syn_dir1_kpi1": "GDP → Export Share",
        "syn_dir1_kpi2": "GDP/cap → Export/cap",
        "syn_dir1_kpi3": "RF out-of-sample",
        "syn_dir1_footer": "Real [E] signal · out-of-sample prediction fails",
        "syn_dir2_title": "Direction 2 · N",
        "syn_dir2_sub": "Trade structure → Macro fundamentals",
        "syn_dir2_kpi1": "Export/cap → GDP/cap",
        "syn_dir2_kpi2": "RF Inflation (out-of-sample)",
        "syn_dir2_kpi3": "RF GDP/cap (out-of-sample)",
        "syn_dir2_footer": "Symmetric correlations · prediction even more fragile",
        "syn_conclusion_label": "Main conclusion",
        "syn_conclusion_body": (
            "Within this analysis scope, the relationship between macro and trade "
            "<strong>proves to be structural and descriptive</strong> — not predictive. "
            "Macro fundamentals explain <em>where</em> a country positions itself in trade "
            "(level, profile), but not <em>how</em> it will perform out-of-sample."
        ),
        "syn_conclusion_footer": (
            "Conditional result: 16 general World Bank macro indicators · ~170 countries · "
            "2015–2024 median · no sectoral or institutional indicators. "
            "Richer data could reveal a predictive dimension."
        ),
        "syn_found_label": "What was found",
        "syn_found1": (
            "<strong>Size trumps wealth</strong> (Section G–H): absolute GDP "
            "explains export share (ρ=0.94) better than GDP_pc explains export/capita "
            "(ρ=0.82). Being large determines volume; being wealthy determines efficiency."
        ),
        "syn_found2": (
            "<strong>Observed [E] hierarchy</strong> (Section K): on this data, "
            "GDP/capita > CA% > Real Rate > Inflation > Unemployment. "
            "These indicators show a weak-to-moderate signal on trade structure — "
            "a hierarchy to confirm with other sources or scopes."
        ),
        "syn_found3": (
            "<strong>Macro fingerprint by profile</strong> (Section L): across the 6 indicators "
            "analyzed, GDP/capita and CA% stand out as the most "
            "discriminating dimensions between trade profiles. The 4 other indicators (inflation, "
            "unemployment, real rate, exports/GDP) show weak differences within this "
            "window and data scope."
        ),
        "syn_found4": (
            "<strong>Trade Intensity reveals openness</strong> (Section D): wealth "
            "tends toward openness but with high variance. Small wealthy countries "
            "(Singapore, Luxembourg) are the most open in relative terms."
        ),
        "syn_notfound_label": "What was not found",
        "syn_notfound1": (
            "<strong>Out-of-sample prediction impossible</strong> (Section M): high RF R² "
            "(0.72–0.91) but near-zero CV R² (0.05–0.11). The model memorises, "
            "it does not generalise. Dynamics (CAGR, volatility) are even less "
            "predictable (CV R² < 0)."
        ),
        "syn_notfound2": (
            "<strong>Direction 2 not robust</strong> (Section N): trade → macro "
            "does not predict macro outcomes out-of-sample (CV R² ≈ 0 or negative). "
            "On this data, the macro → trade direction seems more informative, "
            "but since Spearman correlations are symmetric, causality "
            "cannot be established from this analysis alone."
        ),
        "syn_notfound3": (
            "<strong>Dynamics not captured</strong> (Section I–J): on this dataset, "
            "trade growth and volatility are not explained by the 16 "
            "available macro indicators (CV R² &lt; 0). These dynamics likely involve "
            "factors not measured here: sectoral policy, "
            "geography, exogenous shocks, bilateral agreements."
        ),
        "syn_orientations_label": "Directions for further work",
        "syn_orientations_html": (
            '<div>→ <strong style="color:#EEF4FB;">Panel data</strong>: exploit the time '
            'dimension (country × year) to capture trade-policy effects</div>'
            '<div>→ <strong style="color:#EEF4FB;">Sectoral indicators</strong>: product '
            'diversification, economic complexity (ECI), geographic dependencies</div>'
            '<div>→ <strong style="color:#EEF4FB;">Institutional variables</strong>: institutional '
            'quality, trade agreements, tariff barriers</div>'
            '<div>→ <strong style="color:#EEF4FB;">Validated non-linear models</strong>: with more '
            'countries or synthetic data to offset n=~170</div>'
            '<div>→ <strong style="color:#EEF4FB;">Exogenous shocks</strong>: COVID, commodities — '
            'identify countries that resist vs amplify macro shocks</div>'
        ),
    },
    "fr": {
        "eyebrow": "Lien · Commerce × Indicateurs · Page 2 sur 2",
        "title": "Stratégie & conclusions",
        "caption": (
            "Sections L · M · N — Empreinte macro par profil, tentative de prédiction, "
            "et inversion du sens de l'analyse"
        ),
        "tab_l": "L · Empreinte macro",
        "tab_m": "M · Tentative de prédiction",
        "tab_n": "N · Direction 2",
        "tab_p": "P · Pays cible",
        "tab_syn": "Synthèse",

        "l_label": "L · Empreinte macro par profil commercial",
        "l_intro": (
            "Idée : chaque profil commercial a-t-il une <em>empreinte macro caractéristique</em> ? "
            "Indicateurs normalisés en percentile (0=pire, 1=meilleur du dataset). "
            "Pour l'inflation et le chômage : percentile inversé (plus bas = meilleur)."
        ),
        "l_rationale": (
            '<strong style="color:#34D399;">Justification des 6 indicateurs</strong> : '
            'le PIB/habitant et le CA% sont les deux signaux [E] dominants (sections G–K). '
            'L\'inflation, le chômage et le taux réel complètent le tableau macro cyclique. '
            'Exports/PIB (indicateur [T]) est inclus uniquement pour visualiser '
            'l\'ouverture commerciale — pas pour établir une causalité.'
        ),
        "radar_gdp_cap": "PIB/hab",
        "radar_cur_acct": "Cpte courant",
        "radar_inflation": "Inflation",
        "radar_unemployment": "Chômage",
        "radar_real_rate": "Taux réel",
        "radar_exports_gdp": "Exports/PIB",
        "profile_dim_label": "Dimension de profil",
        "profile_combined": "Profil combiné",
        "profile_meta": "Méta-profil",
        "profile_structural": "Profil structurel",
        "percentile_hover": "<b>%{theta}</b><br>Percentile = %{r:.2f}<extra></extra>",
        "l_headline_out_of": "sur 5",
        "l_headline_title": "indicateur macro discrimine les profils commerciaux",
        "l_headline_body": (
            '<strong style="color:#34D399;">PIB / habitant</strong> — différences claires et '
            'significatives entre profils (KW p&nbsp;&lt;&nbsp;0,001).<br>'
            '<span style="color:#6890B0;">Inflation · Chômage · Taux réel · CA%</span> — '
            'différences faibles ou non significatives entre profils.<br>'
            '<em style="color:#7090B0;">Sur ces 6 indicateurs macro et 10 ans de données : '
            'le signal dominant est la richesse. Avec plus d\'indicateurs sectoriels '
            'ou des données plus longues, d\'autres dimensions pourraient émerger.</em>'
        ),
        "l_how_to_read_expander": "Comment lire le graphique — limites de la visualisation",
        "l_how_to_read_html": """
        <div style='font-size:13px;color:#A8C0D8;line-height:1.8;'>
        <strong>Axe</strong> = percentile normalisé de l'indicateur (0 = pire pays du dataset, 1 = meilleur).<br>
        <strong>Zone plus grande</strong> = profil macro globalement meilleur sur ces 6 dimensions.<br>
        <strong>Inflation & Chômage</strong> : percentile inversé — un pays à faible inflation/chômage apparaît en position haute.<br>
        <br>
        <strong style='color:#F59E0B;'>Ce n'est pas vraiment un radar au sens classique.</strong><br>
        Un vrai radar suppose des axes <em>indépendants</em> : chaque dimension apporte une information distincte
        et les profils forment des polygones de formes vraiment différentes.
        Ici, le PIB/habitant co-détermine la plupart des autres indicateurs (les pays riches tendent
        à avoir un meilleur CA%, une inflation maîtrisée, un chômage plus bas).
        Résultat : tous les profils ressemblent à des versions concentriques du même polygone —
        pas des empreintes multidimensionnelles distinctes, juste une <em>échelle de richesse</em> affichée sur 6 axes corrélés.<br>
        <br>
        <strong>Portée des résultats</strong> : ces 6 indicateurs macro généraux sur 10 ans capturent
        seulement une fraction du contexte économique d'un pays. Avec des indicateurs sectoriels,
        des politiques commerciales, ou une fenêtre temporelle plus longue, d'autres
        dimensions discriminantes pourraient émerger.
        </div>
        """,
        "l_finding": (
            "<strong>Empreinte visible mais mono-axiale</strong> : les Global Champions / Strong ont "
            "un profil de percentile plus élevé, mais surtout sur le PIB/habitant et le CA%. "
            "Les différences sur l'inflation, le taux réel et exports/PIB sont faibles — "
            "les profils ne se distinguent pas sur ces dimensions. "
            "Conclusion valable pour ce dataset : 6 indicateurs macro généraux, 10 ans, ~170 pays. "
            "Elle ne s'étend pas nécessairement à des contextes mieux instrumentés."
        ),

        "m_label": "M · La performance commerciale est-elle prédictible à partir des fondamentaux macro ?",
        "m_question_label": "Question M",
        "m_question_body": (
            "Les 16 indicateurs macro <strong>combinés</strong> permettent-ils de prédire "
            "la performance commerciale (score combiné, part des exports, score structurel) "
            "hors-échantillon (validation croisée) ?"
        ),
        "methodology_label": "Méthodologie",
        "methodology_html": (
            '<div>① <strong style="color:#EEF4FB;">Variables explicatives</strong> : 16 indicateurs macro (valeur médiane 2015–2024 par pays)</div>'
            '<div>② <strong style="color:#EEF4FB;">Cibles</strong> : variables commerciales (score combiné, part des exports, score structurel, CAGR export...)</div>'
            '<div>③ <strong style="color:#EEF4FB;">Modèle</strong> : Random Forest Regressor (300 arbres, max_features=√n)</div>'
            '<div>④ <strong style="color:#EEF4FB;">Évaluation</strong> : R² in-sample vs R² validation croisée (Leave-One-Out / K-Fold)</div>'
            '<div>⑤ <strong style="color:#EEF4FB;">Dataset</strong> : ~170–183 pays, n limité</div>'
        ),
        "key_results_label": "Résultats clés",
        "res_combined_score": "Combined Score",
        "res_structural_score": "Structural Score",
        "res_export_share": "Export Share",
        "res_cagr_export": "CAGR Export",
        "res_volatility": "Volatilité",
        "r2_cv_legend": "R² = in-sample (entraînement) | CV = validation croisée hors-échantillon",
        "m_overfit_label": "R² Train vs OLS vs CV — Surapprentissage massif",
        "trace_r2_train": "R² Train RF (in-sample)",
        "trace_r2_ols": "R² OLS (linéaire, référence)",
        "trace_r2_cv": "R² CV RF (hors-échantillon)",
        "m_overfit_chart_title": "Surapprentissage : R² Train RF élevé, OLS modéré, CV RF quasi nul",
        "m_overfit_finding": (
            "<strong>L'OLS comme référence linéaire</strong> : un modèle linéaire simple "
            "(régression OLS sur 16 indicateurs) atteint R² = 0,12–0,28 — bien en dessous "
            "du RF train (0,45–0,91) mais au même niveau que le RF CV (0,05–0,11). "
            "Cela signifie que le RF n'apprend pas de structure non linéaire utile : "
            "il mémorise le bruit in-sample sans capturer plus de signal que l'OLS hors-échantillon."
        ),
        "m_fi_label": "M · Importance des variables — Ce que le RF a appris",
        "m_fi_intro": (
            "Même en surapprenant, le RF révèle "
            "sur quelles dimensions il s'appuie. Pour les variables de niveau (Combined Score, "
            "Structural Score), les indicateurs [T] et [S] dominent — le RF apprend une "
            "relation tautologique. Pour le CAGR, aucun indicateur ne se distingue — "
            "le RF est incapable de trouver un signal."
        ),
        "m_fi_selectbox_label": "Variable commerciale cible — voir quels indicateurs macro ont été utilisés",
        "m_fi_chart_title": (
            "Importance des variables RF — {sel_fi}<br>"
            "<span style='font-size:11px;color:#6890B0;'>"
            "[T] rouge · [S] ambre · [E] vert</span>"
        ),
        "m_fi_x": "Importance (RF)",
        "m_fi_finding_intro": "<strong>Le RF apprend surtout à partir de [{flag}]</strong> : l'indicateur le plus important est <em>{lbl}</em> (importance={val:.3f}). ",
        "m_fi_finding_ts": (
            "[T] ou [S] dominant → le RF exploite une relation tautologique ou de taille, "
            "pas un signal économique causal. Cohérent avec le R² CV quasi nul : "
            "la relation apprise est mécanique, pas généralisable."
        ),
        "m_fi_finding_e": (
            "[E] dominant → signal économique réel, mais insuffisant pour généraliser "
            "(n=~170 trop petit pour 16 variables non linéaires)."
        ),
        "m_resid_label": "M · Analyse des résidus — Part des exports vs PIB/habitant (OLS)",
        "m_resid_intro": (
            "Résidus OLS bivariés : Export/PIB ~ PIB/habitant. Les pays au-dessus de la droite "
            "exportent <em>plus</em> que ce que leur niveau de richesse prédit — "
            "les pays en-dessous, <em>moins</em>."
        ),
        "m_resid_chart_title": "Résidus OLS : Export/PIB ~ PIB/habitant",
        "gdp_pc_usd_axis": "PIB par habitant (USD)",
        "residual_axis": "Résidu (pp — écart vs OLS)",
        "reference_label": "Référence",
        "m_resid_pos_finding": "<strong>Valeurs extrêmes positives (sur-exportateurs)</strong> : {items}. Ces pays exportent significativement plus que ce que prédit leur PIB/habitant — modèle de développement tiré par l'export ou positionnement de niche.",
        "m_resid_neg_finding": "<strong>Valeurs extrêmes négatives (sous-exportateurs)</strong> : {items}. Ces pays exportent moins que ce que prédit leur niveau de richesse — grandes économies domestiques ou dépendance aux importations.",
        "interpretation_label": "Interprétation",
        "m_interp_finding1": (
            "<strong>Surapprentissage structurel</strong> : le RF mémorise des patterns in-sample "
            "mais ne généralise pas. R² train élevé (0,72–0,91) vs CV quasi nul "
            "(0,05–0,11) — l'écart est massif. Cause principale : n≈170 pays pour "
            "16 variables — ratio insuffisant pour un modèle non linéaire robuste."
        ),
        "m_interp_finding2": (
            "<strong>Pour les dynamiques (CAGR, volatilité)</strong> : R² CV négatif "
            "— le modèle performe moins bien qu'une constante. La performance commerciale dynamique "
            "n'est pas prévisible à partir des fondamentaux macro avec ces données."
        ),
        "m_interp_finding3": (
            "<strong>Ce que cela révèle</strong> : la relation macro → commerce est "
            "<em>descriptive et structurelle</em>, pas prédictive. Connaître les "
            "16 indicateurs d'un pays ne suffit pas à prédire sa future performance commerciale "
            "hors-échantillon. Ce n'est pas un échec — c'est un résultat analytique important "
            "qui oriente les conclusions."
        ),
        "m_interp_finding4": (
            "<strong>Ce qu'il faudrait</strong> : des données panel longitudinales (pays × temps), "
            "des indicateurs sectoriels, des politiques commerciales, des accords bilatéraux — "
            "des dimensions non capturées par les 16 indicateurs macro de la Banque mondiale."
        ),

        "n_label": "N · Direction 2 — Structure commerciale → Fondamentaux macro",
        "n_question_label": "Question N",
        "n_question_body": (
            "Direction 1 (G→M) : <em>les fondamentaux macro expliquent la structure commerciale</em>.<br>"
            "Direction 2 (N) : <em>la structure commerciale peut-elle exprimer un indicateur macro ?</em><br>"
            "Une combinaison de variables commerciales peut-elle reconstruire un niveau macro ?"
        ),
        "n_summary_label": "Tableau récapitulatif — Signaux les plus forts, Direction 2",
        "n_table_html": (
            '<table style="width:100%;border-collapse:collapse;background:#1A2840;'
            'border-radius:8px;overflow:hidden;">'
            '<thead><tr>'
            '<th style="{th}">Indicateur macro</th>'
            '<th style="{th}">Variable commerciale la plus corrélée</th>'
            '<th style="{th}">ρ</th>'
            '<th style="{th}">Sig.</th>'
            '</tr></thead><tbody>'
            '<tr><td style="{td}color:#A8C0D8;">PIB / habitant</td>'
            '<td style="{td}color:#EEF4FB;">Exports / habitant</td>'
            '<td style="{td}font-weight:700;color:#10B981;">+0,71</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0,001</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;">Chômage %</td>'
            '<td style="{td}color:#EEF4FB;">Balance / PIB</td>'
            '<td style="{td}font-weight:700;color:#EF4444;">−0,19</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0,05</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;">Inflation %</td>'
            '<td style="{td}color:#EEF4FB;">Exports / habitant</td>'
            '<td style="{td}font-weight:700;color:#EF4444;">−0,18</td>'
            '<td style="{td}color:#10B981;">✓ p&lt;0,05</td></tr>'
            '<tr><td style="{td}color:#A8C0D8;border:none;">Compte courant % PIB</td>'
            '<td style="{td}color:#EEF4FB;border:none;">Balance / PIB</td>'
            '<td style="{td}font-weight:700;color:#10B981;border:none;">+0,58</td>'
            '<td style="{td}color:#10B981;border:none;">✓ p&lt;0,001</td></tr>'
            '</tbody></table>'
        ),
        "n2_label": "N.2 · Commerce → PIB par habitant",
        "n2_finding": (
            "<strong>Signal le plus fort (Direction 2)</strong> : Exports/habitant → PIB/habitant "
            "(ρ = +0,71, p&lt;0,001, n≈171). La structure commerciale d'un pays — notamment son niveau "
            "d'export par habitant — est fortement corrélée à son niveau de vie. "
            "Mais la causalité est bidirectionnelle : les pays riches exportent plus <em>et</em> "
            "les pays qui exportent plus s'enrichissent."
        ),
        "n2a_chart_title": "Exports/habitant → PIB/habitant",
        "export_pc_usd": "Exports par habitant (USD)",
        "gdp_pc_usd": "PIB par habitant (USD)",
        "other_countries_label": "Autres pays",
        "n2b_chart_title": "Intensité commerciale → PIB/habitant",
        "trade_intensity_xm_gdp": "Intensité commerciale (exp+imp)/PIB",
        "n2_china_finding": (
            "<strong>Cas de la Chine</strong> : valeur extrême sur la dimension « volume » — exports "
            "totaux massifs mais PIB/habitant modéré (~12k USD). Intensité commerciale ≈ 0,37, position "
            "intermédiaire. Illustre que le volume d'export absolu n'implique pas un niveau de vie "
            "équivalent : la répartition de la valeur créée reste inégale."
        ),

        "n3_label": "N.3 · Commerce → Chômage",
        "n3_chart_title": "Balance commerciale / PIB → Chômage",
        "balance_gdp_axis": "Balance / PIB  (exp−imp)/PIB",
        "unemployment_axis": "Chômage (% de la population active)",
        "n3_finding1": (
            "<strong>Signal faible mais consistant</strong> : ρ ≈ −0,19 (p&lt;0,05). "
            "Un excédent commercial est légèrement associé à un chômage plus faible — "
            "une relation logique (exports nets = emplois domestiques nets)."
        ),
        "n3_finding2": (
            "<strong>Groupe Allemagne / Japon</strong> : fort excédent commercial "
            "+ faible chômage (~3–5%). Configuration la plus cohérente avec la théorie."
        ),
        "n3_finding3": (
            "<strong>Anomalie du Brésil</strong> : excédent commercial positif mais "
            "chômage élevé (~11%). Illustre que la structure sectorielle (matières premières) "
            "peut découpler la balance commerciale de l'emploi domestique."
        ),
        "n3_finding4": (
            "<strong>Signal asymétrique vs Direction 1</strong> : le chômage n'était "
            "pas un discriminant significatif des profils commerciaux (Section L). "
            "La Direction 2 révèle une logique complémentaire : la balance nette reflète "
            "partiellement la capacité d'absorption de main-d'œuvre."
        ),

        "n4_label": "N.4 · Commerce → Inflation",
        "n4_chart_title": "Exports/habitant → Inflation",
        "inflation_annual_axis": "Inflation (% annuel)",
        "n4_finding1": (
            "<strong>Signal négatif faible</strong> : les pays à forts exports par habitant "
            "tendent vers une inflation plus basse — cohérent avec l'offre de "
            "biens disponibles sur le marché domestique et la compétitivité "
            "des devises des pays exportateurs."
        ),
        "n4_finding2": (
            "<strong>Asymétrie Direction 1 vs 2</strong> : en Direction 1, "
            "l'inflation n'était pas un discriminant significatif des profils commerciaux. "
            "La Direction 2 révèle une relation faible mais réelle : les grands "
            "exportateurs maintiennent mieux la stabilité des prix."
        ),
        "n4_finding3": (
            "<strong>Forte variance</strong> : les pays à faibles exports montrent "
            "une inflation de 1% à 30%+. La structure commerciale explique une fraction "
            "minime de la variance de l'inflation — d'autres facteurs (politique monétaire, "
            "prix de l'énergie, cycles) dominent."
        ),

        "n5_label": "N.5 · Résultat Inverse-L — Un seul indicateur discriminant",
        "n5_box": (
            '<strong style="color:#A78BFA;">Rappel de la section L</strong> : en Direction 1, '
            'seul le PIB/habitant discrimine significativement les profils commerciaux. '
            'Chômage, inflation, taux d\'intérêt : aucun signal robuste inter-profils.'
        ),
        "n5_finding1": (
            "<strong>Le même résultat en sens inverse</strong> : la Direction 2 confirme que "
            "le PIB/habitant est le seul indicateur macro fortement corrélé aux variables commerciales "
            "(ρ=+0,71 pour exports/habitant). Inverser le sens — « commerce → macro » — "
            "donne exactement la même hiérarchie : le PIB/habitant domine, les autres "
            "indicateurs restent marginaux (|ρ| &lt; 0,20)."
        ),
        "n5_finding2": (
            "<strong>Interprétation</strong> : ce n'est pas que le commerce « cause » la richesse, "
            "ni que la richesse « cause » le commerce. Le PIB/habitant et la structure commerciale sont "
            "deux manifestations du même phénomène : le niveau de développement économique. "
            "Les deux variables sont codéterminées — le sens analytique choisi "
            "ne modifie pas le signal statistique."
        ),

        "n6_label": "N.6 · Résultat Inverse-M — La prédiction échoue dans les deux sens",
        "n6_finding_intro": (
            "<strong>La section M</strong> a montré que les indicateurs macro ne peuvent "
            "pas prédire la performance commerciale hors-échantillon (R² CV quasi nul ou négatif). "
            "<strong>La section N.6</strong> inverse la question : un résultat macro peut-il "
            "être prédit à partir des variables commerciales ?"
        ),
        "n6_trace_train": "R² Train (in-sample)",
        "n6_trace_cv": "R² CV (hors-échantillon)",
        "n6_gdp_cap": "PIB/habitant",
        "n6_chart_title": "Direction 2 — R² Train vs CV (Commerce → Macro)",
        "n6_finding1": (
            "<strong>Même pattern que la Direction 1</strong> : R² train élevé (0,38–0,82), "
            "CV quasi nul ou négatif. Le RF mémorise les données mais ne généralise "
            "dans aucun des deux sens."
        ),
        "n6_finding2": (
            "<strong>La Direction 2 est encore plus fragile</strong> : pour l'inflation, CV = −0,49. "
            "Utiliser les variables commerciales pour prédire l'inflation d'un pays inconnu "
            "donnerait un résultat pire qu'une prédiction constante à la moyenne."
        ),

        "n_conclusion_label": "Conclusion de la Direction 2",
        "n_conclusion1": (
            "<strong>Corrélations symétriques, interprétation asymétrique</strong> : "
            "les valeurs de ρ sont identiques dans les deux sens par construction. Ce qui change, "
            "c'est la question : « commerce → macro » demande si la structure d'un pays peut reconstruire "
            "son niveau macro. La réponse (R² CV ≈ 0) est non hors-échantillon."
        ),
        "n_conclusion2": (
            "<strong>Signaux additionnels issus de N</strong> : balance/PIB → chômage et "
            "exports/habitant → inflation sont deux relations que l'analyse de la Direction 1 "
            "n'avait pas mises en évidence. Elles sont faibles (|ρ| &lt; 0,20) mais économiquement "
            "cohérentes."
        ),
        "n_conclusion3": (
            "<strong>Conclusion directionnelle</strong> : les données soutiennent mieux "
            "le sens macro → commerce que commerce → macro. Les fondamentaux économiques "
            "structurent la position commerciale ; la structure commerciale ne prédit pas "
            "les résultats macro. La relation est réelle mais descriptive, pas prédictive."
        ),

        "p_label": "P · Pays cible — Diagnostic Commerce × Macro",
        "country_label": "Pays",
        "scorecard_label": "Fiche pays — {country}",
        "chip_combined_score": "Combined Score",
        "chip_structural_score": "Structural Score",
        "chip_f_norm_rank": "F Norm Rank",
        "chip_g_norm_rank": "G Norm Rank",
        "chip_combined_profile": "Profil combiné",
        "chip_meta_profile": "Méta-profil",
        "bm_export_share": "Part des exports",
        "bm_import_share": "Part des imports",
        "bm_coverage_ratio": "Ratio de couverture",
        "bm_cagr_export": "CAGR Export",
        "bm_export_volatility": "Volatilité des exports",
        "bm_balance_gdp": "Balance/PIB",
        "median_label": "Médiane",
        "global_fallback": "global",
        "trade_metrics_vs_median_title": "Métriques commerciales vs médiane du groupe ({profile})",
        "mac_gdp_cap": "PIB/habitant",
        "mac_current_account": "Compte courant %",
        "mac_inflation": "Inflation %",
        "mac_unemployment": "Chômage %",
        "mac_real_rate": "Taux réel %",
        "mac_expense_gdp": "Dépense % PIB",
        "group_fallback": "groupe",
        "vs_group_median": "▲ / ▼ vs médiane du groupe <strong style=\"color:#6890B0;\">{group}</strong>",
        "na_suffix": "n/d",
        "radar_vs_peers_label": "Radar — {country} vs groupe « {profile} »",
        "percentile_hover2": "<b>%{theta}</b><br>Percentile=%{r:.2f}<extra></extra>",
        "position_trend_label": "Position relative à la tendance structurelle PIB/habitant",
        "descriptive_reading_box": (
            "Lecture descriptive — pas une prédiction. La droite OLS (PIB/habitant → combined_score) "
            "montre où se situent en général les pays avec ce niveau de richesse. "
            "Le résidu dit si ce pays est au-dessus ou en-dessous de cette tendance structurelle, "
            "pas ce qu'il devrait être."
        ),
        "structural_residual_label": "Résidu structurel",
        "structural_residual_body": (
            'Score observé : <strong style="color:{rc};">{ccs:.2f}</strong> · '
            'Tendance pour ce PIB/habitant : <strong>{trend:.2f}</strong><br>'
            '<strong style="color:{rc};">{country} est {dir} '
            'de la tendance structurelle</strong> '
            '(résidu = {resid:+.2f}, soit {rpct:.0f}%)<br>'
            '<span style="font-size:12px;color:#7090B0;">'
            'PIB/habitant : ${cg:,.0f} — '
            'ρ(PIB/hab → combined_score) = +0,33 [E]</span>'
        ),
        "above_trend": "au-dessus",
        "below_trend": "en-dessous",
        "surplus_word": "excédentaire",
        "deficit_word": "déficitaire",
        "ca_pct_finding": (
            "<strong>CA% = {ca:+.1f}%</strong> ({cad} vs médiane {cam:+.1f}%) "
            "— deuxième signal [E] dominant (η²=0,241–0,250 sections I/J). {coherence}"
        ),
        "ca_coherent": "Signal cohérent : excédent courant et surperformance trade.",
        "ca_mixed": (
            "Signal mixte : la balance courante contredit la position relative au PIB/habitant — "
            "facteurs idiosyncratiques jouent un rôle."
        ),
        "scatter_explorer_label": "Scatter Explorer — Explorer n'importe quelle paire",
        "tc_combined_score": "Combined Score",
        "tc_structural_score": "Structural Score",
        "tc_export_share": "Part des exports",
        "tc_coverage_ratio": "Ratio de couverture",
        "tc_cagr_export": "CAGR Export",
        "tc_volatility_export": "Volatilité Export",
        "tc_balance_gdp": "Balance/PIB",
        "tc_export_pc": "Export/Habitant",
        "mc_gdp_capita": "PIB/Habitant",
        "mc_ca_gdp": "CA % PIB",
        "mc_inflation": "Inflation %",
        "mc_unemployment": "Chômage %",
        "mc_real_rate": "Taux réel %",
        "mc_expense_gdp": "Dépense % PIB",
        "trade_var_y_label": "Variable trade (Y)",
        "macro_ind_x_label": "Indicateur macro (X)",
        "how_to_read_expander": "Comment lire ce graphique",
        "how_to_read_scatter_html": """
                <div style='font-size:13px;color:#A8C0D8;line-height:1.9;'>
                <strong style='color:#EEF4FB;'>ρ (Spearman) et le ✓ / ✗</strong><br>
                ρ mesure la force de la relation monotone sur l'ensemble des ~170 pays.
                ✓ (p&lt;0.05) = la tendance globale est statistiquement réelle.
                ✗ = même la tendance générale est du bruit — il n'y a pas de signal entre ces deux variables.<br>
                <em>Attention : ✓ ne veut pas dire que la position d'un pays individuel est prédictible —
                le CV R² reste quasi-nul (Section M).</em><br><br>
                <strong style='color:#EEF4FB;'>Droite OLS (bleue pointillée)</strong><br>
                Approximation linéaire de la tendance centrale. Elle n'est pas un modèle de prédiction —
                c'est une référence visuelle pour voir l'écart d'un pays.<br><br>
                <strong style='color:#EEF4FB;'>Position du pays (étoile dorée) par rapport à la droite</strong><br>
                <span style='color:#10B981;'>Au-dessus</span> = ce pays performe mieux sur Y que ce que sa valeur X laisse attendre
                (résidu positif — avantage idiosyncratique).<br>
                <span style='color:#EF4444;'>En-dessous</span> = ce pays performe moins bien que la tendance
                (résidu négatif — facteurs limitants non capturés par X).<br><br>
                <strong style='color:#EEF4FB;'>Couleur des points</strong><br>
                Profil trade combiné — permet de voir si des clusters de profils
                se regroupent ou se dispersent sur la paire choisie.
                </div>
                """,

        "syn_label": "Synthèse — Ce que l'analyse révèle",
        "syn_scope_html": (
            '<span style="font-size:11px;font-weight:700;color:#4A6890;text-transform:uppercase;'
            'letter-spacing:.07em;">Périmètre</span>'
            '<span style="font-size:13px;color:#A8C0D8;">~<strong style="color:#EEF4FB;">170</strong> pays</span>'
            '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">16</strong> indicateurs macro World Bank</span>'
            '<span style="font-size:13px;color:#A8C0D8;">Médiane <strong style="color:#EEF4FB;">2015–2024</strong></span>'
            '<span style="font-size:13px;color:#A8C0D8;"><strong style="color:#EEF4FB;">2</strong> directions d\'analyse</span>'
            '<span style="font-size:12px;color:#F59E0B;font-style:italic;font-weight:600;">'
            '⚠ conclusions valables sur ce périmètre uniquement</span>'
        ),
        "syn_dir1_title": "Direction 1 · G → M",
        "syn_dir1_sub": "Fondamentaux macro → Structure trade",
        "syn_dir1_kpi1": "GDP → Export Share",
        "syn_dir1_kpi2": "GDP/cap → Export/cap",
        "syn_dir1_kpi3": "RF hors-échantillon",
        "syn_dir1_footer": "Signal réel [E] · prédiction hors-échantillon échoue",
        "syn_dir2_title": "Direction 2 · N",
        "syn_dir2_sub": "Structure trade → Fondamentaux macro",
        "syn_dir2_kpi1": "Export/cap → GDP/cap",
        "syn_dir2_kpi2": "RF Inflation (hors-éch.)",
        "syn_dir2_kpi3": "RF GDP/cap (hors-éch.)",
        "syn_dir2_footer": "Corrélations symétriques · prédiction encore plus fragile",
        "syn_conclusion_label": "Conclusion principale",
        "syn_conclusion_body": (
            "Sur ce périmètre d'analyse, la relation entre macro et trade <strong>se révèle "
            "structurelle et descriptive</strong> — pas prédictive. Les fondamentaux macro "
            "permettent d'expliquer <em>où</em> un pays se positionne dans le trade "
            "(niveau, profil), mais pas <em>comment</em> il va performer out-of-sample."
        ),
        "syn_conclusion_footer": (
            "Résultat conditionnel : 16 indicateurs macro WB généraux · ~170 pays · "
            "médiane 2015–2024 · pas d'indicateurs sectoriels ni institutionnels. "
            "Des données plus riches pourraient révéler une dimension prédictive."
        ),
        "syn_found_label": "Ce qui a été trouvé",
        "syn_found1": (
            "<strong>Taille prime sur la richesse</strong> (Section G–H) : GDP absolu "
            "explique mieux l'export share (ρ=0.94) que GDP_pc explique l'export/capita "
            "(ρ=0.82). Être grand détermine le volume ; être riche détermine l'efficacité."
        ),
        "syn_found2": (
            "<strong>Hiérarchie [E] observée</strong> (Section K) : sur ces données, "
            "GDP/capita > CA% > Real Rate > Inflation > Unemployment. "
            "Ces indicateurs montrent un signal faible à modéré sur la structure trade — "
            "hiérarchie à confirmer avec d'autres sources ou périmètres."
        ),
        "syn_found3": (
            "<strong>Empreinte macro par profil</strong> (Section L) : sur les 6 indicateurs "
            "analysés, GDP/capita et CA% ressortent comme les dimensions les plus "
            "différenciatrices entre profils trade. Les 4 autres indicateurs (inflation, "
            "chômage, taux réel, exports/GDP) montrent des différences faibles sur cette "
            "fenêtre et ce périmètre de données."
        ),
        "syn_found4": (
            "<strong>Trade Intensity révèle l'ouverture</strong> (Section D) : la richesse "
            "incline vers l'ouverture mais avec variance élevée. Les petits pays riches "
            "(Singapour, Luxembourg) sont les plus ouverts en relatif."
        ),
        "syn_notfound_label": "Ce qui n'a pas été trouvé",
        "syn_notfound1": (
            "<strong>Prédiction out-of-sample impossible</strong> (Section M) : RF R² élevé "
            "(0.72–0.91) mais CV R² quasi-nul (0.05–0.11). Le modèle mémorise, "
            "ne généralise pas. Les dynamiques (CAGR, volatilité) sont encore moins "
            "prédictibles (CV R² < 0)."
        ),
        "syn_notfound2": (
            "<strong>Direction 2 non robuste</strong> (Section N) : trade → macro "
            "ne prédit pas les outcomes macro hors-échantillon (CV R² ≈ 0 ou négatif). "
            "Sur ces données, la direction macro → trade semble plus informative, "
            "mais les corrélations Spearman étant symétriques, la causalité "
            "ne peut pas être établie depuis cette seule analyse."
        ),
        "syn_notfound3": (
            "<strong>Dynamiques non capturées</strong> (Section I–J) : sur ce dataset, "
            "la croissance et la volatilité trade ne sont pas expliquées par les 16 "
            "indicateurs macro disponibles (CV R² &lt; 0). Ces dynamiques impliquent "
            "probablement des facteurs non mesurés ici : politique sectorielle, "
            "géographie, chocs exogènes, accords bilatéraux."
        ),
        "syn_orientations_label": "Orientations pour aller plus loin",
        "syn_orientations_html": (
            '<div>→ <strong style="color:#EEF4FB;">Données panel</strong> : exploiter la dimension '
            'temporelle (pays × année) pour capturer les effets de politique commerciale</div>'
            '<div>→ <strong style="color:#EEF4FB;">Indicateurs sectoriels</strong> : diversification '
            'produit, complexité économique (ECI), dépendances géographiques</div>'
            '<div>→ <strong style="color:#EEF4FB;">Variables institutionnelles</strong> : qualité '
            'institutionnelle, accords commerciaux, barrières tarifaires</div>'
            '<div>→ <strong style="color:#EEF4FB;">Modèles non-linéaires validés</strong> : avec plus '
            'de pays ou données synthétiques pour pallier n=~170</div>'
            '<div>→ <strong style="color:#EEF4FB;">Chocs exogènes</strong> : COVID, matières premières — '
            'identifier les pays qui résistent vs amplifient les chocs macro</div>'
        ),
    },
}

LINK_FILES  = ROOT / "Analysis" / "Analysis_files" / "Link"
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
IND_FILES   = ROOT / "Analysis" / "Analysis_files" / "Indicators"

GDP    = "GDP (current US$)"
GDP_pc = "GDP per capita (current US$)"
EXP    = "Exports of goods and services (current US$)"
IMP    = "Imports of goods and services (current US$)"
POP    = "Population, total"
CA_PCT = "Current account balance (% of GDP)"
INFL   = "Inflation, consumer prices (annual %)"
UNEMP  = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
RATE    = "Real interest rate (%)"
EXP_PCT = "Exports of goods and services (% of GDP)"
EXPENSE = "Expense (% of GDP)"

_COMB_C  = {"Global Champion": "#10B981", "Competitive": "#60A5FA",
            "Vulnerable": "#F59E0B", "Fragile": "#EF4444"}
_META_C  = {
    "Global Leader":      "#10B981",
    "Established Power":  "#60A5FA",
    "Rising Challenger":  "#A78BFA",
    "Volatile Emerging":  "#F59E0B",
    "Peripheral":         "#6890B0",
}

# ════════════════════════════════════════════════════════════════
# DATA LOADING — same df_master as page 15
# ════════════════════════════════════════════════════════════════

@st.cache_data
def build_master():
    df_str  = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_comb = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_ind  = pd.read_csv(IND_FILES / "df_indicators.csv",
                          usecols=["country_name", "entity_type",
                                   "indicator_name", "year", "value"])
    df_sp   = pd.read_csv(LINK_FILES / "df_spearman.csv")

    ci = df_ind[df_ind["entity_type"] == "COUNTRY"].copy()
    df_latest = (ci.sort_values("year")
                 .groupby(["country_name", "indicator_name"])["value"].last()
                 .unstack().reset_index())
    df_latest.columns.name = None

    # Only pull columns unique to df_comb (avoid merge conflicts with df_str overlaps)
    _comb_unique = ["country_name", "combined_score", "f_norm_rank", "g_norm_rank",
                    "combined_profile", "strategic_quadrant_combined"]
    df_m = (df_str
            .merge(df_comb[[c for c in _comb_unique if c in df_comb.columns]],
                   on="country_name", how="inner")
            .merge(df_latest, on="country_name", how="left"))

    df_m["export_pc"]       = df_m["total_export"] / df_m[POP]
    df_m["import_pc"]       = df_m["total_import"] / df_m[POP]
    df_m["balance_dep"]     = (df_m[EXP] - df_m[IMP]) / df_m[GDP]
    df_m["export_dep"]      = df_m[EXP] / df_m[GDP]
    df_m["import_dep"]      = df_m[IMP] / df_m[GDP]
    df_m["trade_intensity"] = (df_m[EXP] + df_m[IMP]) / df_m[GDP]

    df_fi = pd.read_csv(LINK_FILES / "fi_matrix.csv")
    _ind_order = sorted(df_sp["indicator"].unique())
    df_fi.index = _ind_order

    return df_m, df_sp, df_fi

df_m, df_sp, df_fi = build_master()

# ════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1F3050", plot_bgcolor="#162035",
    font=dict(family="IBM Plex Sans", color="#C4D5E8", size=13),
    title_font=dict(color="#EEF4FB", size=15),
    legend=dict(bgcolor="#1F3050", bordercolor="#304870", borderwidth=1),
    margin=dict(l=60, r=40, t=50, b=60),
    xaxis=dict(gridcolor="#304870", linecolor="#304870"),
    yaxis=dict(gridcolor="#304870", linecolor="#304870"),
)

def _pl(fig, **kw):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kw})
    return fig

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_l, tab_m, tab_n, tab_p, tab_syn = st.tabs([
    t(STRINGS, "tab_l"),
    t(STRINGS, "tab_m"),
    t(STRINGS, "tab_n"),
    t(STRINGS, "tab_p"),
    t(STRINGS, "tab_syn"),
])

# ────────────────────────────────────────────────────────────────
# TAB L — PROFILE FINGERPRINT (RADAR)
# ────────────────────────────────────────────────────────────────
with tab_l:
    section_label(t(STRINGS, "l_label"))

    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'l_intro')}</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #34D399;'
        'border-radius:0 8px 8px 0;padding:.5rem .9rem;margin:.2rem 0 .9rem 0;font-size:12px;'
        f'color:#7090B0;">{t(STRINGS, "l_rationale")}</div>',
        unsafe_allow_html=True,
    )

    # ── Radar indicators ──
    RADAR_INDS = [GDP_pc, CA_PCT, INFL, UNEMP, RATE, EXP_PCT]
    RADAR_SHORT = {
        GDP_pc:   t(STRINGS, "radar_gdp_cap"),
        CA_PCT:   t(STRINGS, "radar_cur_acct"),
        INFL:     t(STRINGS, "radar_inflation"),
        UNEMP:    t(STRINGS, "radar_unemployment"),
        RATE:     t(STRINGS, "radar_real_rate"),
        EXP_PCT:  t(STRINGS, "radar_exports_gdp"),
    }
    RADAR_INV = {INFL, UNEMP, RATE}  # lower is better for trade

    # Compute percentile ranks
    df_r = df_m.copy()
    for ind in RADAR_INDS:
        if ind not in df_r.columns:
            continue
        ranked = df_r[ind].rank(pct=True, na_option="keep")
        df_r[f"_r_{ind}"] = (1 - ranked) if ind in RADAR_INV else ranked

    r_cols = [f"_r_{ind}" for ind in RADAR_INDS if f"_r_{ind}" in df_r.columns]
    r_labels = [RADAR_SHORT[ind] for ind in RADAR_INDS if f"_r_{ind}" in df_r.columns]

    # Group by profile selector
    profile_dim = st.radio(
        t(STRINGS, "profile_dim_label"),
        ["combined_profile", "meta_profile", "structural_profile"],
        format_func=lambda x: {
            "combined_profile":  t(STRINGS, "profile_combined"),
            "meta_profile":      t(STRINGS, "profile_meta"),
            "structural_profile": t(STRINGS, "profile_structural"),
        }.get(x, x),
        horizontal=True,
    )

    _dim_colors = {
        "combined_profile":  _COMB_C,
        "meta_profile":      _META_C,
        "structural_profile": {"Strong": "#10B981", "Good": "#60A5FA",
                               "Moderate": "#F59E0B", "Weak": "#EF4444"},
    }
    color_map_l = _dim_colors.get(profile_dim, {})
    palette_fallback = ["#10B981", "#60A5FA", "#F59E0B", "#EF4444", "#A78BFA", "#6890B0"]

    # Compute group means
    df_radar_data = (df_r.groupby(profile_dim)[r_cols].mean()
                     .fillna(0.5)
                     .rename(columns={f"_r_{ind}": RADAR_SHORT[ind] for ind in RADAR_INDS
                                      if f"_r_{ind}" in df_r.columns}))

    # Sample sizes per group
    n_per_group = df_r.groupby(profile_dim)["country_name"].count()

    # Group size chips
    _chips_html = "".join(
        f'<span style="display:inline-block;background:{color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])}22;'
        f'border:1px solid {color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])};'
        f'border-radius:12px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;color:#C4D5E8;">'
        f'<strong style="color:{color_map_l.get(str(g), palette_fallback[i % len(palette_fallback)])}">{g}</strong>'
        f' · n={n}</span>'
        for i, (g, n) in enumerate(n_per_group.items())
    )
    st.markdown(
        f'<div style="margin:.3rem 0 .7rem 0;">{_chips_html}</div>',
        unsafe_allow_html=True,
    )

    def _hex_to_rgba(hex_color, alpha=0.13):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_radar = go.Figure()
    for i, (prof, row) in enumerate(df_radar_data.iterrows()):
        vals = list(row.values)
        vals_closed = vals + [vals[0]]
        lbs_closed  = r_labels + [r_labels[0]]
        n = n_per_group.get(prof, 0)
        color = color_map_l.get(str(prof), palette_fallback[i % len(palette_fallback)])

        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=lbs_closed,
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(color=color, width=2),
            name=f"{prof} (n={n})",
            hovertemplate=t(STRINGS, "percentile_hover"),
        ))

    fig_radar.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis", "margin", "title_font")},
        title="",
        polar=dict(
            bgcolor="#162035",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#304870", tickfont=dict(size=10, color="#6890B0")),
            angularaxis=dict(gridcolor="#304870", tickfont=dict(size=12, color="#C4D5E8")),
        ),
        height=520,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Headline finding : 1/5 indicators ───────────────────
    st.markdown(
        '<div style="background:#0F1E2E;border:2px solid #34D399;border-radius:12px;'
        'padding:1.1rem 1.4rem;margin:.8rem 0 1rem 0;">'
        '<div style="display:flex;align-items:center;gap:1.2rem;">'
        '<div style="flex-shrink:0;text-align:center;">'
        '<div style="font-size:52px;font-weight:800;color:#34D399;line-height:1;">1</div>'
        f'<div style="font-size:13px;color:#7090B0;font-weight:600;">{t(STRINGS, "l_headline_out_of")}</div>'
        '</div>'
        '<div>'
        '<div style="font-size:15px;font-weight:700;color:#EEF4FB;margin-bottom:.35rem;">'
        f'{t(STRINGS, "l_headline_title")}'
        '</div>'
        f'<div style="font-size:13px;color:#A8C0D8;line-height:1.7;">{t(STRINGS, "l_headline_body")}</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(t(STRINGS, "l_how_to_read_expander")):
        st.markdown(t(STRINGS, "l_how_to_read_html"), unsafe_allow_html=True)

    finding(t(STRINGS, "l_finding"))

# ────────────────────────────────────────────────────────────────
# TAB M — TENTATIVE DE PREDICTION
# ────────────────────────────────────────────────────────────────
with tab_m:
    section_label(t(STRINGS, "m_label"))

    st.markdown(
        '<div style="background:#1A2840;border:1px solid #304870;border-left:3px solid #60A5FA;'
        'border-radius:0 10px 10px 0;padding:1rem 1.1rem;margin-bottom:1rem;">'
        '<div style="font-size:12px;font-weight:700;color:#93C5FD;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">{t(STRINGS, "m_question_label")}</div>'
        f'<div style="font-size:14px;color:#A8C0D8;line-height:1.8;">{t(STRINGS, "m_question_body")}</div>'
        '</div>',
        unsafe_allow_html=True)

    # Methodology box
    col_m1, col_m2 = st.columns([3, 2])

    with col_m1:
        section_label(t(STRINGS, "methodology_label"))
        st.markdown(
            f'<div style="font-size:13px;color:#A8C0D8;line-height:2.0;">{t(STRINGS, "methodology_html")}</div>',
            unsafe_allow_html=True)

    with col_m2:
        section_label(t(STRINGS, "key_results_label"))
        # Key results from notebook analysis — displayed as reference
        _results = [
            (t(STRINGS, "res_combined_score"),   0.78, 0.08, "#F59E0B"),
            (t(STRINGS, "res_structural_score"), 0.72, 0.05, "#F59E0B"),
            (t(STRINGS, "res_export_share"),     0.91, 0.11, "#F59E0B"),
            (t(STRINGS, "res_cagr_export"),      0.45, -0.03, "#EF4444"),
            (t(STRINGS, "res_volatility"),       0.51, -0.02, "#EF4444"),
        ]
        for label, r2_train, r2_cv, cv_color in _results:
            cv_txt = f"{r2_cv:+.2f}"
            cv_col = "#EF4444" if r2_cv < 0 else "#F59E0B" if r2_cv < 0.15 else "#10B981"
            st.markdown(
                f'<div style="background:#162035;border:1px solid #304870;border-radius:8px;'
                f'padding:.4rem .8rem;margin-bottom:4px;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;color:#A8C0D8;">{label}</span>'
                f'<span style="font-size:13px;">'
                f'<span style="color:#EEF4FB;">R²={r2_train:.2f}</span>'
                f'<span style="color:#304870;"> | </span>'
                f'<span style="color:{cv_col};">CV={cv_txt}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='font-size:11px;color:#4A6890;margin-top:.3rem;'>{t(STRINGS, 'r2_cv_legend')}</div>",
            unsafe_allow_html=True,
        )

    thin_divider()

    # Visual comparison: RF Train vs OLS R² vs RF CV
    section_label(t(STRINGS, "m_overfit_label"))
    _labels  = [t(STRINGS, "res_combined_score"), t(STRINGS, "res_structural_score"),
                t(STRINGS, "res_export_share"), t(STRINGS, "res_cagr_export"), t(STRINGS, "res_volatility")]
    _r2_tr   = [0.78, 0.72, 0.91, 0.45, 0.51]
    _r2_ols  = [0.22, 0.18, 0.28, 0.12, 0.15]  # OLS linear benchmark
    _r2_cv   = [0.08, 0.05, 0.11, -0.03, -0.02]
    _cv_cols = ["#F59E0B" if v > 0 else "#EF4444" for v in _r2_cv]

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        name=t(STRINGS, "trace_r2_train"),
        x=_labels, y=_r2_tr,
        marker_color="#304870", opacity=0.85,
    ))
    fig_m.add_trace(go.Bar(
        name=t(STRINGS, "trace_r2_ols"),
        x=_labels, y=_r2_ols,
        marker_color="#A78BFA", opacity=0.8,
    ))
    fig_m.add_trace(go.Bar(
        name=t(STRINGS, "trace_r2_cv"),
        x=_labels, y=_r2_cv,
        marker_color=_cv_cols,
    ))
    fig_m.add_hline(y=0, line_color="#6890B0", line_width=1)
    _pl(fig_m, title=t(STRINGS, "m_overfit_chart_title"),
        barmode="group", height=400,
        yaxis=dict(gridcolor="#304870", title="R²"),
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_m, use_container_width=True)

    finding(t(STRINGS, "m_overfit_finding"))

    thin_divider()
    section_label(t(STRINGS, "m_fi_label"))
    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'm_fi_intro')}</p>",
        unsafe_allow_html=True,
    )

    _FI_FLAG = {
        ind: flag
        for ind, flag in df_sp[["indicator", "ind_flag"]].drop_duplicates().values
    }
    _FI_SHORT = {
        ind: lbl
        for ind, lbl in df_sp[["indicator", "ind_label"]].drop_duplicates().values
    }
    _FLAG_C = {"[E]": "#10B981", "[S]": "#F59E0B", "[T]": "#EF4444"}

    # NOTE: these are the literal column names of the fi_matrix dataframe (df_fi)
    # and the st.selectbox value is matched directly against df_fi.columns — left
    # untranslated per the functional-lookup rule.
    _fi_targets = ["Combined Score", "Structural Score", "CAGR Export", "Volatility Export"]
    _fi_targets_avail = [tgt for tgt in _fi_targets if tgt in df_fi.columns]
    sel_fi = st.selectbox(
        t(STRINGS, "m_fi_selectbox_label"),
        _fi_targets_avail,
        key="sel_fi",
    )

    if sel_fi and sel_fi in df_fi.columns:
        _fi_col = df_fi[sel_fi].sort_values(ascending=True)
        _fi_colors = [_FLAG_C.get(_FI_FLAG.get(ind, ""), "#6890B0") for ind in _fi_col.index]
        _fi_labels = [_FI_SHORT.get(ind, ind) for ind in _fi_col.index]

        fig_fi = go.Figure(go.Bar(
            x=_fi_col.values,
            y=_fi_labels,
            orientation="h",
            marker_color=_fi_colors,
            text=[f"{v:.3f}" for v in _fi_col.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance = %{x:.4f}<extra></extra>",
        ))
        _pl(fig_fi,
            title=t(STRINGS, "m_fi_chart_title").format(sel_fi=sel_fi),
            xaxis_title=t(STRINGS, "m_fi_x"),
            height=400,
            margin=dict(l=160, r=80, t=65, b=40),
            xaxis=dict(gridcolor="#304870"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        _top_flag = _FI_FLAG.get(_fi_col.idxmax(), "?")
        _top_lbl  = _FI_SHORT.get(_fi_col.idxmax(), _fi_col.idxmax())
        finding(
            t(STRINGS, "m_fi_finding_intro").format(flag=_top_flag, lbl=_top_lbl, val=_fi_col.max())
            + (t(STRINGS, "m_fi_finding_ts") if _top_flag in ("[T]", "[S]") else t(STRINGS, "m_fi_finding_e"))
        )

    # Residual analysis — OLS Export Share vs GDP/capita
    thin_divider()
    section_label(t(STRINGS, "m_resid_label"))

    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'm_resid_intro')}</p>",
        unsafe_allow_html=True,
    )

    _df_res = df_m[["country_name", GDP_pc, "export_dep", "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    _df_res["export_pct"] = _df_res["export_dep"] * 100  # convert to %

    _xr = _df_res[GDP_pc].values
    _yr = _df_res["export_pct"].values
    _pr = np.polyfit(_xr, _yr, 1)
    _df_res["predicted"] = np.polyval(_pr, _xr)
    _df_res["residual"]  = _df_res["export_pct"] - _df_res["predicted"]

    _refs_res = ["USA", "China", "Germany", "Japan", "Brazil", "Korea, Rep."]
    _df_res["_gdp_f"] = _df_res[GDP_pc].apply(fmt_usd)
    _df_res_bg  = _df_res[~_df_res["country_name"].isin(_refs_res)]
    _df_res_ref = _df_res[_df_res["country_name"].isin(_refs_res)]

    fig_res = go.Figure()
    # Background points coloured by profile
    for cat, color in _COMB_C.items():
        sub = _df_res_bg[_df_res_bg["combined_profile"] == cat]
        if sub.empty:
            continue
        fig_res.add_trace(go.Scatter(
            x=sub[GDP_pc], y=sub["residual"],
            mode="markers", name=cat,
            customdata=sub[["_gdp_f"]].values,
            marker=dict(color=color, size=5, opacity=0.55),
            text=sub["country_name"],
            hovertemplate="<b>%{text}</b><br>GDP/cap=%{customdata[0]}<br>Residual=%{y:.2f}pp<extra></extra>",
        ))
    # Reference countries as gold stars
    if not _df_res_ref.empty:
        fig_res.add_trace(go.Scatter(
            x=_df_res_ref[GDP_pc], y=_df_res_ref["residual"],
            mode="markers+text",
            customdata=_df_res_ref[["_gdp_f"]].values,
            marker=dict(color="#FFD700", size=13, symbol="star",
                        line=dict(color="#FFFFFF", width=1.5)),
            text=_df_res_ref["country_name"], textposition="top center",
            textfont=dict(size=9, color="#FFD700"),
            hovertemplate="<b>%{text}</b><br>GDP/cap=%{customdata[0]}<br>Residual=%{y:.2f}pp<extra></extra>",
            name=t(STRINGS, "reference_label"), showlegend=False,
        ))
    fig_res.add_hline(y=0, line_color="#6890B0", line_width=1.5, line_dash="dash")
    _pl(fig_res,
        title=t(STRINGS, "m_resid_chart_title"),
        xaxis_title=t(STRINGS, "gdp_pc_usd_axis"),
        yaxis_title=t(STRINGS, "residual_axis"),
        height=420,
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_res, use_container_width=True)

    # Top outliers table
    _top_pos = _df_res.nlargest(3, "residual")[["country_name", "export_pct", "predicted", "residual"]]
    _top_neg = _df_res.nsmallest(3, "residual")[["country_name", "export_pct", "predicted", "residual"]]

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        finding(
            t(STRINGS, "m_resid_pos_finding").format(items=" · ".join(
                f"<em>{r['country_name']}</em> (+{r['residual']:.1f}pp)"
                for _, r in _top_pos.iterrows()
            ))
        )
    with col_res2:
        finding(
            t(STRINGS, "m_resid_neg_finding").format(items=" · ".join(
                f"<em>{r['country_name']}</em> ({r['residual']:.1f}pp)"
                for _, r in _top_neg.iterrows()
            ))
        )

    thin_divider()
    section_label(t(STRINGS, "interpretation_label"))
    finding(t(STRINGS, "m_interp_finding1"))
    finding(t(STRINGS, "m_interp_finding2"))
    finding(t(STRINGS, "m_interp_finding3"))
    finding(t(STRINGS, "m_interp_finding4"))

# ────────────────────────────────────────────────────────────────
# TAB N — DIRECTION 2
# ────────────────────────────────────────────────────────────────
with tab_n:
    section_label(t(STRINGS, "n_label"))

    st.markdown(
        '<div style="background:#1A2840;border:1px solid #304870;border-left:3px solid #A78BFA;'
        'border-radius:0 10px 10px 0;padding:1rem 1.1rem;margin-bottom:1rem;">'
        '<div style="font-size:12px;font-weight:700;color:#C4B5FD;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">{t(STRINGS, "n_question_label")}</div>'
        f'<div style="font-size:14px;color:#A8C0D8;line-height:1.8;">{t(STRINGS, "n_question_body")}</div>'
        '</div>',
        unsafe_allow_html=True)

    # ── Tableau de synthèse ───────────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n_summary_label"))

    _th = ("background:#162035;padding:.45rem .8rem;font-size:11px;font-weight:700;"
           "color:#7090B0;text-transform:uppercase;letter-spacing:.06em;"
           "border-bottom:1px solid #304870;")
    _td = "padding:.4rem .8rem;font-size:13px;border-bottom:1px solid #1F3050;"
    _tbl = t(STRINGS, "n_table_html").format(th=_th, td=_td)
    st.markdown(_tbl, unsafe_allow_html=True)

    # ── N.2 · Trade → GDP/capita ──────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n2_label"))

    finding(t(STRINGS, "n2_finding"))

    _refs_n2 = ["USA", "China", "Germany", "Japan", "Brazil"]
    _df_n2 = df_m[["country_name", "export_pc", "trade_intensity", GDP_pc,
                    "combined_profile"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    _df_n2["_xf_a"] = _df_n2["export_pc"].apply(fmt_usd)
    _df_n2["_gdp_f"] = _df_n2[GDP_pc].apply(fmt_usd)

    col_n2a, col_n2b = st.columns(2)

    with col_n2a:
        fig_n2a = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n2[(_df_n2["combined_profile"] == cat)
                         & (~_df_n2["country_name"].isin(_refs_n2))]
            if sub.empty:
                continue
            fig_n2a.add_trace(go.Scatter(
                x=sub["export_pc"], y=sub[GDP_pc],
                mode="markers", name=cat,
                customdata=sub[["_xf_a", "_gdp_f"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>GDP/cap=%{customdata[1]}<extra></extra>",
            ))
        _ref_n2a = _df_n2[_df_n2["country_name"].isin(_refs_n2)]
        if not _ref_n2a.empty:
            fig_n2a.add_trace(go.Scatter(
                x=_ref_n2a["export_pc"], y=_ref_n2a[GDP_pc],
                mode="markers+text",
                customdata=_ref_n2a[["_xf_a", "_gdp_f"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n2a["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>GDP/cap=%{customdata[1]}<extra></extra>",
                name=t(STRINGS, "reference_label"), showlegend=False,
            ))
        _xs2a = _df_n2["export_pc"].values
        _ys2a = _df_n2[GDP_pc].values
        _xl2a = np.linspace(_xs2a.min(), _xs2a.max(), 100)
        fig_n2a.add_trace(go.Scatter(
            x=_xl2a, y=np.polyval(np.polyfit(_xs2a, _ys2a, 1), _xl2a),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n2a.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text="ρ = +0.71 ✓", showarrow=False,
            xanchor="right", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981"),
        )
        _pl(fig_n2a, title=t(STRINGS, "n2a_chart_title"),
            xaxis_title=t(STRINGS, "export_pc_usd"),
            yaxis_title=t(STRINGS, "gdp_pc_usd"), height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=0.5,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n2a, use_container_width=True)

    with col_n2b:
        fig_n2b = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n2[(_df_n2["combined_profile"] == cat)
                         & (~_df_n2["country_name"].isin(_refs_n2))]
            if sub.empty:
                continue
            fig_n2b.add_trace(go.Scatter(
                x=sub["trade_intensity"], y=sub[GDP_pc],
                mode="markers", name=cat,
                customdata=sub[["_gdp_f"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Trade Int.=%{x:.2f}<br>GDP/cap=%{customdata[0]}<extra></extra>",
            ))
        _ref_n2b = _df_n2[_df_n2["country_name"].isin(_refs_n2)]
        if not _ref_n2b.empty:
            fig_n2b.add_trace(go.Scatter(
                x=_ref_n2b["trade_intensity"], y=_ref_n2b[GDP_pc],
                mode="markers+text",
                customdata=_ref_n2b[["_gdp_f"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n2b["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Trade Int.=%{x:.2f}<br>GDP/cap=%{customdata[0]}<extra></extra>",
                name=t(STRINGS, "reference_label"), showlegend=False,
            ))
        _xs2b = _df_n2["trade_intensity"].values
        _ys2b = _df_n2[GDP_pc].values
        _xl2b = np.linspace(_xs2b.min(), _xs2b.max(), 100)
        _rho_ti, _pv_ti = spearmanr(_xs2b, _ys2b)
        _sig_ti = "✓" if _pv_ti < 0.05 else "✗"
        fig_n2b.add_trace(go.Scatter(
            x=_xl2b, y=np.polyval(np.polyfit(_xs2b, _ys2b, 1), _xl2b),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n2b.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_ti:+.3f} {_sig_ti}", showarrow=False,
            xanchor="right", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_ti < 0.05 else "#6890B0"),
        )
        _pl(fig_n2b, title=t(STRINGS, "n2b_chart_title"),
            xaxis_title=t(STRINGS, "trade_intensity_xm_gdp"),
            yaxis_title=t(STRINGS, "gdp_pc_usd"), height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=0.5,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n2b, use_container_width=True)

    finding(t(STRINGS, "n2_china_finding"))

    # ── N.3 · Trade → Chômage ────────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n3_label"))

    _df_n3 = df_m[["country_name", "balance_dep", UNEMP, "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    _refs_n3 = ["Germany", "Japan", "China", "Brazil", "USA"]

    col_n3a, col_n3b = st.columns([3, 2])

    with col_n3a:
        fig_n3 = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n3[(_df_n3["combined_profile"] == cat)
                         & (~_df_n3["country_name"].isin(_refs_n3))]
            if sub.empty:
                continue
            fig_n3.add_trace(go.Scatter(
                x=sub["balance_dep"], y=sub[UNEMP],
                mode="markers", name=cat,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Balance/GDP=%{x:.3f}<br>Unemployment=%{y:.1f}%<extra></extra>",
            ))
        _ref_n3 = _df_n3[_df_n3["country_name"].isin(_refs_n3)]
        if not _ref_n3.empty:
            fig_n3.add_trace(go.Scatter(
                x=_ref_n3["balance_dep"], y=_ref_n3[UNEMP],
                mode="markers+text",
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n3["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Balance/GDP=%{x:.3f}<br>Unemployment=%{y:.1f}%<extra></extra>",
                name=t(STRINGS, "reference_label"), showlegend=False,
            ))
        _xs3 = _df_n3["balance_dep"].values
        _ys3 = _df_n3[UNEMP].values
        _rho_n3, _pv_n3 = spearmanr(_xs3, _ys3)
        _sig_n3 = "✓" if _pv_n3 < 0.05 else "✗"
        _xl3 = np.linspace(_xs3.min(), _xs3.max(), 100)
        fig_n3.add_trace(go.Scatter(
            x=_xl3, y=np.polyval(np.polyfit(_xs3, _ys3, 1), _xl3),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n3.add_annotation(
            x=0.5, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_n3:+.3f} {_sig_n3}", showarrow=False,
            xanchor="center", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_n3 < 0.05 else "#6890B0"),
        )
        _pl(fig_n3, title=t(STRINGS, "n3_chart_title"),
            xaxis_title=t(STRINGS, "balance_gdp_axis"),
            yaxis_title=t(STRINGS, "unemployment_axis"), height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n3, use_container_width=True)

    with col_n3b:
        finding(t(STRINGS, "n3_finding1"))
        finding(t(STRINGS, "n3_finding2"))
        finding(t(STRINGS, "n3_finding3"))
        finding(t(STRINGS, "n3_finding4"))

    # ── N.4 · Trade → Inflation ───────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n4_label"))

    _df_n4 = df_m[["country_name", "export_pc", INFL, "combined_profile"]].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    _df_n4["_xf"] = _df_n4["export_pc"].apply(fmt_usd)
    _refs_n4 = ["USA", "Germany", "Brazil", "China", "Japan"]

    col_n4a, col_n4b = st.columns([3, 2])

    with col_n4a:
        fig_n4 = go.Figure()
        for cat, color in _COMB_C.items():
            sub = _df_n4[(_df_n4["combined_profile"] == cat)
                         & (~_df_n4["country_name"].isin(_refs_n4))]
            if sub.empty:
                continue
            fig_n4.add_trace(go.Scatter(
                x=sub["export_pc"], y=sub[INFL],
                mode="markers", name=cat,
                customdata=sub[["_xf"]].values,
                marker=dict(color=color, size=6, opacity=0.7),
                text=sub["country_name"],
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>Inflation=%{y:.1f}%<extra></extra>",
            ))
        _ref_n4 = _df_n4[_df_n4["country_name"].isin(_refs_n4)]
        if not _ref_n4.empty:
            fig_n4.add_trace(go.Scatter(
                x=_ref_n4["export_pc"], y=_ref_n4[INFL],
                mode="markers+text",
                customdata=_ref_n4[["_xf"]].values,
                marker=dict(color="#FFD700", size=12, symbol="star",
                            line=dict(color="#FFFFFF", width=1.5)),
                text=_ref_n4["country_name"], textposition="top center",
                textfont=dict(size=9, color="#FFD700"),
                hovertemplate="<b>%{text}</b><br>Export/cap=%{customdata[0]}<br>Inflation=%{y:.1f}%<extra></extra>",
                name=t(STRINGS, "reference_label"), showlegend=False,
            ))
        _xs4 = _df_n4["export_pc"].values
        _ys4 = _df_n4[INFL].values
        _rho_n4, _pv_n4 = spearmanr(_xs4, _ys4)
        _sig_n4 = "✓" if _pv_n4 < 0.05 else "✗"
        _xl4 = np.linspace(_xs4.min(), _xs4.max(), 100)
        fig_n4.add_trace(go.Scatter(
            x=_xl4, y=np.polyval(np.polyfit(_xs4, _ys4, 1), _xl4),
            mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
        ))
        fig_n4.add_annotation(
            x=0.5, y=0.98, xref="paper", yref="paper",
            text=f"ρ = {_rho_n4:+.3f} {_sig_n4}", showarrow=False,
            xanchor="center", yanchor="top",
            bgcolor="#162035", bordercolor="#304870",
            font=dict(size=12, color="#10B981" if _pv_n4 < 0.05 else "#6890B0"),
        )
        _pl(fig_n4, title=t(STRINGS, "n4_chart_title"),
            xaxis_title=t(STRINGS, "export_pc_usd"),
            yaxis_title=t(STRINGS, "inflation_annual_axis"), height=420,
            legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0))
        st.plotly_chart(fig_n4, use_container_width=True)

    with col_n4b:
        finding(t(STRINGS, "n4_finding1"))
        finding(t(STRINGS, "n4_finding2"))
        finding(t(STRINGS, "n4_finding3"))

    # ── N.5 · Inverse-L ──────────────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n5_label"))

    st.markdown(
        '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #A78BFA;'
        'border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.4rem 0 .8rem 0;'
        f'font-size:13px;color:#7090B0;">{t(STRINGS, "n5_box")}</div>',
        unsafe_allow_html=True,
    )
    finding(t(STRINGS, "n5_finding1"))
    finding(t(STRINGS, "n5_finding2"))

    # ── N.6 · Inverse-M ──────────────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n6_label"))

    finding(t(STRINGS, "n6_finding_intro"))

    _n6_labels = [t(STRINGS, "n6_gdp_cap"), t(STRINGS, "radar_unemployment"), t(STRINGS, "radar_inflation")]
    _n6_train  = [0.82, 0.41, 0.38]
    _n6_cv     = [0.027, -0.011, -0.487]
    _n6_cv_c   = ["#F59E0B" if v > 0 else "#EF4444" for v in _n6_cv]

    fig_n6 = go.Figure()
    fig_n6.add_trace(go.Bar(
        name=t(STRINGS, "n6_trace_train"),
        x=_n6_labels, y=_n6_train,
        marker_color="#304870", opacity=0.8,
    ))
    fig_n6.add_trace(go.Bar(
        name=t(STRINGS, "n6_trace_cv"),
        x=_n6_labels, y=_n6_cv,
        marker_color=_n6_cv_c,
    ))
    fig_n6.add_hline(y=0, line_color="#6890B0", line_width=1)
    _pl(fig_n6, title=t(STRINGS, "n6_chart_title"),
        barmode="group", height=380,
        yaxis=dict(gridcolor="#304870", title="R²"),
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    st.plotly_chart(fig_n6, use_container_width=True)

    finding(t(STRINGS, "n6_finding1"))
    finding(t(STRINGS, "n6_finding2"))

    # ── Conclusion ────────────────────────────────────────────
    thin_divider()
    section_label(t(STRINGS, "n_conclusion_label"))
    finding(t(STRINGS, "n_conclusion1"))
    finding(t(STRINGS, "n_conclusion2"))
    finding(t(STRINGS, "n_conclusion3"))

# ────────────────────────────────────────────────────────────────
# TAB P — MON PAYS
# ────────────────────────────────────────────────────────────────
with tab_p:
    section_label(t(STRINGS, "p_label"))

    _countries_p = sorted(df_m["country_name"].dropna().tolist())
    _def_p       = "France" if "France" in _countries_p else _countries_p[0]
    _country_p   = st.selectbox(
        t(STRINGS, "country_label"), _countries_p,
        index=_countries_p.index(_def_p),
        key="sel_country_p",
    )

    _row_p = df_m[df_m["country_name"] == _country_p]
    if not _row_p.empty:
        _r = _row_p.iloc[0]

        # ── Scorecard ────────────────────────────────────────────
        thin_divider()
        section_label(t(STRINGS, "scorecard_label").format(country=_country_p))

        _chip_cols = st.columns(6)
        _chips = [
            (t(STRINGS, "chip_combined_score"),   f"{_r['combined_score']:.2f}"      if pd.notna(_r.get("combined_score"))  else "—", "#10B981"),
            (t(STRINGS, "chip_structural_score"), f"{_r['structural_score']:.2f}"    if pd.notna(_r.get("structural_score")) else "—", "#60A5FA"),
            (t(STRINGS, "chip_f_norm_rank"),      f"{_r.get('f_norm_rank', float('nan')):.2f}" if pd.notna(_r.get("f_norm_rank")) else "—", "#A78BFA"),
            (t(STRINGS, "chip_g_norm_rank"),      f"{_r.get('g_norm_rank', float('nan')):.2f}" if pd.notna(_r.get("g_norm_rank")) else "—", "#F59E0B"),
            (t(STRINGS, "chip_combined_profile"), str(_r.get("combined_profile", "—")), "#34D399"),
            (t(STRINGS, "chip_meta_profile"),     str(_r.get("meta_profile",     "—")), "#6890B0"),
        ]
        for _cc, (_lbl, _val, _clr) in zip(_chip_cols, _chips):
            _cc.markdown(
                f'<div style="background:#162035;border:1px solid #304870;'
                f'border-top:3px solid {_clr};border-radius:8px;padding:.5rem .7rem;'
                f'text-align:center;">'
                f'<div style="font-size:10px;color:#6890B0;text-transform:uppercase;'
                f'letter-spacing:.06em;">{_lbl}</div>'
                f'<div style="font-size:16px;font-weight:700;color:{_clr};'
                f'margin-top:.2rem;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:.7rem 0;'></div>", unsafe_allow_html=True)

        _profile_p = str(_r.get("combined_profile", ""))
        _peers_p   = df_m[df_m["combined_profile"] == _profile_p] if _profile_p else df_m

        _bm_all = [
            ("export_share_structural",  t(STRINGS, "bm_export_share")),
            ("import_share_structural",  t(STRINGS, "bm_import_share")),
            ("coverage_ratio_structural", t(STRINGS, "bm_coverage_ratio")),
            ("cagr_export",              t(STRINGS, "bm_cagr_export")),
            ("volatility_export",        t(STRINGS, "bm_export_volatility")),
            ("balance_dep",              t(STRINGS, "bm_balance_gdp")),
        ]
        _bm_avail = [(c, l) for c, l in _bm_all if c in df_m.columns]
        _bm_cols, _bm_labs = zip(*_bm_avail) if _bm_avail else ([], [])

        col_bar_p, col_mac_p = st.columns([3, 2])
        with col_bar_p:
            fig_bar_p = go.Figure()
            fig_bar_p.add_trace(go.Bar(
                name=f"{t(STRINGS, 'median_label')} {_profile_p or t(STRINGS, 'global_fallback')}",
                x=list(_bm_labs), y=[_peers_p[c].median() for c in _bm_cols],
                marker_color="#304870", opacity=0.7,
            ))
            fig_bar_p.add_trace(go.Bar(
                name=_country_p,
                x=list(_bm_labs), y=[_r.get(c, np.nan) for c in _bm_cols],
                marker_color="#10B981",
            ))
            _pl(fig_bar_p,
                title=t(STRINGS, "trade_metrics_vs_median_title").format(profile=_profile_p),
                barmode="group", height=380,
                margin=dict(l=40, r=20, t=50, b=80),
                xaxis=dict(tickangle=-30, gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"),
                legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                            yanchor="top", bgcolor="rgba(26,40,64,0.85)",
                            bordercolor="#304870", borderwidth=1),
            )
            st.plotly_chart(fig_bar_p, use_container_width=True)

        with col_mac_p:
            _KEY_MAC = [
                ("GDP per capita (current US$)",            t(STRINGS, "mac_gdp_cap")),
                ("Current account balance (% of GDP)",      t(STRINGS, "mac_current_account")),
                ("Inflation, consumer prices (annual %)",   t(STRINGS, "mac_inflation")),
                ("Unemployment, total (% of total labor force) (modeled ILO estimate)", t(STRINGS, "mac_unemployment")),
                ("Real interest rate (%)",                  t(STRINGS, "mac_real_rate")),
                ("Expense (% of GDP)",                      t(STRINGS, "mac_expense_gdp")),
            ]
            _grp_lbl = _profile_p or t(STRINGS, "group_fallback")
            st.markdown(
                f'<div style="font-size:10px;color:#4A6890;margin-bottom:4px;">'
                f'{t(STRINGS, "vs_group_median").format(group=_grp_lbl)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            for _ind, _lbl in _KEY_MAC:
                if _ind not in df_m.columns:
                    continue
                _val  = _r.get(_ind)
                _pmed = _peers_p[_ind].median() if _ind in _peers_p.columns else np.nan
                if pd.isna(_val):
                    st.markdown(
                        f'<div style="background:#162035;border:1px solid #2A3848;'
                        f'border-radius:7px;padding:.35rem .7rem;margin-bottom:3px;'
                        f'font-size:12px;color:#4A6890;">{_lbl} — {t(STRINGS, "na_suffix")}</div>',
                        unsafe_allow_html=True)
                    continue
                _ap = (_val > _pmed) if pd.notna(_pmed) else None
                _ap_c     = ("#10B981" if _ap else "#EF4444") if _ap is not None else "#6890B0"
                _ap_arrow = ("▲" if _ap else "▼") if _ap is not None else "—"
                _disp = (f"${_val:,.0f}" if "per capita" in _ind
                         else f"{_val:+.1f}%" if _val < 0 else f"{_val:.1f}%")
                st.markdown(
                    f'<div style="background:#162035;border:1px solid #304870;'
                    f'border-radius:7px;padding:.35rem .7rem;margin-bottom:3px;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:12px;color:#8AABB0;">{_lbl}</span>'
                    f'<span style="font-size:13px;font-weight:600;color:#EEF4FB;">'
                    f'{_disp}&nbsp;'
                    f'<span style="color:{_ap_c};font-size:12px;">{_ap_arrow}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Radar vs peers ────────────────────────────────────────
        thin_divider()
        section_label(t(STRINGS, "radar_vs_peers_label").format(country=_country_p, profile=_profile_p))

        _RI = [GDP_pc, CA_PCT, INFL, UNEMP, RATE, EXP_PCT]
        _RS = {GDP_pc: t(STRINGS, "radar_gdp_cap"), CA_PCT: t(STRINGS, "radar_cur_acct"),
               INFL: t(STRINGS, "radar_inflation"), UNEMP: t(STRINGS, "radar_unemployment"),
               RATE: t(STRINGS, "radar_real_rate"), EXP_PCT: t(STRINGS, "radar_exports_gdp")}
        _RV = {INFL, UNEMP, RATE}

        df_rp = df_m.copy()
        for _ind in _RI:
            if _ind not in df_rp.columns:
                continue
            _rnk = df_rp[_ind].rank(pct=True, na_option="keep")
            df_rp[f"__rp_{_ind}"] = (1 - _rnk) if _ind in _RV else _rnk

        _rp_cs = [f"__rp_{i}" for i in _RI if f"__rp_{i}" in df_rp.columns]
        _rp_ls = [_RS[i] for i in _RI if f"__rp_{i}" in df_rp.columns]

        _crow = df_rp[df_rp["country_name"] == _country_p][_rp_cs]
        _prow = df_rp[df_rp["combined_profile"] == _profile_p][_rp_cs] if _profile_p else df_rp[_rp_cs]

        if not _crow.empty:
            _cv = list(_crow.iloc[0].fillna(0.5).values)
            _pv = list(_prow.median().fillna(0.5).values)
            _lc = _rp_ls + [_rp_ls[0]]

            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatterpolar(
                r=_pv + [_pv[0]], theta=_lc,
                fill="toself", fillcolor="rgba(96,165,250,0.09)",
                line=dict(color="#60A5FA", width=1.5, dash="dot"),
                name=f"{t(STRINGS, 'median_label')} {_profile_p}",
                hovertemplate=t(STRINGS, "percentile_hover2"),
            ))
            fig_rp.add_trace(go.Scatterpolar(
                r=_cv + [_cv[0]], theta=_lc,
                fill="toself", fillcolor="rgba(16,185,129,0.18)",
                line=dict(color="#10B981", width=2.5),
                name=_country_p,
                hovertemplate=t(STRINGS, "percentile_hover2"),
            ))
            fig_rp.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis", "yaxis", "margin", "legend", "title_font")},
                title="",
                polar=dict(
                    bgcolor="#162035",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#304870",
                                    tickfont=dict(size=9, color="#6890B0")),
                    angularaxis=dict(gridcolor="#304870",
                                     tickfont=dict(size=12, color="#C4D5E8")),
                ),
                height=440,
                margin=dict(l=60, r=60, t=60, b=60),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_rp, use_container_width=True)

        # ── Narrative ─────────────────────────────────────────────
        thin_divider()
        section_label(t(STRINGS, "position_trend_label"))

        st.markdown(
            '<div style="background:#182028;border:1px solid #2A3848;border-left:3px solid #6890B0;'
            'border-radius:0 8px 8px 0;padding:.4rem .8rem;margin:.2rem 0 .7rem 0;font-size:12px;'
            f'color:#6890B0;">{t(STRINGS, "descriptive_reading_box")}</div>',
            unsafe_allow_html=True,
        )

        _df_ols = df_m[[GDP_pc, "combined_score"]].dropna()
        _df_ols = _df_ols[_df_ols[GDP_pc] > 0]
        if len(_df_ols) > 20:
            _xlog  = np.log10(_df_ols[GDP_pc].values)
            _ycs   = _df_ols["combined_score"].values
            _poly  = np.polyfit(_xlog, _ycs, 1)
            _cg    = _r.get(GDP_pc)
            _ccs   = _r.get("combined_score")
            if pd.notna(_cg) and pd.notna(_ccs) and _cg > 0:
                _trend = np.polyval(_poly, np.log10(_cg))
                _resid = _ccs - _trend
                _rpct  = abs(_resid / _trend * 100) if _trend else 0
                _abv   = _resid > 0
                _rc    = "#10B981" if _abv else "#EF4444"
                _dir   = t(STRINGS, "above_trend") if _abv else t(STRINGS, "below_trend")
                st.markdown(
                    f'<div style="background:#1A2840;border:1px solid #304870;'
                    f'border-left:3px solid {_rc};border-radius:0 10px 10px 0;'
                    f'padding:.8rem 1.1rem;margin-bottom:.8rem;">'
                    f'<div style="font-size:11px;font-weight:700;color:{_rc};'
                    f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.4rem;">'
                    f'{t(STRINGS, "structural_residual_label")}</div>'
                    f'<div style="font-size:14px;color:#EEF4FB;line-height:1.9;">'
                    + t(STRINGS, "structural_residual_body").format(
                        rc=_rc, ccs=_ccs, trend=_trend, country=_country_p,
                        dir=_dir, resid=_resid, rpct=_rpct, cg=_cg)
                    + '</div></div>',
                    unsafe_allow_html=True,
                )
                _ca  = _r.get(CA_PCT)
                _cam = df_m[CA_PCT].median()
                if pd.notna(_ca):
                    _cad = t(STRINGS, "surplus_word") if _ca > _cam else t(STRINGS, "deficit_word")
                    _cac = "#10B981" if _ca > _cam else "#EF4444"
                    _coherent = (_abv and _ca > _cam) or (not _abv and _ca < _cam)
                    finding(
                        t(STRINGS, "ca_pct_finding").format(
                            ca=_ca, cad=_cad, cam=_cam,
                            coherence=(t(STRINGS, "ca_coherent") if _coherent else t(STRINGS, "ca_mixed")))
                    )

        # ── Scatter Explorer ──────────────────────────────────────
        thin_divider()
        section_label(t(STRINGS, "scatter_explorer_label"))

        _TC = {k: v for k, v in {
            "combined_score":           t(STRINGS, "tc_combined_score"),
            "structural_score":         t(STRINGS, "tc_structural_score"),
            "export_share_structural":  t(STRINGS, "tc_export_share"),
            "coverage_ratio_structural": t(STRINGS, "tc_coverage_ratio"),
            "cagr_export":              t(STRINGS, "tc_cagr_export"),
            "volatility_export":        t(STRINGS, "tc_volatility_export"),
            "balance_dep":              t(STRINGS, "tc_balance_gdp"),
            "export_pc":                t(STRINGS, "tc_export_pc"),
        }.items() if k in df_m.columns}

        _MC = {k: v for k, v in {
            GDP_pc:  t(STRINGS, "mc_gdp_capita"),
            CA_PCT:  t(STRINGS, "mc_ca_gdp"),
            INFL:    t(STRINGS, "mc_inflation"),
            UNEMP:   t(STRINGS, "mc_unemployment"),
            RATE:    t(STRINGS, "mc_real_rate"),
            EXPENSE: t(STRINGS, "mc_expense_gdp"),
        }.items() if k in df_m.columns}

        _c1, _c2 = st.columns(2)
        _stv = _c1.selectbox(t(STRINGS, "trade_var_y_label"), list(_TC.keys()),
                              format_func=lambda x: _TC.get(x, x), key="p_tv")
        _smi = _c2.selectbox(t(STRINGS, "macro_ind_x_label"), list(_MC.keys()),
                              format_func=lambda x: _MC.get(x, x), key="p_mi")

        _USD_COLS = {GDP_pc, "export_pc", "import_pc"}
        _dsc = df_m[["country_name", _stv, _smi, "combined_profile"]].dropna().copy()
        if len(_dsc) >= 10:
            _rho_s, _pv_s = spearmanr(_dsc[_stv], _dsc[_smi])
            _sig_s = _pv_s < 0.05
            _fx = lambda v: fmt_usd(v) if _smi in _USD_COLS else f"{v:.3f}"
            _fy = lambda v: fmt_usd(v) if _stv in _USD_COLS else f"{v:.3f}"
            _dsc["_xf"] = _dsc[_smi].apply(_fx)
            _dsc["_yf"] = _dsc[_stv].apply(_fy)
            _issel = _dsc["country_name"] == _country_p
            _oth   = _dsc[~_issel]
            _sel   = _dsc[_issel]
            _COMB_C2 = {"Global Champion":"#10B981","Competitive":"#60A5FA",
                        "Vulnerable":"#F59E0B","Fragile":"#EF4444"}
            _ptc = [_COMB_C2.get(str(p), "#304870") for p in _oth["combined_profile"]]
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=_oth[_smi], y=_oth[_stv], mode="markers",
                customdata=_oth[["_xf", "_yf"]].values,
                marker=dict(color=_ptc, size=6, opacity=0.65),
                text=_oth["country_name"],
                hovertemplate="<b>%{text}</b><br>X=%{customdata[0]} · Y=%{customdata[1]}<extra></extra>",
                name=t(STRINGS, "other_countries_label"),
            ))
            if not _sel.empty:
                fig_sc.add_trace(go.Scatter(
                    x=_sel[_smi], y=_sel[_stv], mode="markers+text",
                    customdata=_sel[["_xf", "_yf"]].values,
                    marker=dict(color="#FFD700", size=14, symbol="star",
                                line=dict(color="#FFFFFF", width=2)),
                    text=[_country_p], textposition="top center",
                    textfont=dict(size=12, color="#FFFFFF"),
                    hovertemplate=f"<b>{_country_p}</b><br>X=%{{customdata[0]}} · Y=%{{customdata[1]}}<extra></extra>",
                    name=_country_p,
                ))
            _xs2 = _dsc[_smi].values
            _ys2 = _dsc[_stv].values
            _p2  = np.polyfit(_xs2, _ys2, 1)
            _xl2 = np.linspace(_xs2.min(), _xs2.max(), 100)
            fig_sc.add_trace(go.Scatter(
                x=_xl2, y=np.polyval(_p2, _xl2), mode="lines", name="OLS",
                line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
            ))
            _fsc = df_sp[df_sp["indicator"] == _smi]["ind_flag"].values
            _fsc_v = _fsc[0] if len(_fsc) > 0 else "—"
            _fsc_c = {"[E]":"#10B981","[T]":"#EF4444","[S]":"#F59E0B"}.get(_fsc_v, "#60A5FA")
            fig_sc.add_annotation(
                x=0.98, y=0.98, xref="paper", yref="paper",
                text=(f"ρ={_rho_s:+.3f} {'✓' if _sig_s else '✗ n.s.'} · "
                      f"<span style='color:{_fsc_c}'>{_fsc_v}</span>  n={len(_dsc)}"),
                showarrow=False, xanchor="right", yanchor="top",
                bgcolor="#162035", bordercolor="#304870",
                font=dict(size=12, color="#C4D5E8"),
            )
            _pl(fig_sc,
                title=f"{_MC.get(_smi, _smi)} × {_TC.get(_stv, _stv)}",
                xaxis_title=_MC.get(_smi, _smi),
                yaxis_title=_TC.get(_stv, _stv),
                height=460,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            with st.expander(t(STRINGS, "how_to_read_expander")):
                st.markdown(t(STRINGS, "how_to_read_scatter_html"), unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB SYNTHESE
# ────────────────────────────────────────────────────────────────
with tab_syn:
    section_label(t(STRINGS, "syn_label"))

    # ── Contexte & données ────────────────────────────────────
    st.markdown(
        '<div style="background:#162035;border:1px solid #304870;border-radius:8px;'
        'padding:.55rem 1rem;margin-bottom:.9rem;display:flex;flex-wrap:wrap;gap:1.4rem;'
        f'align-items:center;">{t(STRINGS, "syn_scope_html")}</div>',
        unsafe_allow_html=True,
    )

    # ── Schéma des deux directions ────────────────────────────
    _cell = ("background:#162035;border:1px solid #304870;border-radius:10px;"
             "padding:.8rem 1rem;")
    _kpi_n = ("font-size:22px;font-weight:800;line-height:1;")
    _kpi_l = ("font-size:10px;color:#6890B0;margin-top:.15rem;")
    _arrow = ('font-size:28px;font-weight:300;color:#304870;text-align:center;'
              'display:flex;align-items:center;justify-content:center;')

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 56px 1fr;gap:.5rem;'
        f'margin-bottom:1rem;">'

        # Direction 1 box
        f'<div style="{_cell}border-top:3px solid #10B981;">'
        f'<div style="font-size:11px;font-weight:700;color:#10B981;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">{t(STRINGS, "syn_dir1_title")}</div>'
        f'<div style="font-size:12px;color:#7090B0;margin-bottom:.6rem;">'
        f'{t(STRINGS, "syn_dir1_sub")}</div>'
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
        f'<div><div style="{_kpi_n}color:#10B981;">ρ=0.94</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir1_kpi1")}</div></div>'
        f'<div><div style="{_kpi_n}color:#10B981;">ρ=0.82</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir1_kpi2")}</div></div>'
        f'<div><div style="{_kpi_n}color:#F59E0B;">CV≈0.05</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir1_kpi3")}</div></div>'
        f'</div>'
        f'<div style="font-size:11px;color:#4A6890;margin-top:.5rem;">'
        f'{t(STRINGS, "syn_dir1_footer")}</div>'
        f'</div>'

        # Arrow
        f'<div style="{_arrow}">⇄</div>'

        # Direction 2 box
        f'<div style="{_cell}border-top:3px solid #A78BFA;">'
        f'<div style="font-size:11px;font-weight:700;color:#A78BFA;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:.5rem;">{t(STRINGS, "syn_dir2_title")}</div>'
        f'<div style="font-size:12px;color:#7090B0;margin-bottom:.6rem;">'
        f'{t(STRINGS, "syn_dir2_sub")}</div>'
        f'<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
        f'<div><div style="{_kpi_n}color:#A78BFA;">ρ=0.71</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir2_kpi1")}</div></div>'
        f'<div><div style="{_kpi_n}color:#EF4444;">CV≈−0.5</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir2_kpi2")}</div></div>'
        f'<div><div style="{_kpi_n}color:#EF4444;">CV≈0.03</div>'
        f'<div style="{_kpi_l}">{t(STRINGS, "syn_dir2_kpi3")}</div></div>'
        f'</div>'
        f'<div style="font-size:11px;color:#4A6890;margin-top:.5rem;">'
        f'{t(STRINGS, "syn_dir2_footer")}</div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Conclusion principale ─────────────────────────────────
    st.markdown(
        '<div style="background:#1F3050;border:1px solid #304870;border-radius:14px;'
        'padding:1.2rem 1.4rem;margin-bottom:1rem;">'
        '<div style="font-size:12px;font-weight:700;color:#34D399;text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:.8rem;">{t(STRINGS, "syn_conclusion_label")}</div>'
        f'<div style="font-size:15px;color:#EEF4FB;line-height:1.9;">{t(STRINGS, "syn_conclusion_body")}</div>'
        '<div style="font-size:12px;color:#4A6890;margin-top:.7rem;font-style:italic;'
        f'border-top:1px solid #304870;padding-top:.6rem;">{t(STRINGS, "syn_conclusion_footer")}</div>'
        '</div>',
        unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        section_label(t(STRINGS, "syn_found_label"))
        finding(t(STRINGS, "syn_found1"))
        finding(t(STRINGS, "syn_found2"))
        finding(t(STRINGS, "syn_found3"))
        finding(t(STRINGS, "syn_found4"))

    with col_s2:
        section_label(t(STRINGS, "syn_notfound_label"))
        finding(t(STRINGS, "syn_notfound1"))
        finding(t(STRINGS, "syn_notfound2"))
        finding(t(STRINGS, "syn_notfound3"))

    thin_divider()
    section_label(t(STRINGS, "syn_orientations_label"))

    st.markdown(
        f'<div style="font-size:13px;color:#A8C0D8;line-height:2.1;">{t(STRINGS, "syn_orientations_html")}</div>',
        unsafe_allow_html=True)

thin_divider()
