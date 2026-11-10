"""
Trade × Indicators — Link · Signals & Links
For each trade dimension, which macro indicators explain it best?
Sections D · G · H · I · J · K du notebook Link
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
from utils.styles import inject_styles, kpi, section_label, thin_divider, finding, fmt_usd
from utils.i18n import t

inject_styles()

STRINGS = {
    "en": {
        "eyebrow": "Link · Trade × Indicators · Page 1 of 2",
        "title": "Signals & Links",
        "caption": (
            "For each trade dimension, which macro indicator explains it best? "
            "Sections D · G · H · I · J · K — 16 indicators × 44 trade variables"
        ),
        "kpi_countries": "Countries analyzed",
        "kpi_sig_e": "Sig. [E] pairs",
        "kpi_max_signal": "Max signal (all)",
        "kpi_top_signal_e": "Top signal [E]",
        "ef_label": "E+F · Indicator Classification — Analytical Basis",
        "ef_intro": (
            "Before analyzing each trade dimension (G–K), sections E and F of the notebook "
            "classified the 16 macro indicators into three families according to their analytical nature. "
            "This classification is used throughout the page to distinguish "
            "causal signals from statistical artefacts."
        ),
        "flag_t_title": "[T] Tautological",
        "flag_t_body": (
            " — indicator is directly derived "
            "from trade flows (e.g. Exports USD). "
            "High correlation is structural, not causal."
        ),
        "flag_s_title": "[S] Size-driven",
        "flag_s_body": (
            " — indicator reflects country "
            "size (GDP, Population). Correlation with trade volumes is scale effect, "
            "not economic signal."
        ),
        "flag_e_title": "[E] Economic",
        "flag_e_body": (
            " — genuine macro signal "
            "independent of trade definition (e.g. GDP/capita, Inflation). "
            "Correlations here are analytically meaningful."
        ),
        "ef_footer": (
            "<strong style='color:#6890B0;'>Section E</strong> · 704 Spearman pairs "
            "(44 variables × 16 indicators). "
            "<strong style='color:#6890B0;'>Section F</strong> · RF Feature Importance + "
            "KW η² confirm the [E] vs [S] ranking. "
            "Correlation heatmaps are available at the beginning of each tab below "
            "— they show which [E] (and [S]) indicators explain each variable family."
        ),
        "tab_d": "D · Trade/GDP Bridge",
        "tab_gh": "G–H · Exports & Imports",
        "tab_ij": "I–J · Balance & Profiles",
        "tab_k": "K · Scores",
        "question_label": "Question",
        "note_label": "Analytical note",

        "d_intro": (
            "Three derived metrics that directly link trade structure and macro: "
            "Trade Intensity = (X+M)/GDP · Elasticity = CAGR_export/CAGR_GDP · "
            "Trade per Capita = volume normalized by population."
        ),
        "d_heat_title": "Correlations [E] — Trade/GDP Bridge variables (Tab D) · * = p<0.05",
        "d1_label": "D.1 · Trade Intensity",
        "d1_question": "Does a country's wealth determine its trade openness?",
        "d1_chart_title": "Trade Intensity (X+M)/GDP vs GDP/capita",
        "gdp_pc_log": "GDP per capita (US$, log)",
        "trade_intensity_title": "Trade Intensity (X+M)/GDP",
        "d1_finding": (
            "<strong>Yes, but non-linear</strong>: wealthier countries tend toward higher trade "
            "intensity, with increasing dispersion. Large countries "
            "(Leaders) have variable intensities depending on their domestic structure — "
            "USA and China have moderate intensity despite their size, "
            "while small wealthy countries (Singapore, Luxembourg) are the most open."
        ),

        "d2_label": "D.2 · Trade/GDP Elasticity — 6 Situations",
        "d2_question": "Does foreign trade grow faster than the economy? In which profiles?",
        "d2_chart_title": "CAGR Export vs CAGR GDP — diagonal = elasticity 1",
        "cagr_gdp_pct": "CAGR GDP (%)",
        "cagr_export_pct": "CAGR Export (%)",
        "d2b_chart_title": "Country distribution<br>by situation (2015–2024)",
        "d2b_x": "No. of countries",
        "d2_finding": (
            "<strong>Majority GDP-led</strong>: most countries have export growth "
            "slower than their GDP (elasticity < 1). Export-led countries are often "
            "emerging economies in a phase of global integration. "
            "<strong>Export Resilient</strong>: countries whose exports grow even when "
            "the economy contracts — a sign of robust external competitiveness."
        ),
        "d2_expander_title": "Definition of the 6 situations",
        "d2_table_html": """
        <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Situation</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Condition</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Interpretation</th>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#10B981;font-weight:600;">Export-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; CAGR_GDP &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              Exports grow faster than the economy — dominant external engine</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#60A5FA;font-weight:600;">GDP-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_GDP &gt; CAGR_X &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              Economy grows faster than exports — growth driven by domestic demand</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#A78BFA;font-weight:600;">Export Resilient</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; 0, CAGR_GDP ≤ 0</td>
            <td style="padding:.4rem .8rem;">
              Exports grow while the economy contracts — robust external competitiveness</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#F59E0B;font-weight:600;">Export Contraction</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X ≤ 0, CAGR_GDP &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              Economy grows but exports fall — domestic substitution or loss of competitiveness</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#FB923C;font-weight:600;">Moderate Decline</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_GDP &lt; 0, |X| &lt; |GDP|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, exports hold up better than GDP</td>
          </tr>
          <tr>
            <td style="padding:.4rem .8rem;">
              <span style="color:#EF4444;font-weight:600;">Severe Decline</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_GDP &lt; 0, |X| &gt; |GDP|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, exports collapse faster than GDP</td>
          </tr>
        </table>
        <div style="margin-top:.8rem;font-size:12px;color:#4A6890;">
          CAGR_X = compound annual growth rate of exports (2015–2024) ·
          CAGR_GDP = same calculation for nominal GDP ·
          Elasticity = CAGR_X / CAGR_GDP
        </div>
        </div>
        """,

        "d3_label": "D.3 · Export per Capita vs GDP/capita",
        "d3_question": (
            "Do wealthy countries export more per capita? "
            "Which countries reposition when normalized by population?"
        ),
        "d3_chart_title": "Export per Capita vs GDP/capita",
        "export_pc_log": "Export per Capita (US$, log)",
        "d3b_chart_title": (
            "Rank Mover — Volume vs Per Capita<br>"
            "<span style='font-size:11px;color:#6890B0;'>"
            "Δ > 0 : better per capita rank than by volume</span>"
        ),
        "d3b_x": "Δ rank (volume − per capita)",
        "d3b_hover": (
            "<b>%{y}</b><br>"
            "Rang volume: %{customdata[0]:.0f}<br>"
            "Rang per capita: %{customdata[1]:.0f}<br>"
            "Δ = %{x:+.0f}<extra></extra>"
        ),
        "d3_finding": (
            "<strong>Yes, strong correlation</strong> (ρ≈0.82): wealthy countries export "
            "more per capita. <strong>Rank Movers</strong>: large poor countries "
            "(China, India) lose ranks when normalized — their volume is tied to "
            "size, not productivity. Conversely, small wealthy countries "
            "(Luxembourg, Singapore, Switzerland) gain massively in per capita rank."
        ),

        "gh_heat_title": "Correlations [E]+[S] — Export & Import variables (Tabs G–H) · * = p<0.05",
        "subtab_g": "G · Exports",
        "subtab_h": "H · Imports",
        "subtab_synth": "G+H · Summary",

        "g1_label": "G.1 · Export Level — Variables Explained by Macro Fundamentals",
        "g1_intro": (
            "G.1 covers 4 variables: "
            "World Export Share, Export/GDP (openness), Export per Capita. "
            "All respond positively to [S] and [E] indicators."
        ),
        "g11_question": "Does absolute economic size determine global export share?",
        "g11a_title": "Export Share vs GDP (ρ[S]=+0.94)",
        "gdp_log": "GDP (US$, log)",
        "export_share_pct": "Export Share (%)",
        "g11b_title": "Export per Capita vs GDP/capita (ρ[E]≈+0.82)",
        "g11_finding": (
            "<strong>Size determinism [S]</strong>: GDP (US$) ρ=+0.94 — "
            "absolute economic size quasi-mechanically explains global share. "
            "Secondary [E] signal: CA% GDP ρ=+0.48 — surplus countries export "
            "more than their size implies. "
            "Hard threshold at ~$259B: below it, export share is near zero."
        ),

        "g12_question": (
            "Does a current account surplus boost export share? "
            "Who over/under-performs relative to their size?"
        ),
        "g12a_title": "Export Share vs Current Account % GDP (ρ[E]=+0.48)",
        "ca_pct_gdp": "CA% GDP",
        "trade_eff_hover": "<b>%{y}</b><br>Efficience = %{x:.2f}×<extra></extra>",
        "trade_eff_ref_label": "Reference (1×)",
        "trade_eff_title": "Trade Efficiency = Export Share / GDP Share",
        "trade_eff_x": "Export Share / GDP Share (×)",
        "g12_finding": (
            "<strong>Trade Efficiency Ratio</strong> = Export Share / GDP Share. "
            "> 1 = over-performer (exports more than its economic weight). "
            "<strong>Germany ~2.6×</strong>: export-industrial model. "
            "<strong>China ~2.4×</strong>: massive global integration. "
            "<strong>USA ~0.65×</strong>: under-performer — dominant domestic economy. "
            "CA% [E] (ρ=+0.48): surplus countries structurally over-perform."
        ),

        "kw_size_question": "Do size profiles group countries with structurally different macro fundamentals?",
        "kw1_title": "GDP per Capita by size profile (KW η² large)",
        "gdp_pc_usd": "GDP per Capita (US$)",
        "kw1_hover": "<b>%{text}</b><br>GDP/capita=%{customdata[0]}<extra></extra>",
        "kw2_title": "Current Account % GDP by size profile",
        "kw_size_finding": (
            "<strong>Size profiles strongly discriminate macro fundamentals</strong>: "
            "Median GDP_pc for Leaders (~$45k) is 4–5× that of Small (~$8k). "
            "CA%: Leaders have a slightly negative median (large domestic market → "
            "structural deficit), Mid are the most in surplus (emerging exporting economies). "
            "The boxplot reveals intra-group variance: "
            "even Small have outliers at $50k+ (Luxembourg, Singapore)."
        ),

        "g16_question": "Does a high real interest rate penalize global export share?",
        "g16a_title": "Export Share vs Real Interest Rate % (ρ[E]=-0.36)",
        "real_rate_pct": "Real Interest Rate (%)",
        "g16b_title": "Export/GDP vs Real Interest Rate % (ρ[E]=-0.18)",
        "export_gdp_ratio": "Export / GDP",
        "g16_finding": (
            "<strong>High real rate → compressed exports</strong>: ρ=-0.36 [E] for "
            "Export Share, ρ=-0.18 for Export/GDP — causal economic signal. "
            "Mechanism: high real rates raise the cost of export credit and domestic "
            "production, reducing external competitiveness. "
            "Moderate signal because GDP size dominates (ρ[S]=0.94) — the Real Rate "
            "refines the analysis once size is controlled for."
        ),

        "other_vars_expander": "Other variables tested (G.1.3 External Debt · G.1.4 FDI)",
        "g13_note": (
            "<strong>G.1.3 · External Debt stocks (% GNI) — Null result</strong>: "
            "no significant relationship between external debt and export performance. "
            "Question tested: «&nbsp;Do more externally indebted countries export "
            "differently?&nbsp;» — Answer: no. "
            "External debt is itself dominated by size [S] and does not discriminate "
            "export efficiency once size is controlled for."
        ),
        "g14_note": (
            "<strong>G.1.4 · FDI Inflows — Tautological [S]</strong>: "
            "ρ≈+0.76 with Export Share but it is a [S] signal (tautological). "
            "Large countries attract more FDI <em>and</em> export more — "
            "same cause (size), not a causal FDI → exports relationship. "
            "Question tested: «&nbsp;Do FDI-recipient countries export more?&nbsp;» "
            "— Answer: yes, but because of size, not FDI itself."
        ),

        "g13_question": "Which countries have an export-oriented economy independent of their size?",
        "g13a_title": "Export/GDP (openness) vs CA% GDP (ρ[E]=+0.41)",
        "ca_pct_title": "Current Account % GDP",
        "g13b_title": "Export/GDP vs Expense % GDP (ρ[E]=+0.28)",
        "expense_pct_gdp": "Expense % GDP",
        "g13_finding": (
            "<strong>Three [E] signals for export openness</strong>: "
            "CA% ρ=+0.41 (causal: surplus → structural export), "
            "GDP_pc ρ=+0.30 (wealth → productive capacity), "
            "Expense% ρ=+0.28 (developed states — welfare state + industry). "
            "Large countries (USA, China, Brazil) have moderate Export/GDP — "
            "the domestic economy prevails over external trade."
        ),

        "g14_question": "Do wealthy countries with high public spending export more per capita?",
        "g14a_title": "Export per Capita vs GDP/capita (ρ[E]=+0.82)",
        "g14b_title": "Export per Capita vs Expense % GDP (ρ[E]=+0.50)",
        "g14_finding": (
            "<strong>Two dominant [E] signals for export/capita</strong>: "
            "GDP_pc ρ=+0.82 (log-log slope = 1.22: +1% income → +1.22% exports per capita); "
            "Expense% GDP ρ=+0.50: developed-state signal — "
            "Germany, France, Nordics export massively per capita. "
            "CA% ρ=+0.55, Real Rate ρ=-0.38 complete the hierarchy."
        ),

        "g25_label": "G.2–G.5 · Export Dynamics — Idiosyncratic",
        "g25_intro": (
            "Growth, consistency, stability "
            "and resilience of exports do not respond to macro indicators. "
            "This null result is <strong>analytically central</strong>: "
            "the level is predictable, the trajectory is not."
        ),
        "g2_question": "Does the GDP growth rate (CAGR GDP) explain export growth?",
        "g2_chart_title": "CAGR Export vs CAGR GDP (ρ={rho:+.2f}) — no significant [E] indicator",
        "g2_finding": (
            "<strong>Analytically significant null result</strong>: "
            "CAGR GDP ρ≈{rho:+.2f}. "
            "All [E] indicators are non-significant (p>0.05). "
            "KW non-significant for all size profiles (H=3.0, p=0.39). "
            "<strong>Export growth is idiosyncratic</strong> — "
            "what a country does with its trade position depends on factors "
            "not captured by the 16 macro indicators."
        ),
        "g3_note": (
            "<strong>G.3 · Export Consistency (negfreq) — Null result</strong>: "
            "Question tested: «&nbsp;Do countries with better macro fundamentals have "
            "fewer negative years?&nbsp;» — Answer: no. "
            "RF R²=0.854 (size-driven) vs OLS R²=0.105 — massive gap confirming that "
            "the relationship is non-linear and driven by size, not macro [E]. "
            "No significant [E] indicator. The frequency of declines is idiosyncratic."
        ),

        "g4_question": (
            "Do wealthy countries export with greater stability? "
            "Is export resilience macro-explained?"
        ),
        "g4a_title": "Export Volatility by size profile<br>"
                     "<span style='font-size:11px;color:#6890B0;'>"
                     "Leaders = stable, Small = high variance</span>",
        "export_volatility_pct": "Export Volatility (%)",
        "g4b_title": "Export Volatility vs Real Interest Rate",
        "g4_finding": (
            "<strong>Weak but consistent signal</strong>: Real Rate ρ=+0.20 — "
            "high real rates → more volatile exports (export financing cost). "
            "GDP_pc ρ=-0.15 (p<0.05) — wealthy countries export more regularly. "
            "The boxplot confirms: Leaders clustered below 20%, Small dispersed 0–80%. "
            "The size effect protects export regularity, but modestly."
        ),
        "g5_note": (
            "<strong>G.5 · Export Resilience — Null result [E]</strong>: "
            "Question tested: «&nbsp;Do macro fundamentals protect export "
            "resilience against shocks?&nbsp;» — Answer: no. "
            "Only signal: GDP (US$) ρ=+0.20 [S] (size) — large countries "
            "hold up slightly better, but due to their absolute size. "
            "OLS R²=0.029 — near zero. Export resilience is idiosyncratic "
            "(trade policies, product/market diversification)."
        ),

        "g7_label": "G.7 · Two Distinct Worlds",
        "g7_finding1": (
            "<strong>World 1 — Level explained</strong>: "
            "Export Share ρ[S]=0.94, Export/Capita ρ[E]=0.82, "
            "Export/GDP ρ[E]=0.41. "
            "GDP size, CA%, GDP_pc and Expense% determine <em>how much</em> a country exports."
        ),
        "g7_finding2": (
            "<strong>World 2 — Dynamics unexplained</strong>: "
            "CAGR Export (all [E] n.s.), Neg.Freq (RF R²=0.85 but OLS=0.11 → "
            "the relationship is non-linear and size-driven), Resilience (ρ_max=0.20). "
            "What a country <em>does</em> with its starting position is idiosyncratic."
        ),

        "h13_label": "H.1–H.3 · Import Level — Mirror of G, even stronger relationship",
        "h13_intro": (
            "Imports respond even "
            "better to macro fundamentals than exports: Import/capita presents "
            "4 strong simultaneous [E] signals (vs 2–3 for Export/capita)."
        ),
        "h1_question": "Does economic size explain imports even better than exports?",
        "h1a_title": "Import Share vs GDP (ρ=+0.96 > Export ρ=0.94)",
        "import_share_pct": "Import Share (%)",
        "h1b_title": "Import Share vs Export Share — symmetry",
        "equilibrium_label": "Equilibrium (1:1)",
        "h1_finding": (
            "<strong>Near-mechanical relationship</strong>: ρ=+0.96 vs Export ρ=+0.94. "
            "USA = largest importer (10.7%) despite export efficiency < 1 — "
            "the domestic market absorbs the whole world. "
            "[E] signal: GDP_pc ρ=+0.45, CA% ρ=+0.42 — wealthy "
            "and surplus countries also import proportionally more."
        ),

        "h3_question": (
            "Which [E] indicators best explain imports per capita? "
            "(Best-explained variable in H)"
        ),
        "h3a_title": "Import per Capita vs GDP/capita (ρ=+0.925)",
        "import_pc_log": "Import per Capita (US$, log)",
        "h3b_title": "Import per Capita × [E] (4 strong signals)",
        "h3_finding": (
            "<strong>Import/capita: best-explained variable in all of G+H</strong>. "
            "4 [E] signals simultaneously significant and strong: "
            "GDP_pc ρ=+0.925 (wealth = purchasing power = imports), "
            "Expense% ρ=+0.57 (public spending → collective consumption), "
            "Inflation ρ=-0.47 (inflation compresses imports by eroding purchasing power — "
            "stronger signal here than for export/capita ρ=-0.32), "
            "Real Rate ρ=-0.36 (expensive credit → fewer imports)."
        ),

        "h2_question": "Is import openness linked to wealth and public spending?",
        "h2a_title": "Import/GDP vs GDP/capita (ρ[E]=+0.22)",
        "import_gdp_ratio": "Import / GDP",
        "h2b_title": "Import/GDP vs Expense % GDP (ρ[E]=+0.36)",
        "h2_finding": (
            "<strong>Expense% GDP: stronger signal for import openness than export</strong> "
            "(ρ=+0.36 vs +0.28). "
            "The developed state generates import demand through public consumption. "
            "USA and Japan: below trend — Import/GDP low despite wealth and spending "
            "(domestic production substituting imports). "
            "Inflation ρ=-0.16 (borderline, p=0.051) — trend but not confirmed."
        ),

        "h47_label": "H.4–H.7 · Import Dynamics — Mainly Idiosyncratic",
        "h4_question": (
            "Does GDP better predict import growth than export growth? "
            "(Direct CAGR comparison)"
        ),
        "h4_chart_title": "CAGR Import vs CAGR GDP (ρ={rho:+.2f})",
        "cagr_import_pct": "CAGR Import (%)",
        "h4x_chart_title": "CAGR Export vs CAGR GDP (ρ={rho:+.2f}) — G.2 reference",
        "h4_finding": (
            "<strong>Imports follow the GDP cycle better than exports</strong>: "
            "CAGR Import ρ≈{rho_m:+.2f} vs CAGR Export ρ≈{rho_x:+.2f}. "
            "Interpretation: when the economy grows, domestic demand "
            "pulls imports mechanically. Export supply depends on "
            "external competitiveness — more idiosyncratic. "
            "<strong>China</strong>: negative CAGR Import despite positive GDP — "
            "massive domestic substitution."
        ),

        "h5_question": (
            "Do large countries have more frequent import declines "
            "than small ones? (Counter-intuitive result)"
        ),
        "h5a_title": "Import decline frequency by size profile<br>"
                     "<span style='font-size:11px;color:#F59E0B;'>"
                     "Paradox: Leaders have more declines than Small</span>",
        "negfreq_import_y": "Frequency of negative years (ratio)",
        "h5b_title": "Import decline frequency by structural profile",
        "h5_finding": (
            "<strong>Paradox H.5</strong>: GDP (US$) ρ=+0.21 — "
            "<em>large</em> countries more frequently have import decline years, "
            "contrary to what would be expected. "
            "Explanation: large economies (USA, China, EU) absorb global shocks "
            "(financial crises, COVID) more cyclically — "
            "their large import dependencies become fragile during recessions. "
            "Small countries have more stable imports because more constrained "
            "(no cyclical boom/bust in domestic demand). "
            "No significant [E] signal — the pattern is size-driven [S]."
        ),

        "h6_question": "Does inflation make imports more volatile? (Signal absent for exports)",
        "h6a_title": "Import Volatility vs Inflation % (ρ[E]=+0.20)",
        "inflation_cpi": "Inflation % (CPI)",
        "import_volatility_pct": "Import Volatility (%)",
        "h6b_title": "Import Volatility vs GDP/capita (ρ[E]=-0.24)",
        "h6_finding": (
            "<strong>Inflation signal — specific to imports (H.6)</strong>: "
            "Inflation ρ=+0.20 (p<0.01) for import volatility, "
            "absent for export volatility (ρ=+0.08 n.s.). "
            "Inflation creates uncertainty in purchasing decisions → "
            "erratic import behavior. "
            "GDP_pc ρ=-0.24: wealthy countries import more regularly "
            "(financial buffer against supply shocks)."
        ),
        "h7_note": (
            "<strong>H.7 · Import Resilience — Absolute null result</strong>: "
            "Question tested: «&nbsp;Do macro fundamentals protect import "
            "resilience against shocks?&nbsp;» — Answer: no, "
            "and paradoxically GDP (US$) ρ=-0.10 (n.s., slightly negative) — "
            "large countries are <em>marginally less resilient</em> "
            "(mirror of H.5 paradox: large import dependencies = more exposed). "
            "OLS R²=0.013 — near zero. This is the most null result in all of G+H. "
            "Import resilience is entirely idiosyncratic."
        ),

        "h8_label": "H.8 · Imports vs Exports — What differs structurally",
        "h8_finding1": (
            "<strong>Level: imports even better explained than exports</strong>: "
            "Import Share ρ=0.96 > Export Share ρ=0.94. "
            "Import/capita: 4 strong [E] signals (GDP_pc 0.925, Expense% 0.57, "
            "Inflation -0.47, Real Rate -0.36) vs Export/capita: 2–3 signals. "
            "Importing is more mechanically linked to macro fundamentals."
        ),
        "h8_finding2": (
            "<strong>Dynamics: imports slightly more predictable</strong>: "
            "CAGR Import ρ≈{rho_m:+.2f} vs CAGR Export ρ≈{rho_x:+.2f} (CAGR GDP). "
            "Inflation ρ=+0.20 for import volatility — signal absent for exports. "
            "Imports follow domestic demand, exports follow "
            "competitiveness — more idiosyncratic by nature."
        ),
        "h8_finding3": (
            "<strong>Country anomalies</strong>: "
            "USA = over-importer (Import Share 10.7%, Export Efficiency ~0.65×); "
            "Germany = over-exporter (Efficiency ~2.6×); "
            "China = import paradox (negative CAGR despite positive GDP = substitution). "
            "These patterns confirm that macro describes structure, not trajectory."
        ),

        "gh_synth_label": "G+H · Summary — Level vs Dynamics",
        "gh_synth_table_html": """
        <div style="font-size:13px;color:#A8C0D8;line-height:1.9;margin-bottom:1.2rem;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Variable</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Top signal</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">ρ (top)</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Flag</th>
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Conclusion</th>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export Share</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.94</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Deterministic size</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import Share</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.96</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Even more mechanical</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/capita</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP_pc</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.82</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Wealth + Expense% (0.50)</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/capita</td>
            <td style="padding:.35rem .7rem;text-align:center;">GDP_pc</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0.925</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#60A5FA;font-weight:600;">
              4 strong [E] signals</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;">CA% GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0.41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Moderate — surplus → openness</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;">Expense%</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0.36</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Stronger here than Export/GDP</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Export</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Idiosyncratic</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Import</td>
            <td style="padding:.35rem .7rem;text-align:center;">CAGR GDP</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">~+0.41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[T]</span></td>
            <td style="padding:.35rem .7rem;">GDP cycle → more predictable</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Export Volatility</td>
            <td style="padding:.35rem .7rem;text-align:center;">Real Rate</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0.20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Weak</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Import Volatility</td>
            <td style="padding:.35rem .7rem;text-align:center;">Inflation</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0.20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#A78BFA;">Signal unique to H</td>
          </tr>
          <tr>
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Resilience X/M</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Null — idiosyncratic</td>
          </tr>
        </table>
        <div style="margin-top:.7rem;font-size:11px;color:#4A6890;">
          [S] = Size-driven · [E] = Economic (wealth/structure) ·
          [T] = Tautological · n.s. = non-significant
        </div>
        </div>
        """,
        "gh_key_message": "Key Message",
        "gh_key_finding1": (
            "<strong>Level = predictable by macro</strong>: "
            "size [S] explains who dominates world trade (ρ>0.94). "
            "Wealth and structure [E] (GDP_pc, CA%, Expense%, Real Rate) "
            "explain efficiency and openness (ρ=0.28–0.925). "
            "These relationships are <em>descriptive and structural</em>."
        ),
        "gh_key_finding2": (
            "<strong>Dynamics = idiosyncratic</strong>: "
            "growth, consistency and resilience of exports do not respond "
            "to the 16 macro indicators. "
            "Partial exception: imports follow the GDP cycle slightly "
            "(CAGR Import ρ≈0.41) because domestic demand mechanically pulls imports. "
            "What a country <em>does</em> with its starting position remains unpredictable."
        ),

        "ij_heat_title": "Correlations [E] — Balance & Openness variables (Tab I) · * = p<0.05",
        "subtab_i": "I · Balance",
        "subtab_j": "J · Quadrants & Profiles",

        "i_label": "I · Trade Balance × Macro — CA% as the Central Signal",
        "i_intro": (
            "Section I links balance variables "
            "(coverage ratio, balance/GDP, balance profile) to macro indicators. "
            "One [E] signal dominates consistently across three different angles."
        ),
        "i1_question": "Do surplus countries (CA% > 0) better cover their imports with their exports?",
        "i1a_title": "Coverage Ratio (X/M) vs CA% GDP (ρ[E]=+0.53)",
        "coverage_ratio_title": "Coverage Ratio (X/M)",
        "xm_equal_label": "X = M",
        "i1b_title": "Coverage Ratio × [E]",
        "i1_finding": (
            "<strong>CA% [E] ρ=+0.53 — cleanest causal signal in I</strong>: "
            "a current surplus implies that exports structurally cover "
            "imports (CA% includes income + transfers, so the relationship is not "
            "circular). Strong/Good profiles concentrate above the X=M line "
            "and to the right of CA%=0."
        ),

        "i3_question": "Does the trade balance (X−M)/GDP directly reflect the current account?",
        "i3a_title": "Balance/GDP vs CA% GDP (ρ[E]=+0.48)",
        "balance_gdp_title": "(X−M) / GDP",
        "i3b_title": "Balance/GDP vs GDP/capita (ρ={rho:+.3f})",
        "i3_finding": (
            "<strong>CA% ρ=+0.48 — second independent confirmation</strong>: "
            "three different angles (coverage ratio I.1, balance/GDP I.3, profile I.5) "
            "converge on the same signal. The relationship is not circular: "
            "CA% includes investment income and current transfers, "
            "not only the goods/services balance."
        ),

        "i5_question": "Which [E] indicators best discriminate trade balance profiles?",
        "i5a_title": "CA% GDP by balance profile (KW η²=0.241)",
        "i5b_title": "GDP per Capita by balance profile (moderate η²)",
        "i5_finding": (
            "<strong>Balance Profile × KW — strong discrimination</strong>: "
            "CA% GDP η²=0.241 (large) — the strongest signal in all of section I. "
            "Medians: Strong Surplus ~+2%, Slight Surplus +1.7%, "
            "Slight Deficit -1.1%, Structural Deficit -4.8%. "
            "Monotone and robust progression (Spearman and KW consistent). "
            "GDP_pc discriminates moderately: Structural Deficit = poor countries by median, "
            "but high variance (wealthy countries also have structural deficits)."
        ),

        "i_other_expander": "Sections tested — results referenced in Tab D",
        "i2_note": (
            "<strong>I.2 · Trade Intensity × Expense% (ρ=+0.34)</strong>: "
            "analyzed in Tab D · D.1 Trade Intensity. "
            "Result: Expense% [E] ρ=+0.34 — developed states have a "
            "more trade-open economy. See D.1 for the full analysis."
        ),
        "i4_note": (
            "<strong>I.4 · Trade Elasticity × Macro</strong>: "
            "analyzed in Tab D · D.2. "
            "Result: macro does not predict trade/GDP elasticity. "
            "The situation (Export-led, GDP-led, etc.) is idiosyncratic. "
            "See D.2 for the 6-situation classification."
        ),

        "i_synth_label": "I · Summary — CA% as Central Signal",
        "i_synth_finding": (
            "<strong>CA% GDP is the most consistent [E] signal in section I</strong>: "
            "three independent analyses converge: "
            "I.1 Coverage Ratio ρ=+0.53, I.3 Balance/GDP ρ=+0.48, "
            "I.5 Balance Profile η²=0.241. "
            "Interpretation: the current account captures external competitive position "
            "more completely than the trade balance alone."
        ),

        "j_label": "J · Trade Quadrants & Profiles × Macro",
        "j_intro": (
            "Section J tests whether the different "
            "trade classification systems (structural, static, dynamic, strategic) "
            "correspond to distinct macro fundamentals. "
            "General rule: <strong>static = explained</strong>, "
            "<strong>dynamic = idiosyncratic</strong>."
        ),
        "j1_question": "Do structural profiles (Strong/Good/Moderate/Weak) correspond to different macro fundamentals?",
        "j1a_title": "GDP/capita by structural profile (η²[E]=0.137)",
        "j1b_title": "CA% GDP by structural profile (η²[E]=0.099)",
        "j1_finding": (
            "<strong>Structural profile partially macro-explained</strong>: "
            "η² GDP_pc=0.137 (large), CA%=0.099 (medium). "
            "Important nuance: the structural_score is 60% dominated by z_export_share — "
            "a large country can reach 'Strong' by size alone, "
            "independently of its dynamic performance. "
            "<strong>Germany contradiction</strong>: highest structural score (+4.6) "
            "but classified Declining in J.3 — size masks dynamic deterioration."
        ),

        "j2_question": "Is the static quadrant (Dominant/Balanced/Vulnerable/Marginal) best discriminated by CA%?",
        "j2a_title": "CA% GDP by static quadrant (η²[E]=0.250 — max in J)",
        "j2b_title": "GDP/capita by static quadrant (η²[E]=0.128)",
        "j2_finding": (
            "<strong>Strongest signal in all of section J</strong>: "
            "CA% η²=0.250 (KW H=47.0, p<0.0001). "
            "Full consistency with I.1 (ρ=0.53), I.3 (ρ=0.48), I.5 (η²=0.241): "
            "four independent analyses confirm that CA% is "
            "the macro determinant of static positioning. "
            "Dominant = surplus countries by median, "
            "Vulnerable = deficit countries despite large size."
        ),

        "j3_question": "Is the dynamic quadrant (Star/Risky Grower/Stable/Declining) predictable from macro fundamentals?",
        "j3_note": (
            "<strong>J.3 · Dynamic quadrant — Null result [E]</strong>: "
            "no macro indicator significantly discriminates between "
            "Star, Risky Grower, Stable Stagnant and Declining. "
            "This is the strongest confirmation of G.2/G.3/G.5 results: "
            "the dynamic quadrant synthesizes export growth and stability — "
            "two dimensions both idiosyncratic. "
            "<strong>Germany nuance</strong>: classified Declining (z_cagr < median) "
            "but in Stars in J.4 (absolute cagr > 0%) — the same performance can be "
            "Declining in relative and Growing in absolute terms depending on the threshold."
        ),
        "j4_note": (
            "<strong>J.4 · Strategic quadrant (Stars/Cash Cow/Emerging/Marginal)</strong>: "
            "GDP_pc η²=0.137, CA% η²=0.096 — signal present but moderate. "
            "Stars membership ≠ structural quality: several Stars show "
            "a low coverage ratio (I.1) and a CA deficit (I.3). "
            "The BCG framework captures size × growth but ignores the balance dimension — "
            "complementary to static quadrant J.2, not substitutable."
        ),

        "j_synth_label": "J · Summary — Static Explained, Dynamic Idiosyncratic",
        "j_synth_finding1": (
            "<strong>Static positioning = macro-explained</strong>: "
            "J.1 (structural profile η²=0.137) and J.2 (static quadrant η²=0.250) "
            "confirm that CA% and GDP_pc determine where a country positions itself "
            "in world trade in terms of size and balance."
        ),
        "j_synth_finding2": (
            "<strong>Dynamics = idiosyncratic</strong>: "
            "J.3 (dynamic quadrant) = zero macro signal. "
            "Consistent with G.2 (CAGR Export n.s.), G.3 (neg.freq n.s.), G.5 (resilience n.s.). "
            "Same conclusion from three different levels of analysis."
        ),
        "j_synth_finding3": (
            "<strong>G/H/I enrichment → CA% convergence</strong>: "
            "I.1 ρ=+0.53, I.3 ρ=+0.48, I.5 η²=0.241, J.2 η²=0.250 — "
            "four independent analyses in two different sections "
            "converge on the same indicator. "
            "CA% GDP is the most robust causal [E] signal across all of G→J."
        ),

        "k_label": "K · Composite Scores × Macro",
        "k_intro": (
            "Section K analyzes composite scores "
            "(structural, bilateral resilience, combined): are they macro-explained? "
            "Key result: two dimensions with two distinct analytical logics."
        ),
        "k_heat_title": "Correlations [E] — Composite Scores (Tab K) · * = p<0.05",
        "k1_question": "Is bilateral resilience (geographic export diversification) linked to macro fundamentals?",
        "k1_note": (
            "<strong>K.1 · g_norm_rank (bilateral resilience) — Null result</strong>: "
            "no macro indicator predicts the geographic concentration of exports. "
            "Hypothesis tested: better CA% → better partner diversification. "
            "Result: all ρ<0.18, p>0.05. "
            "Export diversification is the product of trade policies, "
            "bilateral agreements and geography — not aggregated macro fundamentals."
        ),

        "k3_question": "Does the combined profile (Champion/Competitive/Vulnerable/Fragile) discriminate macro [E] fundamentals?",
        "k3a_title": "GDP/capita by combined profile (KW H=19.3, p=0.0002)",
        "k3b_title": "CA% GDP by combined profile",
        "k3_finding": (
            "<strong>Global Champion clearly distinct</strong>: "
            "Median GDP_pc for Global Champion ~$34k vs Competitive/Vulnerable/Fragile at $5–8k — "
            "4–7× gap. The combined profile identifies a real economic elite. "
            "CA% discriminates moderately: Champions tend toward surplus, "
            "Fragile toward deficit, but intra-group variance is high. "
            "The combined profile is a reliable signal of development level."
        ),

        "k1k2_question": "Are the continuous scores (combined_score, structural_score) correlated with [E] indicators?",
        "k1_chart_title": "Combined Score vs GDP/capita (ρ[E]=+0.33)",
        "combined_score_title": "Combined Score",
        "structural_score_title": "Structural Score",
        "k1_finding": (
            "<strong>Combined Score ρ[E]=+0.33 (GDP_pc)</strong>: moderate signal. "
            "Adding the bilateral resilience score (g_norm_rank, not explained by macro) "
            "slightly dilutes the structural_score signal (ρ=+0.32). "
            "The Δρ≈0 difference confirms that bilateral resilience neither adds nor "
            "removes macro signal — the two dimensions are orthogonal."
        ),

        "k_hier_label": "K · [E] Signal Hierarchy — Final Summary",
        "k_hier_intro": (
            "Ranking of [E] indicators by average |ρ| across all 44 trade variables "
            "— recap of all sections G to K."
        ),
        "k_hier_chart_title": "[E] Indicators — avg |ρ| across all trade variables (G–K)",
        "k_hier_finding": (
            "<strong>G→K hierarchy confirmed</strong>: "
            "GDP_pc > CA% > Real Rate > Expense% > Inflation > Unemployment. "
            "GDP_pc: dominant [E] signal (wealth → all aspects of trade position). "
            "CA%: second signal, consistent across 4+ independent analyses (I.1/I.3/I.5/J.2). "
            "Real Rate and Expense%: secondary signals, relevant for openness (X/GDP, M/GDP). "
            "Inflation and Unemployment: weak signals, mainly for import volatility."
        ),

        "k5_label": "K.5 · Two dimensions, two logics",
        "k5_finding1": (
            "<strong>F (structural score) = explained</strong>: "
            "GDP_pc ρ=+0.32, CA% ρ=+0.30 — consistent across all of G/H/I/J/K. "
            "The structural score measures position in world trade — "
            "determined by wealth and external balance."
        ),
        "k5_finding2": (
            "<strong>G (bilateral resilience = g_norm_rank) = unexplained</strong>: "
            "all indicators n.s. The geographic concentration of trade partners "
            "is entirely idiosyncratic — geography, history, "
            "trade agreements, national policies."
        ),
        "k5_finding3": (
            "<strong>Implication for the combined_score</strong>: "
            "by weighting 70% structural + 30% bilateral, the combined_score inherits "
            "the macro signal from the structural and the idiosyncracy from the bilateral. "
            "This is analytically correct: the two dimensions capture different realities."
        ),

        "k_general_expander": "General conclusion G→K",
        "k_general_finding1": (
            "<strong>Trade structure = macro-explained (level)</strong>: "
            "Export/Import Share [S], Export/Import per Capita [E], "
            "Coverage Ratio [E], static profiles [E]. "
            "Size prevails (GDP US$ ρ>0.94), wealth refines (GDP_pc ρ=0.33–0.92)."
        ),
        "k_general_finding2": (
            "<strong>Trade dynamics = idiosyncratic</strong>: "
            "CAGR, volatility, resilience, consistency, dynamic quadrant — "
            "all non-significant [E]. "
            "What a country does with its structural position depends on factors "
            "not captured by 16 macro indicators over 10 years."
        ),
        "k_general_finding3": (
            "<strong>Size vs Wealth</strong>: "
            "being large (GDP US$) prevails for world volume (ρ=0.94–0.96). "
            "Being wealthy (GDP_pc) prevails for per capita efficiency (ρ=0.82–0.925). "
            "CA% is the most robust causal signal: surplus → external competitiveness."
        ),
    },
    "fr": {
        "eyebrow": "Lien · Commerce × Indicateurs · Page 1 sur 2",
        "title": "Signaux & liens",
        "caption": (
            "Pour chaque dimension commerciale, quel indicateur macro l'explique le mieux ? "
            "Sections D · G · H · I · J · K — 16 indicateurs × 44 variables commerciales"
        ),
        "kpi_countries": "Pays analysés",
        "kpi_sig_e": "Paires [E] signif.",
        "kpi_max_signal": "Signal max (tous)",
        "kpi_top_signal_e": "Signal n°1 [E]",
        "ef_label": "E+F · Classification des indicateurs — Base analytique",
        "ef_intro": (
            "Avant d'analyser chaque dimension commerciale (G–K), les sections E et F du notebook "
            "ont classé les 16 indicateurs macro en trois familles selon leur nature analytique. "
            "Cette classification est utilisée tout au long de la page pour distinguer "
            "les signaux causaux des artefacts statistiques."
        ),
        "flag_t_title": "[T] Tautologique",
        "flag_t_body": (
            " — l'indicateur est directement dérivé "
            "des flux commerciaux (ex. Exports USD). "
            "La forte corrélation est structurelle, pas causale."
        ),
        "flag_s_title": "[S] Lié à la taille",
        "flag_s_body": (
            " — l'indicateur reflète la taille "
            "du pays (PIB, population). La corrélation avec les volumes commerciaux est un effet "
            "d'échelle, pas un signal économique."
        ),
        "flag_e_title": "[E] Économique",
        "flag_e_body": (
            " — véritable signal macro "
            "indépendant de la définition du commerce (ex. PIB/habitant, inflation). "
            "Les corrélations ici sont analytiquement significatives."
        ),
        "ef_footer": (
            "<strong style='color:#6890B0;'>Section E</strong> · 704 paires de Spearman "
            "(44 variables × 16 indicateurs). "
            "<strong style='color:#6890B0;'>Section F</strong> · L'importance des variables (RF) + "
            "le η² de KW confirment le classement [E] vs [S]. "
            "Les heatmaps de corrélation sont disponibles au début de chaque onglet ci-dessous "
            "— elles montrent quels indicateurs [E] (et [S]) expliquent chaque famille de variables."
        ),
        "tab_d": "D · Passerelle Commerce/PIB",
        "tab_gh": "G–H · Exports & Imports",
        "tab_ij": "I–J · Balance & Profils",
        "tab_k": "K · Scores",
        "question_label": "Question",
        "note_label": "Note analytique",

        "d_intro": (
            "Trois métriques dérivées qui relient directement la structure commerciale et le macro : "
            "Intensité commerciale = (X+M)/PIB · Élasticité = CAGR_export/CAGR_PIB · "
            "Commerce par habitant = volume normalisé par la population."
        ),
        "d_heat_title": "Corrélations [E] — Variables Commerce/PIB (Onglet D) · * = p<0,05",
        "d1_label": "D.1 · Intensité commerciale",
        "d1_question": "La richesse d'un pays déterminine-t-elle son ouverture commerciale ?",
        "d1_chart_title": "Intensité commerciale (X+M)/PIB vs PIB/habitant",
        "gdp_pc_log": "PIB par habitant (US$, log)",
        "trade_intensity_title": "Intensité commerciale (X+M)/PIB",
        "d1_finding": (
            "<strong>Oui, mais de façon non linéaire</strong> : les pays plus riches tendent vers une "
            "intensité commerciale plus élevée, avec une dispersion croissante. Les grands pays "
            "(Leaders) ont des intensités variables selon leur structure domestique — "
            "les USA et la Chine ont une intensité modérée malgré leur taille, "
            "tandis que les petits pays riches (Singapour, Luxembourg) sont les plus ouverts."
        ),

        "d2_label": "D.2 · Élasticité Commerce/PIB — 6 situations",
        "d2_question": "Le commerce extérieur croît-il plus vite que l'économie ? Dans quels profils ?",
        "d2_chart_title": "CAGR Export vs CAGR PIB — diagonale = élasticité 1",
        "cagr_gdp_pct": "CAGR PIB (%)",
        "cagr_export_pct": "CAGR Export (%)",
        "d2b_chart_title": "Répartition des pays<br>par situation (2015–2024)",
        "d2b_x": "Nombre de pays",
        "d2_finding": (
            "<strong>Majoritairement tiré par le PIB</strong> : la plupart des pays ont une "
            "croissance des exports plus lente que leur PIB (élasticité < 1). Les pays tirés par "
            "l'export sont souvent des économies émergentes en phase d'intégration mondiale. "
            "<strong>Export Resilient</strong> : des pays dont les exports croissent même quand "
            "l'économie se contracte — signe d'une compétitivité externe robuste."
        ),
        "d2_expander_title": "Définition des 6 situations",
        "d2_table_html": """
        <div style="font-size:13px;color:#A8C0D8;line-height:2.0;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Situation</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Condition</th>
            <th style="text-align:left;padding:.4rem .8rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Interprétation</th>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#10B981;font-weight:600;">Export-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; CAGR_PIB &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              Les exports croissent plus vite que l'économie — moteur externe dominant</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#60A5FA;font-weight:600;">GDP-led</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_PIB &gt; CAGR_X &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              L'économie croît plus vite que les exports — croissance tirée par la demande intérieure</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#A78BFA;font-weight:600;">Export Resilient</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &gt; 0, CAGR_PIB ≤ 0</td>
            <td style="padding:.4rem .8rem;">
              Les exports croissent alors que l'économie se contracte — compétitivité externe robuste</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#F59E0B;font-weight:600;">Export Contraction</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X ≤ 0, CAGR_PIB &gt; 0</td>
            <td style="padding:.4rem .8rem;">
              L'économie croît mais les exports chutent — substitution domestique ou perte de compétitivité</td>
          </tr>
          <tr style="border-bottom:1px solid #1F3050;">
            <td style="padding:.4rem .8rem;">
              <span style="color:#FB923C;font-weight:600;">Moderate Decline</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_PIB &lt; 0, |X| &lt; |PIB|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, les exports résistent mieux que le PIB</td>
          </tr>
          <tr>
            <td style="padding:.4rem .8rem;">
              <span style="color:#EF4444;font-weight:600;">Severe Decline</span></td>
            <td style="padding:.4rem .8rem;font-family:monospace;font-size:12px;">
              CAGR_X &lt; 0, CAGR_PIB &lt; 0, |X| &gt; |PIB|</td>
            <td style="padding:.4rem .8rem;">
              Double contraction, les exports s'effondrent plus vite que le PIB</td>
          </tr>
        </table>
        <div style="margin-top:.8rem;font-size:12px;color:#4A6890;">
          CAGR_X = taux de croissance annuel composé des exports (2015–2024) ·
          CAGR_PIB = même calcul pour le PIB nominal ·
          Élasticité = CAGR_X / CAGR_PIB
        </div>
        </div>
        """,

        "d3_label": "D.3 · Export par habitant vs PIB/habitant",
        "d3_question": (
            "Les pays riches exportent-ils plus par habitant ? "
            "Quels pays changent de position une fois normalisé par la population ?"
        ),
        "d3_chart_title": "Export par habitant vs PIB/habitant",
        "export_pc_log": "Export par habitant (US$, log)",
        "d3b_chart_title": (
            "Rank Mover — Volume vs par habitant<br>"
            "<span style='font-size:11px;color:#6890B0;'>"
            "Δ > 0 : meilleur rang par habitant que par volume</span>"
        ),
        "d3b_x": "Δ rang (volume − par habitant)",
        "d3b_hover": (
            "<b>%{y}</b><br>"
            "Rang volume : %{customdata[0]:.0f}<br>"
            "Rang par habitant : %{customdata[1]:.0f}<br>"
            "Δ = %{x:+.0f}<extra></extra>"
        ),
        "d3_finding": (
            "<strong>Oui, forte corrélation</strong> (ρ≈0,82) : les pays riches exportent "
            "plus par habitant. <strong>Rank Movers</strong> : les grands pays pauvres "
            "(Chine, Inde) perdent des rangs une fois normalisés — leur volume est lié à "
            "la taille, pas à la productivité. À l'inverse, les petits pays riches "
            "(Luxembourg, Singapour, Suisse) gagnent massivement en rang par habitant."
        ),

        "gh_heat_title": "Corrélations [E]+[S] — Variables Export & Import (Onglets G–H) · * = p<0,05",
        "subtab_g": "G · Exports",
        "subtab_h": "H · Imports",
        "subtab_synth": "G+H · Synthèse",

        "g1_label": "G.1 · Niveau d'export — Variables expliquées par les fondamentaux macro",
        "g1_intro": (
            "G.1 couvre 4 variables : "
            "part mondiale des exports, Export/PIB (ouverture), Export par habitant. "
            "Toutes répondent positivement aux indicateurs [S] et [E]."
        ),
        "g11_question": "La taille économique absolue déterminine-t-elle la part mondiale des exports ?",
        "g11a_title": "Part des exports vs PIB (ρ[S]=+0,94)",
        "gdp_log": "PIB (US$, log)",
        "export_share_pct": "Part des exports (%)",
        "g11b_title": "Export par habitant vs PIB/habitant (ρ[E]≈+0,82)",
        "g11_finding": (
            "<strong>Déterminisme de taille [S]</strong> : PIB (US$) ρ=+0,94 — "
            "la taille économique absolue explique quasi mécaniquement la part mondiale. "
            "Signal [E] secondaire : CA% PIB ρ=+0,48 — les pays excédentaires exportent "
            "plus que leur taille ne le laisse supposer. "
            "Seuil net à ~259 Md$ : au-dessous, la part des exports est quasi nulle."
        ),

        "g12_question": (
            "Un excédent courant augmente-t-il la part des exports ? "
            "Qui sur/sous-performe par rapport à sa taille ?"
        ),
        "g12a_title": "Part des exports vs compte courant % PIB (ρ[E]=+0,48)",
        "ca_pct_gdp": "CA% PIB",
        "trade_eff_hover": "<b>%{y}</b><br>Efficience = %{x:.2f}×<extra></extra>",
        "trade_eff_ref_label": "Référence (1×)",
        "trade_eff_title": "Efficience commerciale = Part des exports / Part du PIB",
        "trade_eff_x": "Part des exports / Part du PIB (×)",
        "g12_finding": (
            "<strong>Ratio d'efficience commerciale</strong> = Part des exports / Part du PIB. "
            "> 1 = surperformeur (exporte plus que son poids économique). "
            "<strong>Allemagne ~2,6×</strong> : modèle export-industriel. "
            "<strong>Chine ~2,4×</strong> : intégration mondiale massive. "
            "<strong>USA ~0,65×</strong> : sous-performeur — économie domestique dominante. "
            "CA% [E] (ρ=+0,48) : les pays excédentaires surperforment structurellement."
        ),

        "kw_size_question": "Les profils de taille regroupent-ils des pays aux fondamentaux macro structurellement différents ?",
        "kw1_title": "PIB par habitant par profil de taille (KW η² élevé)",
        "gdp_pc_usd": "PIB par habitant (US$)",
        "kw1_hover": "<b>%{text}</b><br>PIB/habitant=%{customdata[0]}<extra></extra>",
        "kw2_title": "Compte courant % PIB par profil de taille",
        "kw_size_finding": (
            "<strong>Les profils de taille discriminent fortement les fondamentaux macro</strong> : "
            "le PIB/hab médian des Leaders (~45k$) est 4 à 5× celui des Small (~8k$). "
            "CA% : les Leaders ont une médiane légèrement négative (grand marché intérieur → "
            "déficit structurel), les Mid sont les plus excédentaires (économies émergentes exportatrices). "
            "Le boxplot révèle la variance intra-groupe : "
            "même les Small ont des valeurs extrêmes à 50k$+ (Luxembourg, Singapour)."
        ),

        "g16_question": "Un taux d'intérêt réel élevé pénalise-t-il la part mondiale des exports ?",
        "g16a_title": "Part des exports vs taux d'intérêt réel % (ρ[E]=-0,36)",
        "real_rate_pct": "Taux d'intérêt réel (%)",
        "g16b_title": "Export/PIB vs taux d'intérêt réel % (ρ[E]=-0,18)",
        "export_gdp_ratio": "Export / PIB",
        "g16_finding": (
            "<strong>Taux réel élevé → exports compressés</strong> : ρ=-0,36 [E] pour "
            "la part des exports, ρ=-0,18 pour Export/PIB — signal économique causal. "
            "Mécanisme : des taux réels élevés augmentent le coût du crédit export et de la "
            "production domestique, réduisant la compétitivité externe. "
            "Signal modéré car la taille du PIB domine (ρ[S]=0,94) — le taux réel "
            "affine l'analyse une fois la taille contrôlée."
        ),

        "other_vars_expander": "Autres variables testées (G.1.3 Dette externe · G.1.4 IDE)",
        "g13_note": (
            "<strong>G.1.3 · Stock de dette externe (% RNB) — Résultat nul</strong> : "
            "aucune relation significative entre la dette externe et la performance export. "
            "Question testée : «&nbsp;Les pays plus endettés à l'externe exportent-ils "
            "différemment ?&nbsp;» — Réponse : non. "
            "La dette externe est elle-même dominée par la taille [S] et ne discrimine pas "
            "l'efficience export une fois la taille contrôlée."
        ),
        "g14_note": (
            "<strong>G.1.4 · Entrées d'IDE — Tautologique [S]</strong> : "
            "ρ≈+0,76 avec la part des exports, mais c'est un signal [S] (tautologique). "
            "Les grands pays attirent plus d'IDE <em>et</em> exportent plus — "
            "même cause (taille), pas une relation causale IDE → exports. "
            "Question testée : «&nbsp;Les pays récipiendaires d'IDE exportent-ils plus ?&nbsp;» "
            "— Réponse : oui, mais à cause de la taille, pas de l'IDE lui-même."
        ),

        "g13_question": "Quels pays ont une économie orientée export indépendamment de leur taille ?",
        "g13a_title": "Export/PIB (ouverture) vs CA% PIB (ρ[E]=+0,41)",
        "ca_pct_title": "Compte courant % PIB",
        "g13b_title": "Export/PIB vs dépense % PIB (ρ[E]=+0,28)",
        "expense_pct_gdp": "Dépense % PIB",
        "g13_finding": (
            "<strong>Trois signaux [E] pour l'ouverture export</strong> : "
            "CA% ρ=+0,41 (causal : excédent → export structurel), "
            "PIB/hab ρ=+0,30 (richesse → capacité productive), "
            "Dépense% ρ=+0,28 (États développés — État social + industrie). "
            "Les grands pays (USA, Chine, Brésil) ont un Export/PIB modéré — "
            "l'économie domestique prévaut sur le commerce extérieur."
        ),

        "g14_question": "Les pays riches à forte dépense publique exportent-ils plus par habitant ?",
        "g14a_title": "Export par habitant vs PIB/habitant (ρ[E]=+0,82)",
        "g14b_title": "Export par habitant vs dépense % PIB (ρ[E]=+0,50)",
        "g14_finding": (
            "<strong>Deux signaux [E] dominants pour l'export/habitant</strong> : "
            "PIB/hab ρ=+0,82 (pente log-log = 1,22 : +1% de revenu → +1,22% d'exports par habitant) ; "
            "Dépense% PIB ρ=+0,50 : signal des États développés — "
            "l'Allemagne, la France, les pays nordiques exportent massivement par habitant. "
            "CA% ρ=+0,55, taux réel ρ=-0,38 complètent la hiérarchie."
        ),

        "g25_label": "G.2–G.5 · Dynamique des exports — Idiosyncratique",
        "g25_intro": (
            "La croissance, la régularité, la stabilité "
            "et la résilience des exports ne répondent pas aux indicateurs macro. "
            "Ce résultat nul est <strong>analytiquement central</strong> : "
            "le niveau est prévisible, la trajectoire ne l'est pas."
        ),
        "g2_question": "Le taux de croissance du PIB (CAGR PIB) explique-t-il la croissance des exports ?",
        "g2_chart_title": "CAGR Export vs CAGR PIB (ρ={rho:+.2f}) — aucun indicateur [E] significatif",
        "g2_finding": (
            "<strong>Résultat nul analytiquement significatif</strong> : "
            "CAGR PIB ρ≈{rho:+.2f}. "
            "Tous les indicateurs [E] sont non significatifs (p>0,05). "
            "KW non significatif pour tous les profils de taille (H=3,0, p=0,39). "
            "<strong>La croissance des exports est idiosyncratique</strong> — "
            "ce qu'un pays fait de sa position commerciale dépend de facteurs "
            "non capturés par les 16 indicateurs macro."
        ),
        "g3_note": (
            "<strong>G.3 · Régularité des exports (negfreq) — Résultat nul</strong> : "
            "Question testée : «&nbsp;Les pays aux meilleurs fondamentaux macro ont-ils "
            "moins d'années négatives ?&nbsp;» — Réponse : non. "
            "RF R²=0,854 (piloté par la taille) vs OLS R²=0,105 — écart massif confirmant que "
            "la relation est non linéaire et pilotée par la taille, pas par le macro [E]. "
            "Aucun indicateur [E] significatif. La fréquence des baisses est idiosyncratique."
        ),

        "g4_question": (
            "Les pays riches exportent-ils avec plus de stabilité ? "
            "La résilience export est-elle expliquée par le macro ?"
        ),
        "g4a_title": "Volatilité des exports par profil de taille<br>"
                     "<span style='font-size:11px;color:#6890B0;'>"
                     "Leaders = stables, Small = forte variance</span>",
        "export_volatility_pct": "Volatilité des exports (%)",
        "g4b_title": "Volatilité des exports vs taux d'intérêt réel",
        "g4_finding": (
            "<strong>Signal faible mais consistant</strong> : taux réel ρ=+0,20 — "
            "des taux réels élevés → exports plus volatils (coût du financement export). "
            "PIB/hab ρ=-0,15 (p<0,05) — les pays riches exportent plus régulièrement. "
            "Le boxplot confirme : les Leaders sont regroupés sous 20%, les Small dispersés 0–80%. "
            "L'effet de taille protège la régularité des exports, mais modestement."
        ),
        "g5_note": (
            "<strong>G.5 · Résilience des exports — Résultat nul [E]</strong> : "
            "Question testée : «&nbsp;Les fondamentaux macro protègent-ils la résilience "
            "des exports face aux chocs ?&nbsp;» — Réponse : non. "
            "Seul signal : PIB (US$) ρ=+0,20 [S] (taille) — les grands pays "
            "résistent légèrement mieux, mais du fait de leur taille absolue. "
            "OLS R²=0,029 — proche de zéro. La résilience export est idiosyncratique "
            "(politiques commerciales, diversification produits/marchés)."
        ),

        "g7_label": "G.7 · Deux mondes distincts",
        "g7_finding1": (
            "<strong>Monde 1 — Niveau expliqué</strong> : "
            "Part des exports ρ[S]=0,94, Export/habitant ρ[E]=0,82, "
            "Export/PIB ρ[E]=0,41. "
            "La taille du PIB, le CA%, le PIB/hab et la dépense% déterminent <em>combien</em> un pays exporte."
        ),
        "g7_finding2": (
            "<strong>Monde 2 — Dynamique inexpliquée</strong> : "
            "CAGR Export (tout [E] n.s.), Neg.Freq (RF R²=0,85 mais OLS=0,11 → "
            "la relation est non linéaire et pilotée par la taille), Résilience (ρ_max=0,20). "
            "Ce qu'un pays <em>fait</em> de sa position de départ est idiosyncratique."
        ),

        "h13_label": "H.1–H.3 · Niveau d'import — Miroir de G, relation encore plus forte",
        "h13_intro": (
            "Les imports répondent encore "
            "mieux aux fondamentaux macro que les exports : Import/habitant présente "
            "4 signaux [E] forts et simultanés (contre 2–3 pour Export/habitant)."
        ),
        "h1_question": "La taille économique explique-t-elle les imports encore mieux que les exports ?",
        "h1a_title": "Part des imports vs PIB (ρ=+0,96 > Export ρ=0,94)",
        "import_share_pct": "Part des imports (%)",
        "h1b_title": "Part des imports vs part des exports — symétrie",
        "equilibrium_label": "Équilibre (1:1)",
        "h1_finding": (
            "<strong>Relation quasi mécanique</strong> : ρ=+0,96 vs Export ρ=+0,94. "
            "Les USA = plus gros importateur (10,7%) malgré une efficience export < 1 — "
            "le marché domestique absorbe le monde entier. "
            "Signal [E] : PIB/hab ρ=+0,45, CA% ρ=+0,42 — les pays riches "
            "et excédentaires importent aussi proportionnellement plus."
        ),

        "h3_question": (
            "Quels indicateurs [E] expliquent le mieux les imports par habitant ? "
            "(Variable la mieux expliquée en H)"
        ),
        "h3a_title": "Import par habitant vs PIB/habitant (ρ=+0,925)",
        "import_pc_log": "Import par habitant (US$, log)",
        "h3b_title": "Import par habitant × [E] (4 signaux forts)",
        "h3_finding": (
            "<strong>Import/habitant : variable la mieux expliquée de tout G+H</strong>. "
            "4 signaux [E] simultanément significatifs et forts : "
            "PIB/hab ρ=+0,925 (richesse = pouvoir d'achat = imports), "
            "Dépense% ρ=+0,57 (dépense publique → consommation collective), "
            "Inflation ρ=-0,47 (l'inflation compresse les imports en érodant le pouvoir d'achat — "
            "signal plus fort ici que pour l'export/habitant ρ=-0,32), "
            "Taux réel ρ=-0,36 (crédit coûteux → moins d'imports)."
        ),

        "h2_question": "L'ouverture à l'import est-elle liée à la richesse et à la dépense publique ?",
        "h2a_title": "Import/PIB vs PIB/habitant (ρ[E]=+0,22)",
        "import_gdp_ratio": "Import / PIB",
        "h2b_title": "Import/PIB vs dépense % PIB (ρ[E]=+0,36)",
        "h2_finding": (
            "<strong>Dépense% PIB : signal plus fort pour l'ouverture import que pour l'export</strong> "
            "(ρ=+0,36 vs +0,28). "
            "L'État développé génère de la demande d'import via la consommation publique. "
            "USA et Japon : sous la tendance — Import/PIB faible malgré richesse et dépense "
            "(production domestique substituant les imports). "
            "Inflation ρ=-0,16 (limite, p=0,051) — tendance mais non confirmée."
        ),

        "h47_label": "H.4–H.7 · Dynamique des imports — Majoritairement idiosyncratique",
        "h4_question": (
            "Le PIB prédit-il mieux la croissance des imports que celle des exports ? "
            "(Comparaison directe du CAGR)"
        ),
        "h4_chart_title": "CAGR Import vs CAGR PIB (ρ={rho:+.2f})",
        "cagr_import_pct": "CAGR Import (%)",
        "h4x_chart_title": "CAGR Export vs CAGR PIB (ρ={rho:+.2f}) — référence G.2",
        "h4_finding": (
            "<strong>Les imports suivent mieux le cycle du PIB que les exports</strong> : "
            "CAGR Import ρ≈{rho_m:+.2f} vs CAGR Export ρ≈{rho_x:+.2f}. "
            "Interprétation : quand l'économie croît, la demande domestique "
            "tire mécaniquement les imports. L'offre export dépend de "
            "la compétitivité externe — plus idiosyncratique. "
            "<strong>Chine</strong> : CAGR Import négatif malgré un PIB positif — "
            "substitution domestique massive."
        ),

        "h5_question": (
            "Les grands pays ont-ils des baisses d'import plus fréquentes "
            "que les petits ? (Résultat contre-intuitif)"
        ),
        "h5a_title": "Fréquence des baisses d'import par profil de taille<br>"
                     "<span style='font-size:11px;color:#F59E0B;'>"
                     "Paradoxe : les Leaders ont plus de baisses que les Small</span>",
        "negfreq_import_y": "Fréquence des années négatives (ratio)",
        "h5b_title": "Fréquence des baisses d'import par profil structurel",
        "h5_finding": (
            "<strong>Paradoxe H.5</strong> : PIB (US$) ρ=+0,21 — "
            "les <em>grands</em> pays ont plus souvent des années de baisse d'import, "
            "contrairement à ce qu'on attendrait. "
            "Explication : les grandes économies (USA, Chine, UE) absorbent les chocs mondiaux "
            "(crises financières, COVID) de manière plus cyclique — "
            "leurs grandes dépendances d'import deviennent fragiles en récession. "
            "Les petits pays ont des imports plus stables car plus contraints "
            "(pas de boom/bust cyclique de la demande domestique). "
            "Aucun signal [E] significatif — le pattern est piloté par la taille [S]."
        ),

        "h6_question": "L'inflation rend-elle les imports plus volatils ? (Signal absent pour les exports)",
        "h6a_title": "Volatilité des imports vs inflation % (ρ[E]=+0,20)",
        "inflation_cpi": "Inflation % (IPC)",
        "import_volatility_pct": "Volatilité des imports (%)",
        "h6b_title": "Volatilité des imports vs PIB/habitant (ρ[E]=-0,24)",
        "h6_finding": (
            "<strong>Signal de l'inflation — spécifique aux imports (H.6)</strong> : "
            "Inflation ρ=+0,20 (p<0,01) pour la volatilité des imports, "
            "absent pour la volatilité des exports (ρ=+0,08 n.s.). "
            "L'inflation crée de l'incertitude dans les décisions d'achat → "
            "comportement d'import erratique. "
            "PIB/hab ρ=-0,24 : les pays riches importent plus régulièrement "
            "(coussin financier contre les chocs d'offre)."
        ),
        "h7_note": (
            "<strong>H.7 · Résilience des imports — Résultat nul absolu</strong> : "
            "Question testée : «&nbsp;Les fondamentaux macro protègent-ils la résilience "
            "des imports face aux chocs ?&nbsp;» — Réponse : non, "
            "et paradoxalement PIB (US$) ρ=-0,10 (n.s., légèrement négatif) — "
            "les grands pays sont <em>marginalement moins résilients</em> "
            "(miroir du paradoxe H.5 : les grandes dépendances d'import = plus exposées). "
            "OLS R²=0,013 — proche de zéro. C'est le résultat le plus nul de tout G+H. "
            "La résilience des imports est entièrement idiosyncratique."
        ),

        "h8_label": "H.8 · Imports vs Exports — Ce qui diffère structurellement",
        "h8_finding1": (
            "<strong>Niveau : les imports sont encore mieux expliqués que les exports</strong> : "
            "Part des imports ρ=0,96 > Part des exports ρ=0,94. "
            "Import/habitant : 4 signaux [E] forts (PIB/hab 0,925, Dépense% 0,57, "
            "Inflation -0,47, Taux réel -0,36) contre Export/habitant : 2–3 signaux. "
            "L'import est plus mécaniquement lié aux fondamentaux macro."
        ),
        "h8_finding2": (
            "<strong>Dynamique : les imports sont légèrement plus prévisibles</strong> : "
            "CAGR Import ρ≈{rho_m:+.2f} vs CAGR Export ρ≈{rho_x:+.2f} (CAGR PIB). "
            "Inflation ρ=+0,20 pour la volatilité des imports — signal absent pour les exports. "
            "Les imports suivent la demande domestique, les exports suivent "
            "la compétitivité — plus idiosyncratique par nature."
        ),
        "h8_finding3": (
            "<strong>Anomalies par pays</strong> : "
            "USA = sur-importateur (part des imports 10,7%, efficience export ~0,65×) ; "
            "Allemagne = sur-exportateur (efficience ~2,6×) ; "
            "Chine = paradoxe de l'import (CAGR négatif malgré un PIB positif = substitution). "
            "Ces patterns confirment que le macro décrit la structure, pas la trajectoire."
        ),

        "gh_synth_label": "G+H · Synthèse — Niveau vs Dynamique",
        "gh_synth_table_html": """
        <div style="font-size:13px;color:#A8C0D8;line-height:1.9;margin-bottom:1.2rem;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #304870;">
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Variable</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Signal n°1</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">ρ (n°1)</th>
            <th style="text-align:center;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Flag</th>
            <th style="text-align:left;padding:.35rem .7rem;color:#6890B0;font-weight:600;
                       font-size:11px;text-transform:uppercase;">Conclusion</th>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Part des exports</td>
            <td style="padding:.35rem .7rem;text-align:center;">PIB (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0,94</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Taille déterministe</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Part des imports</td>
            <td style="padding:.35rem .7rem;text-align:center;">PIB (US$)</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0,96</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[S]</span></td>
            <td style="padding:.35rem .7rem;">Encore plus mécanique</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/habitant</td>
            <td style="padding:.35rem .7rem;text-align:center;">PIB/hab</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0,82</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Richesse + Dépense% (0,50)</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/habitant</td>
            <td style="padding:.35rem .7rem;text-align:center;">PIB/hab</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#10B981;">+0,925</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#60A5FA;font-weight:600;">
              4 signaux [E] forts</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Export/PIB</td>
            <td style="padding:.35rem .7rem;text-align:center;">CA% PIB</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0,41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Modéré — excédent → ouverture</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">Import/PIB</td>
            <td style="padding:.35rem .7rem;text-align:center;">Dépense%</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">+0,36</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Plus fort ici que Export/PIB</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Export</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Idiosyncratique</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">CAGR Import</td>
            <td style="padding:.35rem .7rem;text-align:center;">CAGR PIB</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#F59E0B;">~+0,41</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#F59E0B;">[T]</span></td>
            <td style="padding:.35rem .7rem;">Cycle du PIB → plus prévisible</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;background:#162035;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Volatilité export</td>
            <td style="padding:.35rem .7rem;text-align:center;">Taux réel</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0,20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;">Faible</td>
          </tr>
          <tr style="border-bottom:1px solid #1A2840;">
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Volatilité import</td>
            <td style="padding:.35rem .7rem;text-align:center;">Inflation</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#6890B0;">+0,20</td>
            <td style="padding:.35rem .7rem;text-align:center;">
              <span style="color:#10B981;">[E]</span></td>
            <td style="padding:.35rem .7rem;color:#A78BFA;">Signal propre à H</td>
          </tr>
          <tr>
            <td style="padding:.35rem .7rem;font-weight:600;color:#EEF4FB;">
              Résilience X/M</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;text-align:center;color:#EF4444;">n.s.</td>
            <td style="padding:.35rem .7rem;text-align:center;">—</td>
            <td style="padding:.35rem .7rem;color:#EF4444;">Nul — idiosyncratique</td>
          </tr>
        </table>
        <div style="margin-top:.7rem;font-size:11px;color:#4A6890;">
          [S] = Lié à la taille · [E] = Économique (richesse/structure) ·
          [T] = Tautologique · n.s. = non significatif
        </div>
        </div>
        """,
        "gh_key_message": "Message clé",
        "gh_key_finding1": (
            "<strong>Niveau = prévisible par le macro</strong> : "
            "la taille [S] explique qui domine le commerce mondial (ρ>0,94). "
            "La richesse et la structure [E] (PIB/hab, CA%, Dépense%, Taux réel) "
            "expliquent l'efficience et l'ouverture (ρ=0,28–0,925). "
            "Ces relations sont <em>descriptives et structurelles</em>."
        ),
        "gh_key_finding2": (
            "<strong>Dynamique = idiosyncratique</strong> : "
            "la croissance, la régularité et la résilience des exports ne répondent pas "
            "aux 16 indicateurs macro. "
            "Exception partielle : les imports suivent légèrement le cycle du PIB "
            "(CAGR Import ρ≈0,41) car la demande domestique tire mécaniquement les imports. "
            "Ce qu'un pays <em>fait</em> de sa position de départ reste imprévisible."
        ),

        "ij_heat_title": "Corrélations [E] — Variables de balance & d'ouverture (Onglet I) · * = p<0,05",
        "subtab_i": "I · Balance",
        "subtab_j": "J · Quadrants & Profils",

        "i_label": "I · Balance commerciale × Macro — Le CA% comme signal central",
        "i_intro": (
            "La section I relie les variables de balance "
            "(ratio de couverture, balance/PIB, profil de balance) aux indicateurs macro. "
            "Un signal [E] domine de façon cohérente sur trois angles différents."
        ),
        "i1_question": "Les pays excédentaires (CA% > 0) couvrent-ils mieux leurs imports par leurs exports ?",
        "i1a_title": "Ratio de couverture (X/M) vs CA% PIB (ρ[E]=+0,53)",
        "coverage_ratio_title": "Ratio de couverture (X/M)",
        "xm_equal_label": "X = M",
        "i1b_title": "Ratio de couverture × [E]",
        "i1_finding": (
            "<strong>CA% [E] ρ=+0,53 — signal causal le plus net en I</strong> : "
            "un excédent courant implique que les exports couvrent structurellement "
            "les imports (le CA% inclut revenus + transferts, donc la relation n'est pas "
            "circulaire). Les profils Strong/Good se concentrent au-dessus de la ligne X=M "
            "et à droite de CA%=0."
        ),

        "i3_question": "La balance commerciale (X−M)/PIB reflète-t-elle directement le compte courant ?",
        "i3a_title": "Balance/PIB vs CA% PIB (ρ[E]=+0,48)",
        "balance_gdp_title": "(X−M) / PIB",
        "i3b_title": "Balance/PIB vs PIB/habitant (ρ={rho:+.3f})",
        "i3_finding": (
            "<strong>CA% ρ=+0,48 — deuxième confirmation indépendante</strong> : "
            "trois angles différents (ratio de couverture I.1, balance/PIB I.3, profil I.5) "
            "convergent vers le même signal. La relation n'est pas circulaire : "
            "le CA% inclut les revenus d'investissement et les transferts courants, "
            "pas seulement la balance des biens/services."
        ),

        "i5_question": "Quels indicateurs [E] discriminent le mieux les profils de balance commerciale ?",
        "i5a_title": "CA% PIB par profil de balance (KW η²=0,241)",
        "i5b_title": "PIB par habitant par profil de balance (η² modéré)",
        "i5_finding": (
            "<strong>Profil de balance × KW — forte discrimination</strong> : "
            "CA% PIB η²=0,241 (élevé) — le signal le plus fort de toute la section I. "
            "Médianes : Strong Surplus ~+2%, Slight Surplus +1,7%, "
            "Slight Deficit -1,1%, Structural Deficit -4,8%. "
            "Progression monotone et robuste (Spearman et KW cohérents). "
            "Le PIB/hab discrimine modérément : Structural Deficit = pays pauvres en médiane, "
            "mais forte variance (des pays riches ont aussi des déficits structurels)."
        ),

        "i_other_expander": "Sections testées — résultats référencés dans l'onglet D",
        "i2_note": (
            "<strong>I.2 · Intensité commerciale × Dépense% (ρ=+0,34)</strong> : "
            "analysé dans l'onglet D · D.1 Intensité commerciale. "
            "Résultat : Dépense% [E] ρ=+0,34 — les États développés ont une "
            "économie plus ouverte au commerce. Voir D.1 pour l'analyse complète."
        ),
        "i4_note": (
            "<strong>I.4 · Élasticité commerciale × Macro</strong> : "
            "analysé dans l'onglet D · D.2. "
            "Résultat : le macro ne prédit pas l'élasticité Commerce/PIB. "
            "La situation (Export-led, GDP-led, etc.) est idiosyncratique. "
            "Voir D.2 pour la classification à 6 situations."
        ),

        "i_synth_label": "I · Synthèse — Le CA% comme signal central",
        "i_synth_finding": (
            "<strong>Le CA% PIB est le signal [E] le plus cohérent de la section I</strong> : "
            "trois analyses indépendantes convergent : "
            "I.1 Ratio de couverture ρ=+0,53, I.3 Balance/PIB ρ=+0,48, "
            "I.5 Profil de balance η²=0,241. "
            "Interprétation : le compte courant capture la position compétitive externe "
            "plus complètement que la seule balance commerciale."
        ),

        "j_label": "J · Quadrants & profils commerciaux × Macro",
        "j_intro": (
            "La section J teste si les différents "
            "systèmes de classification commerciale (structurel, statique, dynamique, stratégique) "
            "correspondent à des fondamentaux macro distincts. "
            "Règle générale : <strong>statique = expliqué</strong>, "
            "<strong>dynamique = idiosyncratique</strong>."
        ),
        "j1_question": "Les profils structurels (Strong/Good/Moderate/Weak) correspondent-ils à des fondamentaux macro différents ?",
        "j1a_title": "PIB/habitant par profil structurel (η²[E]=0,137)",
        "j1b_title": "CA% PIB par profil structurel (η²[E]=0,099)",
        "j1_finding": (
            "<strong>Profil structurel partiellement expliqué par le macro</strong> : "
            "η² PIB/hab=0,137 (élevé), CA%=0,099 (moyen). "
            "Nuance importante : le structural_score est dominé à 60% par z_export_share — "
            "un grand pays peut atteindre 'Strong' par la seule taille, "
            "indépendamment de sa performance dynamique. "
            "<strong>Contradiction allemande</strong> : score structurel le plus élevé (+4,6) "
            "mais classé Declining en J.3 — la taille masque la détérioration dynamique."
        ),

        "j2_question": "Le quadrant statique (Dominant/Balanced/Vulnerable/Marginal) est-il le mieux discriminé par le CA% ?",
        "j2a_title": "CA% PIB par quadrant statique (η²[E]=0,250 — max de J)",
        "j2b_title": "PIB/habitant par quadrant statique (η²[E]=0,128)",
        "j2_finding": (
            "<strong>Signal le plus fort de toute la section J</strong> : "
            "CA% η²=0,250 (KW H=47,0, p<0,0001). "
            "Pleine cohérence avec I.1 (ρ=0,53), I.3 (ρ=0,48), I.5 (η²=0,241) : "
            "quatre analyses indépendantes confirment que le CA% est "
            "le déterminant macro du positionnement statique. "
            "Dominant = pays excédentaires en médiane, "
            "Vulnerable = pays déficitaires malgré une grande taille."
        ),

        "j3_question": "Le quadrant dynamique (Star/Risky Grower/Stable/Declining) est-il prévisible à partir des fondamentaux macro ?",
        "j3_note": (
            "<strong>J.3 · Quadrant dynamique — Résultat nul [E]</strong> : "
            "aucun indicateur macro ne discrimine significativement entre "
            "Star, Risky Grower, Stable Stagnant et Declining. "
            "C'est la confirmation la plus forte des résultats de G.2/G.3/G.5 : "
            "le quadrant dynamique synthétise la croissance et la stabilité des exports — "
            "deux dimensions toutes deux idiosyncratiques. "
            "<strong>Nuance allemande</strong> : classée Declining (z_cagr < médiane) "
            "mais parmi les Stars en J.4 (cagr absolu > 0%) — la même performance peut être "
            "Declining en relatif et Growing en absolu selon le seuil."
        ),
        "j4_note": (
            "<strong>J.4 · Quadrant stratégique (Stars/Cash Cow/Emerging/Marginal)</strong> : "
            "PIB/hab η²=0,137, CA% η²=0,096 — signal présent mais modéré. "
            "Appartenir aux Stars ≠ qualité structurelle : plusieurs Stars montrent "
            "un faible ratio de couverture (I.1) et un déficit de CA (I.3). "
            "Le cadre BCG capture taille × croissance mais ignore la dimension de balance — "
            "complémentaire au quadrant statique J.2, non substituable."
        ),

        "j_synth_label": "J · Synthèse — Statique expliqué, dynamique idiosyncratique",
        "j_synth_finding1": (
            "<strong>Positionnement statique = expliqué par le macro</strong> : "
            "J.1 (profil structurel η²=0,137) et J.2 (quadrant statique η²=0,250) "
            "confirment que le CA% et le PIB/hab déterminent où un pays se positionne "
            "dans le commerce mondial en termes de taille et de balance."
        ),
        "j_synth_finding2": (
            "<strong>Dynamique = idiosyncratique</strong> : "
            "J.3 (quadrant dynamique) = signal macro nul. "
            "Cohérent avec G.2 (CAGR Export n.s.), G.3 (neg.freq n.s.), G.5 (résilience n.s.). "
            "Même conclusion à trois niveaux d'analyse différents."
        ),
        "j_synth_finding3": (
            "<strong>Enrichissement G/H/I → convergence sur le CA%</strong> : "
            "I.1 ρ=+0,53, I.3 ρ=+0,48, I.5 η²=0,241, J.2 η²=0,250 — "
            "quatre analyses indépendantes dans deux sections différentes "
            "convergent sur le même indicateur. "
            "Le CA% PIB est le signal [E] causal le plus robuste de tout G→J."
        ),

        "k_label": "K · Scores composites × Macro",
        "k_intro": (
            "La section K analyse les scores composites "
            "(structurel, résilience bilatérale, combiné) : sont-ils expliqués par le macro ? "
            "Résultat clé : deux dimensions avec deux logiques analytiques distinctes."
        ),
        "k_heat_title": "Corrélations [E] — Scores composites (Onglet K) · * = p<0,05",
        "k1_question": "La résilience bilatérale (diversification géographique des exports) est-elle liée aux fondamentaux macro ?",
        "k1_note": (
            "<strong>K.1 · g_norm_rank (résilience bilatérale) — Résultat nul</strong> : "
            "aucun indicateur macro ne prédit la concentration géographique des exports. "
            "Hypothèse testée : un meilleur CA% → une meilleure diversification des partenaires. "
            "Résultat : tous ρ<0,18, p>0,05. "
            "La diversification des exports est le produit des politiques commerciales, "
            "des accords bilatéraux et de la géographie — pas des fondamentaux macro agrégés."
        ),

        "k3_question": "Le profil combiné (Champion/Competitive/Vulnerable/Fragile) discrimine-t-il les fondamentaux macro [E] ?",
        "k3a_title": "PIB/habitant par profil combiné (KW H=19,3, p=0,0002)",
        "k3b_title": "CA% PIB par profil combiné",
        "k3_finding": (
            "<strong>Global Champion clairement distinct</strong> : "
            "PIB/hab médian pour Global Champion ~34k$ contre Competitive/Vulnerable/Fragile à 5–8k$ — "
            "écart de 4 à 7×. Le profil combiné identifie une véritable élite économique. "
            "Le CA% discrimine modérément : les Champions tendent vers l'excédent, "
            "les Fragile vers le déficit, mais la variance intra-groupe est élevée. "
            "Le profil combiné est un signal fiable du niveau de développement."
        ),

        "k1k2_question": "Les scores continus (combined_score, structural_score) sont-ils corrélés aux indicateurs [E] ?",
        "k1_chart_title": "Combined Score vs PIB/habitant (ρ[E]=+0,33)",
        "combined_score_title": "Combined Score",
        "structural_score_title": "Structural Score",
        "k1_finding": (
            "<strong>Combined Score ρ[E]=+0,33 (PIB/hab)</strong> : signal modéré. "
            "L'ajout du score de résilience bilatérale (g_norm_rank, non expliqué par le macro) "
            "dilue légèrement le signal du structural_score (ρ=+0,32). "
            "La différence Δρ≈0 confirme que la résilience bilatérale n'ajoute ni "
            "ne retire de signal macro — les deux dimensions sont orthogonales."
        ),

        "k_hier_label": "K · Hiérarchie des signaux [E] — Synthèse finale",
        "k_hier_intro": (
            "Classement des indicateurs [E] par |ρ| moyen sur les 44 variables commerciales "
            "— récapitulatif de toutes les sections G à K."
        ),
        "k_hier_chart_title": "Indicateurs [E] — |ρ| moyen sur toutes les variables commerciales (G–K)",
        "k_hier_finding": (
            "<strong>Hiérarchie G→K confirmée</strong> : "
            "PIB/hab > CA% > Taux réel > Dépense% > Inflation > Chômage. "
            "PIB/hab : signal [E] dominant (richesse → tous les aspects de la position commerciale). "
            "CA% : second signal, cohérent sur 4+ analyses indépendantes (I.1/I.3/I.5/J.2). "
            "Taux réel et Dépense% : signaux secondaires, pertinents pour l'ouverture (X/PIB, M/PIB). "
            "Inflation et Chômage : signaux faibles, surtout pour la volatilité des imports."
        ),

        "k5_label": "K.5 · Deux dimensions, deux logiques",
        "k5_finding1": (
            "<strong>F (score structurel) = expliqué</strong> : "
            "PIB/hab ρ=+0,32, CA% ρ=+0,30 — cohérent sur tout G/H/I/J/K. "
            "Le score structurel mesure la position dans le commerce mondial — "
            "déterminée par la richesse et la balance externe."
        ),
        "k5_finding2": (
            "<strong>G (résilience bilatérale = g_norm_rank) = inexpliqué</strong> : "
            "tous les indicateurs n.s. La concentration géographique des partenaires commerciaux "
            "est entièrement idiosyncratique — géographie, histoire, "
            "accords commerciaux, politiques nationales."
        ),
        "k5_finding3": (
            "<strong>Implication pour le combined_score</strong> : "
            "en pondérant 70% structurel + 30% bilatéral, le combined_score hérite "
            "du signal macro du structurel et de l'idiosyncrasie du bilatéral. "
            "C'est analytiquement correct : les deux dimensions capturent des réalités différentes."
        ),

        "k_general_expander": "Conclusion générale G→K",
        "k_general_finding1": (
            "<strong>Structure commerciale = expliquée par le macro (niveau)</strong> : "
            "Part Export/Import [S], Export/Import par habitant [E], "
            "Ratio de couverture [E], profils statiques [E]. "
            "La taille prévaut (PIB US$ ρ>0,94), la richesse affine (PIB/hab ρ=0,33–0,92)."
        ),
        "k_general_finding2": (
            "<strong>Dynamique commerciale = idiosyncratique</strong> : "
            "CAGR, volatilité, résilience, régularité, quadrant dynamique — "
            "tous [E] non significatifs. "
            "Ce qu'un pays fait de sa position structurelle dépend de facteurs "
            "non capturés par les 16 indicateurs macro sur 10 ans."
        ),
        "k_general_finding3": (
            "<strong>Taille vs Richesse</strong> : "
            "être grand (PIB US$) prévaut pour le volume mondial (ρ=0,94–0,96). "
            "Être riche (PIB/hab) prévaut pour l'efficience par habitant (ρ=0,82–0,925). "
            "Le CA% est le signal causal le plus robuste : excédent → compétitivité externe."
        ),
    },
}

LINK_FILES  = ROOT / "Analysis" / "Analysis_files" / "Link"
TRADE_FILES = ROOT / "Analysis" / "Analysis_files" / "Trade"
IND_FILES   = ROOT / "Analysis" / "Analysis_files" / "Indicators"

# ── Indicator column names ──────────────────────────────────────
GDP    = "GDP (current US$)"
GDP_pc = "GDP per capita (current US$)"
EXP    = "Exports of goods and services (current US$)"
IMP    = "Imports of goods and services (current US$)"
POP    = "Population, total"
CA_PCT = "Current account balance (% of GDP)"
INFL   = "Inflation, consumer prices (annual %)"
UNEMP  = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
RATE    = "Real interest rate (%)"
EXPENSE = "Expense (% of GDP)"

# ── Profile colors ──────────────────────────────────────────────
_SIZE_C  = {"Leader": "#10B981", "Major": "#60A5FA", "Mid": "#F59E0B", "Small": "#6890B0"}
_COMB_C  = {"Global Champion": "#10B981", "Competitive": "#60A5FA",
            "Vulnerable": "#F59E0B", "Fragile": "#EF4444"}
_STRUC_C = {"Strong": "#10B981", "Good": "#60A5FA", "Moderate": "#F59E0B", "Weak": "#EF4444"}
_BAL_C   = {"Strong Surplus": "#10B981", "Slight Surplus": "#60A5FA",
            "Slight Deficit": "#F59E0B", "Structural Deficit": "#EF4444"}
_STATIC_C = {"Dominant": "#10B981", "Balanced": "#60A5FA",
             "Vulnerable": "#F59E0B", "Marginal": "#EF4444"}

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def build_master():
    df_str  = pd.read_csv(TRADE_FILES / "df_structural.csv")
    df_comb = pd.read_csv(TRADE_FILES / "df_combined.csv")
    df_ind  = pd.read_csv(IND_FILES / "df_indicators.csv",
                          usecols=["country_name", "entity_type",
                                   "indicator_name", "year", "value"])
    df_sp   = pd.read_csv(LINK_FILES / "df_spearman.csv")
    df_kw   = pd.read_csv(LINK_FILES / "df_kw.csv")

    ci = df_ind[df_ind["entity_type"] == "COUNTRY"].copy()

    # Latest macro value per country
    df_latest = (ci.sort_values("year")
                 .groupby(["country_name", "indicator_name"])["value"].last()
                 .unstack().reset_index())
    df_latest.columns.name = None

    # CAGR of GDP (for trade elasticity)
    df_gdp = ci[ci["indicator_name"] == GDP].sort_values(["country_name", "year"])
    g_first = df_gdp.groupby("country_name")["value"].first()
    g_last  = df_gdp.groupby("country_name")["value"].last()
    g_ny    = df_gdp.groupby("country_name")["year"].agg(lambda x: x.max() - x.min())
    cagr_gdp = (
        (g_last / g_first.replace(0, np.nan))
        .pow(1 / g_ny.replace(0, np.nan)) - 1
    ).rename("cagr_gdp").reset_index()

    # Build master — combine all sources
    # Only pull columns unique to df_comb (avoid merge conflicts with df_str overlaps)
    _comb_unique = ["country_name", "combined_score", "f_norm_rank", "g_norm_rank",
                    "combined_profile", "strategic_quadrant_combined"]
    df_m = (df_str
            .merge(df_comb[[c for c in _comb_unique if c in df_comb.columns]],
                   on="country_name", how="inner")
            .merge(df_latest, on="country_name", how="left")
            .merge(cagr_gdp,  on="country_name", how="left"))

    # Derived metrics
    df_m["trade_intensity"]    = (df_m[EXP] + df_m[IMP]) / df_m[GDP]
    df_m["export_dep"]         = df_m[EXP] / df_m[GDP]
    df_m["import_dep"]         = df_m[IMP] / df_m[GDP]
    df_m["balance_dep"]        = (df_m[EXP] - df_m[IMP]) / df_m[GDP]
    df_m["export_pc"]          = df_m["total_export"] / df_m[POP]
    df_m["import_pc"]          = df_m["total_import"] / df_m[POP]
    _safe_gdp                  = df_m["cagr_gdp"].replace(0, np.nan)
    df_m["trade_elasticity_x"] = df_m["cagr_export"] / _safe_gdp
    df_m["trade_elasticity_m"] = df_m["cagr_import"] / _safe_gdp

    return df_m, df_sp, df_kw

df_m, df_sp, df_kw = build_master()

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


def _scatter(df, x, y, color_col, color_map, title, xtitle, ytitle,
             xlog=False, ylog=False, height=400, x_clip=None, y_clip=None,
             x_fmt=None, y_fmt=None):
    """Scatter plot colored by categorical profile column.
    x_fmt / y_fmt: None = 3 decimal places, "usd" = $K/B/T
    """
    df_p = df[[x, y, color_col, "country_name"]].replace([np.inf, -np.inf], np.nan).dropna()
    if x_clip:
        df_p = df_p[(df_p[x] >= x_clip[0]) & (df_p[x] <= x_clip[1])]
    if y_clip:
        df_p = df_p[(df_p[y] >= y_clip[0]) & (df_p[y] <= y_clip[1])]
    _f = lambda v, fmt: (fmt_usd(v) if fmt == "usd" else f"{v:.3f}")
    df_p = df_p.copy()
    df_p["_xf"] = df_p[x].apply(lambda v: _f(v, x_fmt))
    df_p["_yf"] = df_p[y].apply(lambda v: _f(v, y_fmt))
    _ht = (f"<b>%{{text}}</b><br>{xtitle}: %{{customdata[0]}}<br>"
           f"{ytitle}: %{{customdata[1]}}<extra></extra>")
    fig = go.Figure()
    for cat in sorted(df_p[color_col].unique()):
        sub = df_p[df_p[color_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y],
            mode="markers",
            name=cat,
            customdata=sub[["_xf", "_yf"]].values,
            marker=dict(color=color_map.get(cat, "#304870"), size=7, opacity=0.8),
            text=sub["country_name"],
            hovertemplate=_ht,
        ))
    _pl(fig, title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=height,
        xaxis=dict(type="log" if xlog else "linear", gridcolor="#304870"),
        yaxis=dict(type="log" if ylog else "linear", gridcolor="#304870"))
    return fig


def _top_correlates(trade_var, n=6, flag_filter=None):
    """Return top correlated macro indicators for a given trade variable."""
    df_t = df_sp[df_sp["trade_var"] == trade_var].copy()
    if flag_filter:
        df_t = df_t[df_t["ind_flag"] == flag_filter]
    return df_t.sort_values("abs_rho", ascending=False).head(n)


def _corr_bar(df_tc, title, height=320):
    """Horizontal bar chart of top correlates."""
    _fc = {"[E]": "#10B981", "[T]": "#EF4444", "[S]": "#F59E0B"}
    colors = [_fc.get(f, "#6890B0") for f in df_tc["ind_flag"]]
    fig = go.Figure(go.Bar(
        x=df_tc["abs_rho"].values[::-1],
        y=df_tc["ind_label"].values[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=df_tc["rho"].values[::-1].round(3),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ = %{text}<extra></extra>",
    ))
    _pl(fig, title=title, xaxis_title="|ρ|", height=height,
        margin=dict(l=160, r=60, t=50, b=40),
        xaxis=dict(gridcolor="#304870", range=[0, 1.05]))
    return fig


def _tab_heatmap(vars_dict, title, flags=None, height=None):
    """Spearman ρ heatmap: rows = trade variables, cols = macro indicators.
    vars_dict: {trade_var_key: display_label}
    flags: list of ind_flag to include — defaults to [E] + [S], never [T]
    """
    _flags = flags or ["[E]", "[S]"]
    _df = df_sp[
        df_sp["trade_var"].isin(vars_dict.keys()) &
        df_sp["ind_flag"].isin(_flags)
    ].copy()
    if _df.empty:
        return None
    _piv = (_df.pivot_table(index="trade_var", columns="ind_label",
                             values="rho", aggfunc="first")
              .reindex(list(vars_dict.keys())).dropna(how="all"))
    _piv_sig = (_df.pivot_table(index="trade_var", columns="ind_label",
                                 values="sig", aggfunc="first")
                  .reindex(list(vars_dict.keys())))
    _ylabs = [vars_dict.get(v, v) for v in _piv.index]
    _text = []
    for r in _piv.index:
        _row = []
        for c in _piv.columns:
            v = _piv.at[r, c] if c in _piv.columns else np.nan
            try:
                s = _piv_sig.at[r, c]
                sig = bool(s) if not pd.isna(s) else False
            except Exception:
                sig = False
            if pd.isna(v):
                _row.append("")
            else:
                _row.append(f"{v:+.2f}{'*' if sig else ''}")
        _text.append(_row)
    _n_rows = max(len(_piv), 1)
    _n_cols = max(len(_piv.columns), 1)
    _row_h  = 62
    _col_w  = 90
    _h = height or max(300, _row_h * _n_rows + 220)
    _w_est  = _col_w * _n_cols + 250          # approx width for colorbar placement
    fig = go.Figure(go.Heatmap(
        z=_piv.values,
        x=list(_piv.columns),
        y=_ylabs,
        text=_text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[
            [0.0,  "#1060B0"],
            [0.35, "#1A3060"],
            [0.5,  "#1A2840"],
            [0.65, "#1A3820"],
            [1.0,  "#10B981"],
        ],
        zmid=0, zmin=-0.6, zmax=0.6,
        showscale=True,
        colorbar=dict(
            title=dict(text="ρ", font=dict(color="#7090B0", size=12)),
            thickness=14,
            len=0.7,
            yanchor="middle", y=0.5,
            tickvals=[-0.5, -0.25, 0, 0.25, 0.5],
            tickfont=dict(color="#7090B0", size=10),
            outlinewidth=0,
        ),
    ))
    _pl(fig, title=title, height=_h,
        margin=dict(l=190, r=80, t=55, b=170),
        xaxis=dict(
            tickangle=-40,
            side="bottom",
            gridcolor="#304870",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#304870",
            tickfont=dict(size=11),
            autorange="reversed",
        ))
    return fig


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown(f'<div class="page-eyebrow">{t(STRINGS, "eyebrow")}</div>',
            unsafe_allow_html=True)
st.title(t(STRINGS, "title"))
st.caption(t(STRINGS, "caption"))
thin_divider()

# ── KPI row ────────────────────────────────────────────────────
_n_sig_e = int(df_sp[(df_sp["sig"]) & (df_sp["ind_flag"] == "[E]")].shape[0])
_top_pair = df_sp[df_sp["sig"]].nlargest(1, "abs_rho").iloc[0]
_top_e    = (df_sp[(df_sp["ind_flag"] == "[E]") & df_sp["sig"]]
             .groupby("ind_label")["abs_rho"].mean().idxmax())

for col, (label, val, color) in zip(st.columns(4), [
    (t(STRINGS, "kpi_countries"),  str(len(df_m)),               "teal"),
    (t(STRINGS, "kpi_sig_e"),      str(_n_sig_e),                "blue"),
    (t(STRINGS, "kpi_max_signal"), f"ρ={_top_pair['rho']:+.3f}", "amber"),
    (t(STRINGS, "kpi_top_signal_e"), _top_e,                     ""),
]):
    col.markdown(kpi(label, val, color), unsafe_allow_html=True)

thin_divider()

# ════════════════════════════════════════════════════════════════
# E+F — CLASSIFICATION DES INDICATEURS
# ════════════════════════════════════════════════════════════════

section_label(t(STRINGS, "ef_label"))
st.markdown(
    f"<p style='font-size:13px;color:#7090B0;margin-bottom:.9rem;'>{t(STRINGS, 'ef_intro')}</p>",
    unsafe_allow_html=True,
)

_flag_cols = st.columns(3)
with _flag_cols[0]:
    st.markdown(
        '<div style="background:#201820;border:1px solid #38242A;'
        'border-left:4px solid #EF4444;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        f'<span style="color:#EF4444;font-size:13px;font-weight:700;">'
        f'{t(STRINGS, "flag_t_title")}</span>'
        f'<span style="color:#A8B8CC;font-size:13px;">{t(STRINGS, "flag_t_body")}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _flag_cols[1]:
    st.markdown(
        '<div style="background:#201E18;border:1px solid #383020;'
        'border-left:4px solid #F59E0B;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        f'<span style="color:#F59E0B;font-size:13px;font-weight:700;">'
        f'{t(STRINGS, "flag_s_title")}</span>'
        f'<span style="color:#A8B8CC;font-size:13px;">{t(STRINGS, "flag_s_body")}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _flag_cols[2]:
    st.markdown(
        '<div style="background:#182020;border:1px solid #243830;'
        'border-left:4px solid #10B981;border-radius:0 9px 9px 0;'
        'padding:.85rem 1.05rem;min-height:90px;">'
        f'<span style="color:#10B981;font-size:13px;font-weight:700;">'
        f'{t(STRINGS, "flag_e_title")}</span>'
        f'<span style="color:#A8B8CC;font-size:13px;">{t(STRINGS, "flag_e_body")}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f"<p style='font-size:12px;color:#4A6890;margin:.6rem 0 .5rem 0;'>{t(STRINGS, 'ef_footer')}</p>",
    unsafe_allow_html=True,
)
thin_divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_d, tab_gh, tab_ij, tab_k = st.tabs([
    t(STRINGS, "tab_d"),
    t(STRINGS, "tab_gh"),
    t(STRINGS, "tab_ij"),
    t(STRINGS, "tab_k"),
])

# ────────────────────────────────────────────────────────────────
# TAB D — TRADE / GDP BRIDGE
# ────────────────────────────────────────────────────────────────

def _question(text):
    st.markdown(
        f'<div style="background:#1A2840;border:1px solid #304870;'
        f'border-left:3px solid #60A5FA;border-radius:0 10px 10px 0;'
        f'padding:.6rem 1rem;margin-bottom:.8rem;">'
        f'<span style="color:#93C5FD;font-size:11px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.07em;">{t(STRINGS, "question_label")}</span><br>'
        f'<span style="color:#EEF4FB;font-size:14px;">{text}</span></div>',
        unsafe_allow_html=True,
    )

def _note(text):
    st.markdown(
        f'<div style="background:#182820;border:1px solid #2A4034;'
        f'border-left:3px solid #34D399;border-radius:0 10px 10px 0;'
        f'padding:.5rem .9rem;margin:.4rem 0 .7rem 0;">'
        f'<span style="color:#34D399;font-size:11px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.07em;">{t(STRINGS, "note_label")}</span><br>'
        f'<span style="color:#9DBFB0;font-size:13px;">{text}</span></div>',
        unsafe_allow_html=True,
    )

def _classify_elast(cx, cg):
    if cx > 0 and cg > 0:
        return "Export-led" if cx > cg else "GDP-led"
    elif cx > 0 and cg <= 0:
        return "Export Resilient"
    elif cx <= 0 and cg > 0:
        return "Export Contraction"
    elif cx < 0 and cg < 0:
        return "Moderate Decline" if cx > cg else "Severe Decline"
    return "Autre"

_ELAST_C = {
    "Export-led":         "#10B981",
    "GDP-led":            "#60A5FA",
    "Export Resilient":   "#A78BFA",
    "Export Contraction": "#F59E0B",
    "Moderate Decline":   "#FB923C",
    "Severe Decline":     "#EF4444",
}


def _kw_box(col, profile_col, order, color_map, height=390, fmt_fn=None):
    """Boxplot of `col` grouped by `profile_col`, ordered by `order`.
    fmt_fn: optional callable to format y values for hover (e.g. fmt_usd).
    """
    fig = go.Figure()
    for cat in order:
        sub = df_m[df_m[profile_col] == cat][["country_name", col]].dropna().copy()
        if sub.empty:
            continue
        if fmt_fn:
            sub["_yf"] = sub[col].apply(fmt_fn)
            fig.add_trace(go.Box(
                y=sub[col].values, name=cat,
                text=sub["country_name"].values,
                customdata=sub[["_yf"]].values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<extra></extra>",
                marker_color=color_map.get(cat, "#304870"),
                boxpoints="outliers", line=dict(width=1.5),
            ))
        else:
            fig.add_trace(go.Box(
                y=sub[col].values, name=cat,
                text=sub["country_name"].values,
                hovertemplate="<b>%{text}</b><br>%{y:.3f}<extra></extra>",
                marker_color=color_map.get(cat, "#304870"),
                boxpoints="outliers", line=dict(width=1.5),
            ))
    return fig


def _add_refs(fig, x_col, y_col, countries, df=None, x_scale=1, y_scale=1, ht=None,
              x_fmt=None, y_fmt=None):
    """Add gold star markers for reference countries on a scatter figure."""
    _src = df if df is not None else df_m
    _ref = _src[_src["country_name"].isin(countries)][[x_col, y_col, "country_name"]].dropna()
    if _ref.empty:
        return fig
    _xv = _ref[x_col].values * x_scale
    _yv = _ref[y_col].values * y_scale
    if ht:
        _cd, _ht = None, ht
    else:
        _f   = lambda v, fmt: (fmt_usd(v) if fmt == "usd" else f"{v:.3f}")
        _cd  = list(zip([_f(v, x_fmt) for v in _xv], [_f(v, y_fmt) for v in _yv]))
        _ht  = "<b>%{text}</b><br>x: %{customdata[0]}<br>y: %{customdata[1]}<extra></extra>"
    fig.add_trace(go.Scatter(
        x=_xv, y=_yv,
        mode="markers+text",
        marker=dict(color="#FFD700", size=11, symbol="star",
                    line=dict(color="#FFFFFF", width=1.5)),
        text=_ref["country_name"].values,
        customdata=_cd,
        textposition="top center",
        textfont=dict(size=9, color="#FFD700"),
        hovertemplate=_ht,
        name="Reference", showlegend=False,
    ))
    return fig


with tab_d:
    section_label("D · Trade/GDP Bridge — Derived Metrics")
    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'd_intro')}</p>",
        unsafe_allow_html=True,
    )
    _D_VARS = {
        "trade_intensity": "Trade Intensity (X+M)/GDP",
        "export_dep":      "Export / GDP (X/GDP)",
        "export_pc":       "Export per Capita",
        "cagr_export":     "CAGR Export",
    }
    _fig_d_heat = _tab_heatmap(
        _D_VARS,
        t(STRINGS, "d_heat_title"),
        flags=["[E]"],
    )
    if _fig_d_heat:
        st.plotly_chart(_fig_d_heat, use_container_width=True)
    thin_divider()

    # ── D.1 ────────────────────────────────────────────────────
    section_label(t(STRINGS, "d1_label"))
    _question(t(STRINGS, "d1_question"))

    df_d1 = df_m[["country_name", GDP_pc, "trade_intensity", "size_profile"]].dropna()
    df_d1 = df_d1[(df_d1[GDP_pc] > 0) & (df_d1["trade_intensity"] > 0)
                  & (df_d1["trade_intensity"] < 5)]
    fig_d1 = _scatter(df_d1, GDP_pc, "trade_intensity", "size_profile", _SIZE_C,
                      t(STRINGS, "d1_chart_title"),
                      t(STRINGS, "gdp_pc_log"), t(STRINGS, "trade_intensity_title"),
                      xlog=True, height=460, x_fmt="usd")
    _add_refs(fig_d1, GDP_pc, "trade_intensity",
              ["USA", "China", "Singapore", "Luxembourg"], df=df_d1, x_fmt="usd")
    st.plotly_chart(fig_d1, use_container_width=True)
    finding(t(STRINGS, "d1_finding"))

    thin_divider()

    # ── D.2 ────────────────────────────────────────────────────
    section_label(t(STRINGS, "d2_label"))
    _question(t(STRINGS, "d2_question"))

    df_d2 = df_m[["country_name", "cagr_gdp", "cagr_export", "structural_profile"]].copy()
    df_d2 = df_d2.replace([np.inf, -np.inf], np.nan).dropna(subset=["cagr_gdp", "cagr_export"])
    df_d2 = df_d2[df_d2["cagr_gdp"].abs() < 0.3]
    df_d2 = df_d2[df_d2["cagr_export"].abs() < 0.5]
    df_d2["situation"] = df_d2.apply(
        lambda r: _classify_elast(r["cagr_export"], r["cagr_gdp"]), axis=1)

    col_d2a, col_d2b = st.columns([3, 2])

    with col_d2a:
        fig_d2 = go.Figure()
        for sit, color in _ELAST_C.items():
            sub = df_d2[df_d2["situation"] == sit]
            if sub.empty:
                continue
            fig_d2.add_trace(go.Scatter(
                x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                mode="markers", name=sit,
                marker=dict(color=color, size=7, opacity=0.85),
                text=sub["country_name"],
                hovertemplate=(
                    "<b>%{text}</b> · %{meta}<br>"
                    "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                ),
                meta=sit,
            ))
        _lim = df_d2[["cagr_gdp", "cagr_export"]].stack().abs().quantile(0.95) * 100
        fig_d2.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                         line=dict(color="#6890B0", dash="dash", width=1.5))
        fig_d2.add_vline(x=0, line_color="#304870", line_dash="dot", line_width=1)
        fig_d2.add_hline(y=0, line_color="#304870", line_dash="dot", line_width=1)
        for ann_x, ann_y, ann_txt in [
            (_lim * 0.6,  _lim * 0.95, "Export-led ↗"),
            (_lim * 0.6,  _lim * 0.3,  "GDP-led ↗"),
            (-_lim * 0.6, _lim * 0.6,  "Export Resilient"),
            (_lim * 0.5,  -_lim * 0.7, "Export Contraction"),
        ]:
            fig_d2.add_annotation(x=ann_x, y=ann_y, text=ann_txt, showarrow=False,
                                  font=dict(size=10, color="#6890B0"))
        _pl(fig_d2, title=t(STRINGS, "d2_chart_title"),
            xaxis_title=t(STRINGS, "cagr_gdp_pct"), yaxis_title=t(STRINGS, "cagr_export_pct"), height=420,
            legend=dict(orientation="v", x=1.01, y=1, bgcolor="#1F3050",
                        bordercolor="#304870", borderwidth=1))
        st.plotly_chart(fig_d2, use_container_width=True)

    with col_d2b:
        sit_counts = df_d2["situation"].value_counts().reindex(list(_ELAST_C.keys())).dropna()
        fig_d2b = go.Figure(go.Bar(
            x=sit_counts.values,
            y=sit_counts.index,
            orientation="h",
            marker_color=[_ELAST_C[s] for s in sit_counts.index],
            text=sit_counts.values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>n = %{x}<extra></extra>",
        ))
        _pl(fig_d2b, title=t(STRINGS, "d2b_chart_title"),
            xaxis_title=t(STRINGS, "d2b_x"), height=420,
            margin=dict(l=150, r=50, t=70, b=40),
            xaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_d2b, use_container_width=True)

    finding(t(STRINGS, "d2_finding"))

    with st.expander(t(STRINGS, "d2_expander_title")):
        st.markdown(t(STRINGS, "d2_table_html"), unsafe_allow_html=True)

    thin_divider()

    # ── D.3 ────────────────────────────────────────────────────
    section_label(t(STRINGS, "d3_label"))
    _question(t(STRINGS, "d3_question"))

    col_d3a, col_d3b = st.columns(2)

    with col_d3a:
        df_d3 = df_m[["country_name", GDP_pc, "export_pc", "combined_profile"]].dropna()
        df_d3 = df_d3[(df_d3[GDP_pc] > 0) & (df_d3["export_pc"] > 0)]
        fig_d3 = _scatter(df_d3, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                          t(STRINGS, "d3_chart_title"),
                          t(STRINGS, "gdp_pc_log"), t(STRINGS, "export_pc_log"),
                          xlog=True, ylog=True, height=440, x_fmt="usd", y_fmt="usd")
        st.plotly_chart(fig_d3, use_container_width=True)

    with col_d3b:
        # Rank mover chart: rank by total volume vs rank by per capita
        df_rk = df_m[["country_name", "total_export", "export_pc"]].dropna()
        df_rk["rank_vol"] = df_rk["total_export"].rank(ascending=False)
        df_rk["rank_pc"]  = df_rk["export_pc"].rank(ascending=False)
        df_rk["delta"]    = df_rk["rank_vol"] - df_rk["rank_pc"]  # >0 = gains per capita
        n_show = 12
        top_gain = df_rk.nlargest(n_show, "delta")
        top_lose = df_rk.nsmallest(n_show, "delta")
        df_rk_show = pd.concat([top_gain, top_lose]).drop_duplicates("country_name")
        df_rk_show = df_rk_show.sort_values("delta")

        fig_rk = go.Figure(go.Bar(
            x=df_rk_show["delta"],
            y=df_rk_show["country_name"],
            orientation="h",
            marker_color=["#10B981" if v > 0 else "#EF4444"
                          for v in df_rk_show["delta"]],
            hovertemplate=t(STRINGS, "d3b_hover"),
            customdata=df_rk_show[["rank_vol", "rank_pc"]].values,
        ))
        fig_rk.add_vline(x=0, line_color="#6890B0", line_width=1)
        _pl(fig_rk,
            title=t(STRINGS, "d3b_chart_title"),
            xaxis_title=t(STRINGS, "d3b_x"),
            height=440,
            margin=dict(l=140, r=40, t=70, b=40),
            xaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_rk, use_container_width=True)

    finding(t(STRINGS, "d3_finding"))

# ────────────────────────────────────────────────────────────────
# TAB G-H — EXPORTS & IMPORTS
# ────────────────────────────────────────────────────────────────
with tab_gh:
    _GH_VARS = {
        "export_share_structural": "Export Share (world)",
        "export_dep":              "Export / GDP",
        "export_pc":               "Export / Capita",
        "cagr_export":             "CAGR Export",
        "volatility_export":       "Export Volatility",
        "import_share_structural": "Import Share (world)",
        "import_dep":              "Import / GDP",
        "import_pc":               "Import / Capita",
        "cagr_import":             "CAGR Import",
        "volatility_import":       "Import Volatility",
    }
    _fig_gh_heat = _tab_heatmap(
        _GH_VARS,
        t(STRINGS, "gh_heat_title"),
        flags=["[E]", "[S]"],
    )
    if _fig_gh_heat:
        st.plotly_chart(_fig_gh_heat, use_container_width=True)
    thin_divider()

    subtab_g, subtab_h, subtab_synth = st.tabs([
        t(STRINGS, "subtab_g"), t(STRINGS, "subtab_h"), t(STRINGS, "subtab_synth")
    ])

    # ════════════════════════════════════════════════════
    # G · EXPORTS
    # ════════════════════════════════════════════════════
    with subtab_g:

        # ── G.1 Export Level ─────────────────────────
        section_label(t(STRINGS, "g1_label"))
        st.markdown(
            f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'g1_intro')}</p>",
            unsafe_allow_html=True,
        )

        # G.1.1 Export Share vs GDP
        _question(t(STRINGS, "g11_question"))
        col_g11a, col_g11b = st.columns(2)
        with col_g11a:
            df_g11 = df_m[["country_name", GDP, "export_share_structural",
                            "size_profile"]].dropna()
            df_g11 = df_g11[df_g11[GDP] > 0]
            fig_g11 = _scatter(df_g11, GDP, "export_share_structural", "size_profile", _SIZE_C,
                               t(STRINGS, "g11a_title"),
                               t(STRINGS, "gdp_log"), t(STRINGS, "export_share_pct"),
                               xlog=True, height=390, x_fmt="usd")
            st.plotly_chart(fig_g11, use_container_width=True)
        with col_g11b:
            df_g11b = df_m[["country_name", GDP_pc, "export_pc",
                             "combined_profile"]].dropna()
            df_g11b = df_g11b[(df_g11b[GDP_pc] > 0) & (df_g11b["export_pc"] > 0)]
            fig_g11b = _scatter(df_g11b, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                                t(STRINGS, "g11b_title"),
                                t(STRINGS, "gdp_pc_log"), t(STRINGS, "export_pc_log"),
                                xlog=True, ylog=True, height=390, x_fmt="usd", y_fmt="usd")
            _add_refs(fig_g11b, GDP_pc, "export_pc",
                      ["USA", "China", "Germany", "Japan", "Brazil"], df=df_g11b,
                      x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_g11b, use_container_width=True)
        finding(t(STRINGS, "g11_finding"))
        thin_divider()

        # G.1.2 Export Share vs CA% + Trade Efficiency
        _question(t(STRINGS, "g12_question"))
        col_g12a, col_g12b = st.columns(2)
        with col_g12a:
            df_g12 = df_m[["country_name", CA_PCT, "export_share_structural",
                            "structural_profile"]].dropna()
            df_g12 = df_g12[df_g12["export_share_structural"] < 0.15]
            fig_g12 = _scatter(df_g12, CA_PCT, "export_share_structural",
                               "structural_profile", _STRUC_C,
                               t(STRINGS, "g12a_title"),
                               t(STRINGS, "ca_pct_gdp"), t(STRINGS, "export_share_pct"), height=390)
            st.plotly_chart(fig_g12, use_container_width=True)
        with col_g12b:
            df_eff = df_m[["country_name", GDP, "export_share_structural",
                            "size_profile"]].dropna()
            df_eff = df_eff[df_eff[GDP] > 0].copy()
            world_gdp = df_eff[GDP].sum()
            df_eff["gdp_share"] = df_eff[GDP] / world_gdp
            df_eff["trade_eff"] = (df_eff["export_share_structural"]
                                   / df_eff["gdp_share"].replace(0, np.nan))
            df_eff = df_eff[(df_eff["trade_eff"] > 0) & (df_eff["trade_eff"] < 12)]
            top_eff = df_eff.nlargest(14, "trade_eff")
            bot_eff = df_eff.nsmallest(9, "trade_eff")
            df_eff_show = (pd.concat([top_eff, bot_eff])
                           .drop_duplicates("country_name")
                           .sort_values("trade_eff"))
            fig_eff = go.Figure(go.Bar(
                x=df_eff_show["trade_eff"],
                y=df_eff_show["country_name"],
                orientation="h",
                marker_color=["#10B981" if v > 1 else "#EF4444"
                              for v in df_eff_show["trade_eff"]],
                text=df_eff_show["trade_eff"].round(1),
                textposition="outside",
                hovertemplate=t(STRINGS, "trade_eff_hover"),
            ))
            fig_eff.add_vline(x=1, line_color="#6890B0", line_dash="dash",
                              line_width=1, annotation_text=t(STRINGS, "trade_eff_ref_label"),
                              annotation_font_color="#6890B0")
            _pl(fig_eff,
                title=t(STRINGS, "trade_eff_title"),
                xaxis_title=t(STRINGS, "trade_eff_x"),
                height=390, margin=dict(l=130, r=70, t=50, b=40))
            st.plotly_chart(fig_eff, use_container_width=True)
        finding(t(STRINGS, "g12_finding"))
        thin_divider()

        # G.1.2.3 KW Boxplots — profils de taille × fondamentaux macro
        _question(t(STRINGS, "kw_size_question"))
        col_kw1, col_kw2 = st.columns(2)
        _kw_order = ["Leader", "Major", "Mid", "Small"]
        with col_kw1:
            fig_kw1 = go.Figure()
            for cat in _kw_order:
                sub = df_m[df_m["size_profile"] == cat][[GDP_pc, "country_name"]].dropna().copy()
                sub["_yf"] = sub[GDP_pc].apply(fmt_usd)
                fig_kw1.add_trace(go.Box(
                    y=sub[GDP_pc].values, name=cat,
                    text=sub["country_name"].values,
                    customdata=sub[["_yf"]].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate=t(STRINGS, "kw1_hover"),
                ))
            _pl(fig_kw1,
                title=t(STRINGS, "kw1_title"),
                yaxis_title=t(STRINGS, "gdp_pc_usd"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_kw1, use_container_width=True)
        with col_kw2:
            fig_kw2 = go.Figure()
            df_kw2 = df_m[["country_name", "size_profile", CA_PCT]].dropna()
            df_kw2 = df_kw2[df_kw2[CA_PCT].abs() < 25]
            for cat in _kw_order:
                sub = df_kw2[df_kw2["size_profile"] == cat]
                fig_kw2.add_trace(go.Box(
                    y=sub[CA_PCT].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>CA%=%{y:.1f}%<extra></extra>",
                ))
            fig_kw2.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_kw2,
                title=t(STRINGS, "kw2_title"),
                yaxis_title=t(STRINGS, "ca_pct_gdp"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_kw2, use_container_width=True)
        finding(t(STRINGS, "kw_size_finding"))
        thin_divider()

        # G.1.6 Export Share × Real Interest Rate
        _question(t(STRINGS, "g16_question"))
        col_g16a, col_g16b = st.columns(2)
        with col_g16a:
            df_g16 = df_m[["country_name", RATE, "export_share_structural",
                            "structural_profile"]].dropna()
            df_g16 = df_g16[(df_g16["export_share_structural"] < 0.15)
                             & (df_g16[RATE].abs() < 30)]
            fig_g16 = _scatter(df_g16, RATE, "export_share_structural",
                               "structural_profile", _STRUC_C,
                               t(STRINGS, "g16a_title"),
                               t(STRINGS, "real_rate_pct"), t(STRINGS, "export_share_pct"), height=390)
            st.plotly_chart(fig_g16, use_container_width=True)
        with col_g16b:
            df_g16b = df_m[["country_name", RATE, "export_dep",
                             "structural_profile"]].dropna()
            df_g16b = df_g16b[(df_g16b["export_dep"] > 0)
                               & (df_g16b["export_dep"] < 1.5)
                               & (df_g16b[RATE].abs() < 30)]
            fig_g16b = _scatter(df_g16b, RATE, "export_dep",
                                "structural_profile", _STRUC_C,
                                t(STRINGS, "g16b_title"),
                                t(STRINGS, "real_rate_pct"), t(STRINGS, "export_gdp_ratio"), height=390)
            st.plotly_chart(fig_g16b, use_container_width=True)
        finding(t(STRINGS, "g16_finding"))
        thin_divider()

        # Notes — parties testées sans signal significatif
        with st.expander(t(STRINGS, "other_vars_expander")):
            _note(t(STRINGS, "g13_note"))
            _note(t(STRINGS, "g14_note"))

        # G.1.3 Export/GDP (Trade Openness) vs CA% + Expense%
        _question(t(STRINGS, "g13_question"))
        col_g13a, col_g13b = st.columns(2)
        with col_g13a:
            df_g13 = df_m[["country_name", CA_PCT, "export_dep",
                            "structural_profile"]].dropna()
            df_g13 = df_g13[(df_g13["export_dep"] > 0) & (df_g13["export_dep"] < 1.5)]
            fig_g13 = _scatter(df_g13, CA_PCT, "export_dep", "structural_profile", _STRUC_C,
                               t(STRINGS, "g13a_title"),
                               t(STRINGS, "ca_pct_title"), t(STRINGS, "export_gdp_ratio"), height=450)
            _add_refs(fig_g13, CA_PCT, "export_dep",
                      ["USA", "China", "Brazil"], df=df_g13)
            st.plotly_chart(fig_g13, use_container_width=True)
        with col_g13b:
            df_g13b = df_m[["country_name", EXPENSE, "export_dep",
                             "structural_profile"]].dropna()
            df_g13b = df_g13b[(df_g13b["export_dep"] > 0) & (df_g13b["export_dep"] < 1.5)]
            fig_g13b = _scatter(df_g13b, EXPENSE, "export_dep",
                                "structural_profile", _STRUC_C,
                                t(STRINGS, "g13b_title"),
                                t(STRINGS, "expense_pct_gdp"), t(STRINGS, "export_gdp_ratio"), height=390)
            st.plotly_chart(fig_g13b, use_container_width=True)
        finding(t(STRINGS, "g13_finding"))
        thin_divider()

        # G.1.4 Export/capita vs GDP_pc + Expense%
        _question(t(STRINGS, "g14_question"))
        col_g14a, col_g14b = st.columns(2)
        with col_g14a:
            df_g14 = df_m[["country_name", GDP_pc, "export_pc",
                            "combined_profile"]].dropna()
            df_g14 = df_g14[(df_g14[GDP_pc] > 0) & (df_g14["export_pc"] > 0)]
            fig_g14 = _scatter(df_g14, GDP_pc, "export_pc", "combined_profile", _COMB_C,
                               t(STRINGS, "g14a_title"),
                               t(STRINGS, "gdp_pc_log"), t(STRINGS, "export_pc_log"),
                               xlog=True, ylog=True, height=450, x_fmt="usd", y_fmt="usd")
            _add_refs(fig_g14, GDP_pc, "export_pc",
                      ["Germany", "France", "Sweden", "Norway"], df=df_g14,
                      x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_g14, use_container_width=True)
        with col_g14b:
            df_g14b = df_m[["country_name", EXPENSE, "export_pc",
                             "combined_profile"]].dropna()
            df_g14b = df_g14b[df_g14b["export_pc"] > 0]
            fig_g14b = _scatter(df_g14b, EXPENSE, "export_pc", "combined_profile", _COMB_C,
                                t(STRINGS, "g14b_title"),
                                t(STRINGS, "expense_pct_gdp"), t(STRINGS, "export_pc_log"),
                                ylog=True, height=390, y_fmt="usd")
            st.plotly_chart(fig_g14b, use_container_width=True)
        finding(t(STRINGS, "g14_finding"))
        thin_divider()

        # ── G.2-G.5 Export Dynamics ──────────────────
        section_label(t(STRINGS, "g25_label"))
        st.markdown(
            f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'g25_intro')}</p>",
            unsafe_allow_html=True,
        )

        _question(t(STRINGS, "g2_question"))
        df_g2 = df_m[["country_name", "cagr_export", "cagr_gdp",
                       "size_profile"]].copy()
        df_g2 = df_g2.replace([np.inf, -np.inf], np.nan).dropna()
        df_g2 = df_g2[(df_g2["cagr_gdp"].abs() < 0.25)
                      & (df_g2["cagr_export"].abs() < 0.4)]
        rho_cx = (df_g2[["cagr_gdp", "cagr_export"]]
                  .corr(method="spearman").iloc[0, 1])
        fig_g2 = go.Figure()
        for cat, color in _SIZE_C.items():
            sub = df_g2[df_g2["size_profile"] == cat]
            if sub.empty:
                continue
            fig_g2.add_trace(go.Scatter(
                x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                mode="markers", name=cat,
                marker=dict(color=color, size=7, opacity=0.75),
                text=sub["country_name"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                ),
            ))
        _pl(fig_g2,
            title=t(STRINGS, "g2_chart_title").format(rho=rho_cx),
            xaxis_title=t(STRINGS, "cagr_gdp_pct"), yaxis_title=t(STRINGS, "cagr_export_pct"),
            height=420)
        st.plotly_chart(fig_g2, use_container_width=True)
        finding(t(STRINGS, "g2_finding").format(rho=rho_cx))
        _note(t(STRINGS, "g3_note"))
        thin_divider()

        _question(t(STRINGS, "g4_question"))
        col_g4a, col_g4b = st.columns(2)
        with col_g4a:
            df_g4 = df_m[["country_name", "volatility_export",
                           "size_profile"]].dropna()
            df_g4 = df_g4[(df_g4["volatility_export"] > 0)
                          & (df_g4["volatility_export"] < 80)]
            fig_g4 = go.Figure()
            for cat in ["Leader", "Major", "Mid", "Small"]:
                sub = df_g4[df_g4["size_profile"] == cat]
                fig_g4.add_trace(go.Box(
                    y=sub["volatility_export"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Volatilité=%{y:.1f}%<extra></extra>",
                ))
            _pl(fig_g4,
                title=t(STRINGS, "g4a_title"),
                yaxis_title=t(STRINGS, "export_volatility_pct"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_g4, use_container_width=True)
        with col_g4b:
            df_g4b = df_m[["country_name", RATE, "volatility_export",
                            "size_profile"]].dropna()
            df_g4b = df_g4b[(df_g4b["volatility_export"] > 0)
                             & (df_g4b["volatility_export"] < 80)
                             & (df_g4b[RATE].abs() < 30)]
            _rho_g4b = (df_g4b[[RATE, "volatility_export"]]
                        .corr(method="spearman").iloc[0, 1])
            fig_g4b = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_g4b[df_g4b["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_g4b.add_trace(go.Scatter(
                    x=sub[RATE], y=sub["volatility_export"],
                    mode="markers", name=cat,
                    marker=dict(color=color, size=6, opacity=0.7),
                    text=sub["country_name"],
                    hovertemplate="<b>%{text}</b><br>Real Rate=%{x:.1f}%<br>Volatility=%{y:.1f}%<extra></extra>",
                ))
            _xs_g4b = df_g4b[RATE].values
            _ys_g4b = df_g4b["volatility_export"].values
            _xl_g4b = np.linspace(_xs_g4b.min(), _xs_g4b.max(), 100)
            fig_g4b.add_trace(go.Scatter(
                x=_xl_g4b, y=np.polyval(np.polyfit(_xs_g4b, _ys_g4b, 1), _xl_g4b),
                mode="lines", name="OLS",
                line=dict(color="#60A5FA", dash="dash", width=1.5), hoverinfo="skip",
            ))
            fig_g4b.add_annotation(
                x=0.5, y=0.98, xref="paper", yref="paper",
                text=f"ρ = {_rho_g4b:+.3f} {'✓' if abs(_rho_g4b) > 0.15 else '✗ n.s.'}",
                showarrow=False, xanchor="center", yanchor="top",
                bgcolor="#162035", bordercolor="#304870",
                font=dict(size=12, color="#F59E0B"),
            )
            _pl(fig_g4b,
                title=t(STRINGS, "g4b_title"),
                xaxis_title=t(STRINGS, "real_rate_pct"),
                yaxis_title=t(STRINGS, "export_volatility_pct"),
                height=390,
                legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0,
                            bgcolor="rgba(0,0,0,0)", borderwidth=0))
            st.plotly_chart(fig_g4b, use_container_width=True)
        finding(t(STRINGS, "g4_finding"))
        _note(t(STRINGS, "g5_note"))
        thin_divider()

        # G.7 Synthesis
        section_label(t(STRINGS, "g7_label"))
        finding(t(STRINGS, "g7_finding1"))
        finding(t(STRINGS, "g7_finding2"))

    # ════════════════════════════════════════════════════
    # H · IMPORTS
    # ════════════════════════════════════════════════════
    with subtab_h:

        # ── H.1-H.3 Import Level ─────────────────────
        section_label(t(STRINGS, "h13_label"))
        st.markdown(
            f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'h13_intro')}</p>",
            unsafe_allow_html=True,
        )

        # H.1 Import Share vs GDP
        _question(t(STRINGS, "h1_question"))
        col_h1a, col_h1b = st.columns(2)
        with col_h1a:
            df_h1 = df_m[["country_name", GDP, "import_share_structural",
                           "size_profile"]].dropna()
            df_h1 = df_h1[df_h1[GDP] > 0]
            fig_h1 = _scatter(df_h1, GDP, "import_share_structural", "size_profile", _SIZE_C,
                              t(STRINGS, "h1a_title"),
                              t(STRINGS, "gdp_log"), t(STRINGS, "import_share_pct"),
                              xlog=True, height=450, x_fmt="usd")
            _add_refs(fig_h1, GDP, "import_share_structural",
                      ["USA"], df=df_h1, x_fmt="usd")
            st.plotly_chart(fig_h1, use_container_width=True)
        with col_h1b:
            df_h1b = df_m[["country_name", "export_share_structural",
                            "import_share_structural", "size_profile"]].dropna()
            df_h1b = df_h1b[(df_h1b["export_share_structural"] > 0)
                             & (df_h1b["import_share_structural"] > 0)]
            fig_h1b = _scatter(df_h1b, "export_share_structural", "import_share_structural",
                               "size_profile", _SIZE_C,
                               t(STRINGS, "h1b_title"),
                               t(STRINGS, "export_share_pct"), t(STRINGS, "import_share_pct"), height=450)
            _mv = max(df_h1b["export_share_structural"].max(),
                      df_h1b["import_share_structural"].max())
            fig_h1b.add_trace(go.Scatter(
                x=[0, _mv], y=[0, _mv], mode="lines", name=t(STRINGS, "equilibrium_label"),
                line=dict(color="#6890B0", dash="dot", width=1.5), hoverinfo="skip",
            ))
            _add_refs(fig_h1b, "export_share_structural", "import_share_structural",
                      ["USA", "China", "Germany", "Japan"], df=df_h1b)
            st.plotly_chart(fig_h1b, use_container_width=True)
        finding(t(STRINGS, "h1_finding"))
        thin_divider()

        # H.3 Import/capita — 4 [E] signals
        _question(t(STRINGS, "h3_question"))
        col_h3a, col_h3b = st.columns(2)
        with col_h3a:
            df_h3 = df_m[["country_name", GDP_pc, "import_pc",
                           "combined_profile"]].dropna()
            df_h3 = df_h3[(df_h3[GDP_pc] > 0) & (df_h3["import_pc"] > 0)]
            fig_h3 = _scatter(df_h3, GDP_pc, "import_pc", "combined_profile", _COMB_C,
                              t(STRINGS, "h3a_title"),
                              t(STRINGS, "gdp_pc_log"), t(STRINGS, "import_pc_log"),
                              xlog=True, ylog=True, height=390, x_fmt="usd", y_fmt="usd")
            st.plotly_chart(fig_h3, use_container_width=True)
        with col_h3b:
            tc_h3 = _top_correlates("import_pc", n=6, flag_filter="[E]")
            st.plotly_chart(_corr_bar(tc_h3, t(STRINGS, "h3b_title"),
                                      height=390),
                            use_container_width=True)
        finding(t(STRINGS, "h3_finding"))
        thin_divider()

        # H.2 Import/GDP openness
        _question(t(STRINGS, "h2_question"))
        col_h2a, col_h2b = st.columns(2)
        with col_h2a:
            df_h2 = df_m[["country_name", GDP_pc, "import_dep",
                           "structural_profile"]].dropna()
            df_h2 = df_h2[(df_h2["import_dep"] > 0) & (df_h2["import_dep"] < 1.5)]
            fig_h2 = _scatter(df_h2, GDP_pc, "import_dep", "structural_profile", _STRUC_C,
                              t(STRINGS, "h2a_title"),
                              t(STRINGS, "gdp_pc_log"), t(STRINGS, "import_gdp_ratio"),
                              xlog=True, height=450, x_fmt="usd")
            _add_refs(fig_h2, GDP_pc, "import_dep",
                      ["USA", "Japan"], df=df_h2, x_fmt="usd")
            st.plotly_chart(fig_h2, use_container_width=True)
        with col_h2b:
            df_h2b = df_m[["country_name", EXPENSE, "import_dep",
                            "structural_profile"]].dropna()
            df_h2b = df_h2b[(df_h2b["import_dep"] > 0) & (df_h2b["import_dep"] < 1.5)]
            fig_h2b = _scatter(df_h2b, EXPENSE, "import_dep",
                               "structural_profile", _STRUC_C,
                               t(STRINGS, "h2b_title"),
                               t(STRINGS, "expense_pct_gdp"), t(STRINGS, "import_gdp_ratio"), height=390)
            st.plotly_chart(fig_h2b, use_container_width=True)
        finding(t(STRINGS, "h2_finding"))
        thin_divider()

        # ── H.4-H.7 Import Dynamics ──────────────────
        section_label(t(STRINGS, "h47_label"))

        _question(t(STRINGS, "h4_question"))
        col_h4a, col_h4b = st.columns(2)
        with col_h4a:
            df_h4 = df_m[["country_name", "cagr_import", "cagr_gdp",
                           "size_profile"]].copy()
            df_h4 = df_h4.replace([np.inf, -np.inf], np.nan).dropna()
            df_h4 = df_h4[(df_h4["cagr_gdp"].abs() < 0.25)
                          & (df_h4["cagr_import"].abs() < 0.4)]
            rho_cm = (df_h4[["cagr_gdp", "cagr_import"]]
                      .corr(method="spearman").iloc[0, 1])
            fig_h4 = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_h4[df_h4["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_h4.add_trace(go.Scatter(
                    x=sub["cagr_gdp"] * 100, y=sub["cagr_import"] * 100,
                    mode="markers", name=cat,
                    marker=dict(color=color, size=7, opacity=0.75),
                    text=sub["country_name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "CAGR GDP=%{x:.1f}% | CAGR Import=%{y:.1f}%<extra></extra>"
                    ),
                ))
            _pl(fig_h4,
                title=t(STRINGS, "h4_chart_title").format(rho=rho_cm),
                xaxis_title=t(STRINGS, "cagr_gdp_pct"), yaxis_title=t(STRINGS, "cagr_import_pct"),
                height=450)
            _add_refs(fig_h4, "cagr_gdp", "cagr_import", ["China"],
                      df=df_h4, x_scale=100, y_scale=100,
                      ht="<b>%{text}</b><br>CAGR GDP=%{x:.1f}% | CAGR Import=%{y:.1f}%<extra></extra>")
            st.plotly_chart(fig_h4, use_container_width=True)
        with col_h4b:
            df_h4x = df_m[["country_name", "cagr_export", "cagr_gdp",
                            "size_profile"]].copy()
            df_h4x = df_h4x.replace([np.inf, -np.inf], np.nan).dropna()
            df_h4x = df_h4x[(df_h4x["cagr_gdp"].abs() < 0.25)
                            & (df_h4x["cagr_export"].abs() < 0.4)]
            rho_cx2 = (df_h4x[["cagr_gdp", "cagr_export"]]
                       .corr(method="spearman").iloc[0, 1])
            fig_h4x = go.Figure()
            for cat, color in _SIZE_C.items():
                sub = df_h4x[df_h4x["size_profile"] == cat]
                if sub.empty:
                    continue
                fig_h4x.add_trace(go.Scatter(
                    x=sub["cagr_gdp"] * 100, y=sub["cagr_export"] * 100,
                    mode="markers", name=cat,
                    marker=dict(color=color, size=7, opacity=0.75),
                    text=sub["country_name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "CAGR GDP=%{x:.1f}% | CAGR Export=%{y:.1f}%<extra></extra>"
                    ),
                ))
            _pl(fig_h4x,
                title=t(STRINGS, "h4x_chart_title").format(rho=rho_cx2),
                xaxis_title=t(STRINGS, "cagr_gdp_pct"), yaxis_title=t(STRINGS, "cagr_export_pct"),
                height=390)
            st.plotly_chart(fig_h4x, use_container_width=True)
        finding(t(STRINGS, "h4_finding").format(rho_m=rho_cm, rho_x=rho_cx2))
        thin_divider()

        # H.5 Import Consistency — paradox
        _question(t(STRINGS, "h5_question"))
        col_h5a, col_h5b = st.columns(2)
        with col_h5a:
            df_h5 = df_m[["country_name", "negfreq_import", "size_profile"]].dropna()
            fig_h5 = go.Figure()
            for cat in ["Leader", "Major", "Mid", "Small"]:
                sub = df_h5[df_h5["size_profile"] == cat]
                fig_h5.add_trace(go.Box(
                    y=sub["negfreq_import"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_SIZE_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Neg.Freq Import=%{y:.0%}<extra></extra>",
                ))
            _pl(fig_h5,
                title=t(STRINGS, "h5a_title"),
                yaxis_title=t(STRINGS, "negfreq_import_y"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_h5, use_container_width=True)
        with col_h5b:
            df_h5b = df_m[["country_name", "negfreq_import",
                            "structural_profile"]].dropna()
            fig_h5b = go.Figure()
            for cat in ["Strong", "Good", "Moderate", "Weak"]:
                sub = df_h5b[df_h5b["structural_profile"] == cat]
                if sub.empty:
                    continue
                fig_h5b.add_trace(go.Box(
                    y=sub["negfreq_import"].values, name=cat,
                    text=sub["country_name"].values,
                    marker_color=_STRUC_C.get(cat, "#304870"),
                    boxpoints="outliers",
                    line=dict(width=1.5),
                    hovertemplate="<b>%{text}</b><br>Neg.Freq Import=%{y:.0%}<extra></extra>",
                ))
            _pl(fig_h5b,
                title=t(STRINGS, "h5b_title"),
                yaxis_title=t(STRINGS, "negfreq_import_y"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_h5b, use_container_width=True)
        finding(t(STRINGS, "h5_finding"))
        thin_divider()

        _question(t(STRINGS, "h6_question"))
        col_h6a, col_h6b = st.columns(2)
        with col_h6a:
            df_h6 = df_m[["country_name", INFL, "volatility_import",
                           "size_profile"]].dropna()
            df_h6 = df_h6[(df_h6["volatility_import"] > 0)
                          & (df_h6["volatility_import"] < 80)
                          & (df_h6[INFL].abs() < 30)]
            fig_h6 = _scatter(df_h6, INFL, "volatility_import", "size_profile", _SIZE_C,
                              t(STRINGS, "h6a_title"),
                              t(STRINGS, "inflation_cpi"), t(STRINGS, "import_volatility_pct"), height=390)
            st.plotly_chart(fig_h6, use_container_width=True)
        with col_h6b:
            df_h6b = df_m[["country_name", GDP_pc, "volatility_import",
                            "size_profile"]].dropna()
            df_h6b = df_h6b[(df_h6b["volatility_import"] > 0)
                            & (df_h6b["volatility_import"] < 80)]
            fig_h6b = _scatter(df_h6b, GDP_pc, "volatility_import",
                               "size_profile", _SIZE_C,
                               t(STRINGS, "h6b_title"),
                               t(STRINGS, "gdp_pc_log"), t(STRINGS, "import_volatility_pct"),
                               xlog=True, height=390, x_fmt="usd")
            st.plotly_chart(fig_h6b, use_container_width=True)
        finding(t(STRINGS, "h6_finding"))
        _note(t(STRINGS, "h7_note"))
        thin_divider()

        # H.8 Synthesis
        section_label(t(STRINGS, "h8_label"))
        finding(t(STRINGS, "h8_finding1"))
        finding(t(STRINGS, "h8_finding2").format(rho_m=rho_cm, rho_x=rho_cx2))
        finding(t(STRINGS, "h8_finding3"))

    # ════════════════════════════════════════════════════
    # G+H SYNTHESIS
    # ════════════════════════════════════════════════════
    with subtab_synth:
        section_label(t(STRINGS, "gh_synth_label"))

        # Summary table
        st.markdown(t(STRINGS, "gh_synth_table_html"), unsafe_allow_html=True)

        thin_divider()
        section_label(t(STRINGS, "gh_key_message"))
        finding(t(STRINGS, "gh_key_finding1"))
        finding(t(STRINGS, "gh_key_finding2"))

# ────────────────────────────────────────────────────────────────
# TAB I-J — BALANCE & PROFILS
# ────────────────────────────────────────────────────────────────

with tab_ij:
    _IJ_VARS = {
        "coverage_ratio_structural": "Coverage Ratio (X/M)",
        "balance_dep":               "Balance / GDP (X−M)",
        "trade_intensity":           "Trade Intensity (X+M)/GDP",
    }
    _fig_ij_heat = _tab_heatmap(
        _IJ_VARS,
        t(STRINGS, "ij_heat_title"),
        flags=["[E]"],
    )
    if _fig_ij_heat:
        st.plotly_chart(_fig_ij_heat, use_container_width=True)
    thin_divider()

    subtab_i, subtab_j = st.tabs([t(STRINGS, "subtab_i"), t(STRINGS, "subtab_j")])

    # ════════════════════════════════════════════════════
    # I · BALANCE
    # ════════════════════════════════════════════════════
    with subtab_i:
        section_label(t(STRINGS, "i_label"))
        st.markdown(
            f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'i_intro')}</p>",
            unsafe_allow_html=True,
        )

        # I.1 Coverage Ratio × CA%
        _question(t(STRINGS, "i1_question"))
        col_i1a, col_i1b = st.columns(2)
        with col_i1a:
            df_i1 = df_m[["country_name", CA_PCT, "coverage_ratio_structural",
                           "structural_profile"]].dropna()
            df_i1 = df_i1[(df_i1["coverage_ratio_structural"] < 5)
                           & (df_i1[CA_PCT].abs() < 25)]
            fig_i1 = _scatter(df_i1, CA_PCT, "coverage_ratio_structural",
                              "structural_profile", _STRUC_C,
                              t(STRINGS, "i1a_title"),
                              t(STRINGS, "ca_pct_title"), t(STRINGS, "coverage_ratio_title"),
                              height=390)
            fig_i1.add_hline(y=1, line_color="#6890B0", line_dash="dot", line_width=1,
                             annotation_text=t(STRINGS, "xm_equal_label"), annotation_font_color="#6890B0")
            fig_i1.add_vline(x=0, line_color="#6890B0", line_dash="dot", line_width=1)
            st.plotly_chart(fig_i1, use_container_width=True)
        with col_i1b:
            tc_i1 = _top_correlates("coverage_ratio_structural", n=6, flag_filter="[E]")
            st.plotly_chart(_corr_bar(tc_i1, t(STRINGS, "i1b_title"), height=390),
                            use_container_width=True)
        finding(t(STRINGS, "i1_finding"))
        thin_divider()

        # I.3 Balance/GDP × CA%
        _question(t(STRINGS, "i3_question"))
        col_i3a, col_i3b = st.columns(2)
        with col_i3a:
            df_i3 = df_m[["country_name", CA_PCT, "balance_dep",
                           "structural_profile"]].dropna()
            df_i3 = df_i3[(df_i3["balance_dep"].abs() < 0.8)
                           & (df_i3[CA_PCT].abs() < 25)]
            fig_i3 = _scatter(df_i3, CA_PCT, "balance_dep",
                              "structural_profile", _STRUC_C,
                              t(STRINGS, "i3a_title"),
                              t(STRINGS, "ca_pct_title"), t(STRINGS, "balance_gdp_title"),
                              height=390)
            fig_i3.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            fig_i3.add_vline(x=0, line_color="#6890B0", line_dash="dot", line_width=1)
            st.plotly_chart(fig_i3, use_container_width=True)
        with col_i3b:
            df_i3b = df_m[["country_name", GDP_pc, "balance_dep",
                            "combined_profile"]].dropna()
            df_i3b = df_i3b[(df_i3b[GDP_pc] > 0)
                             & (df_i3b["balance_dep"].abs() < 0.8)]
            _rho_i3b = (df_i3b[[GDP_pc, "balance_dep"]]
                        .corr(method="spearman").iloc[0, 1])
            fig_i3b = _scatter(df_i3b, GDP_pc, "balance_dep",
                               "combined_profile", _COMB_C,
                               t(STRINGS, "i3b_title").format(rho=_rho_i3b),
                               t(STRINGS, "gdp_pc_log"), t(STRINGS, "balance_gdp_title"),
                               xlog=True, height=390, x_fmt="usd")
            fig_i3b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _add_refs(fig_i3b, GDP_pc, "balance_dep",
                      ["USA", "China", "Germany", "Japan", "Brazil"], df=df_i3b,
                      x_fmt="usd")
            st.plotly_chart(fig_i3b, use_container_width=True)
        finding(t(STRINGS, "i3_finding"))
        thin_divider()

        # I.5 Balance Profile × KW
        _question(t(STRINGS, "i5_question"))
        col_i5a, col_i5b = st.columns(2)
        _bal_order = ["Strong Surplus", "Slight Surplus", "Slight Deficit",
                      "Structural Deficit"]
        with col_i5a:
            df_i5 = df_m[["balance_profile", CA_PCT]].dropna()
            df_i5 = df_i5[df_i5[CA_PCT].abs() < 25]
            fig_i5a = _kw_box(CA_PCT, "balance_profile", _bal_order, _BAL_C)
            fig_i5a.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_i5a,
                title=t(STRINGS, "i5a_title"),
                yaxis_title=t(STRINGS, "ca_pct_title"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_i5a, use_container_width=True)
        with col_i5b:
            fig_i5b = _kw_box(GDP_pc, "balance_profile", _bal_order, _BAL_C, fmt_fn=fmt_usd)
            _pl(fig_i5b,
                title=t(STRINGS, "i5b_title"),
                yaxis_title=t(STRINGS, "gdp_pc_usd"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_i5b, use_container_width=True)
        finding(t(STRINGS, "i5_finding"))

        with st.expander(t(STRINGS, "i_other_expander")):
            _note(t(STRINGS, "i2_note"))
            _note(t(STRINGS, "i4_note"))

        thin_divider()
        section_label(t(STRINGS, "i_synth_label"))
        finding(t(STRINGS, "i_synth_finding"))

    # ════════════════════════════════════════════════════
    # J · QUADRANTS & PROFILS
    # ════════════════════════════════════════════════════
    with subtab_j:
        section_label(t(STRINGS, "j_label"))
        st.markdown(
            f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'j_intro')}</p>",
            unsafe_allow_html=True,
        )

        # J.1 Structural Profile × KW
        _question(t(STRINGS, "j1_question"))
        col_j1a, col_j1b = st.columns(2)
        _struc_order = ["Strong", "Good", "Moderate", "Weak"]
        with col_j1a:
            fig_j1a = _kw_box(GDP_pc, "structural_profile", _struc_order, _STRUC_C, fmt_fn=fmt_usd)
            _pl(fig_j1a,
                title=t(STRINGS, "j1a_title"),
                yaxis_title=t(STRINGS, "gdp_pc_usd"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j1a, use_container_width=True)
        with col_j1b:
            fig_j1b = _kw_box(CA_PCT, "structural_profile", _struc_order, _STRUC_C)
            fig_j1b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_j1b,
                title=t(STRINGS, "j1b_title"),
                yaxis_title=t(STRINGS, "ca_pct_title"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j1b, use_container_width=True)
        finding(t(STRINGS, "j1_finding"))
        thin_divider()

        # J.2 Static Quadrant × KW
        _question(t(STRINGS, "j2_question"))
        col_j2a, col_j2b = st.columns(2)
        _static_order = ["Dominant", "Balanced", "Vulnerable", "Marginal"]
        with col_j2a:
            df_j2 = df_m[["static_quadrant", CA_PCT]].dropna()
            df_j2 = df_j2[df_j2[CA_PCT].abs() < 25]
            fig_j2a = _kw_box(CA_PCT, "static_quadrant", _static_order, _STATIC_C)
            fig_j2a.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
            _pl(fig_j2a,
                title=t(STRINGS, "j2a_title"),
                yaxis_title=t(STRINGS, "ca_pct_title"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j2a, use_container_width=True)
        with col_j2b:
            fig_j2b = _kw_box(GDP_pc, "static_quadrant", _static_order, _STATIC_C, fmt_fn=fmt_usd)
            _pl(fig_j2b,
                title=t(STRINGS, "j2b_title"),
                yaxis_title=t(STRINGS, "gdp_pc_usd"),
                height=390,
                xaxis=dict(gridcolor="#304870"),
                yaxis=dict(gridcolor="#304870"))
            st.plotly_chart(fig_j2b, use_container_width=True)
        finding(t(STRINGS, "j2_finding"))
        thin_divider()

        # J.3 Dynamic Quadrant — null (note)
        _question(t(STRINGS, "j3_question"))
        _note(t(STRINGS, "j3_note"))
        thin_divider()

        # J.4 Strategic Quadrant — note
        _note(t(STRINGS, "j4_note"))
        thin_divider()

        # J synthesis
        section_label(t(STRINGS, "j_synth_label"))
        finding(t(STRINGS, "j_synth_finding1"))
        finding(t(STRINGS, "j_synth_finding2"))
        finding(t(STRINGS, "j_synth_finding3"))

# ────────────────────────────────────────────────────────────────
# TAB K — SCORES
# ────────────────────────────────────────────────────────────────
with tab_k:
    section_label(t(STRINGS, "k_label"))
    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'k_intro')}</p>",
        unsafe_allow_html=True,
    )
    _K_VARS = {
        "structural_score": "Structural Score (F)",
        "f_norm_rank":      "F Norm Rank",
        "g_norm_rank":      "G Norm Rank (Bilateral)",
        "combined_score":   "Combined Score",
    }
    _fig_k_heat = _tab_heatmap(
        _K_VARS,
        t(STRINGS, "k_heat_title"),
        flags=["[E]"],
    )
    if _fig_k_heat:
        st.plotly_chart(_fig_k_heat, use_container_width=True)
    thin_divider()

    # K.1 — Bilateral Resilience (null result)
    _question(t(STRINGS, "k1_question"))
    _note(t(STRINGS, "k1_note"))
    thin_divider()

    # K.3 Combined Profile × KW
    _question(t(STRINGS, "k3_question"))
    col_k3a, col_k3b = st.columns(2)
    _comb_order = ["Global Champion", "Competitive", "Vulnerable", "Fragile"]
    with col_k3a:
        fig_k3a = _kw_box(GDP_pc, "combined_profile", _comb_order, _COMB_C, fmt_fn=fmt_usd)
        _pl(fig_k3a,
            title=t(STRINGS, "k3a_title"),
            yaxis_title=t(STRINGS, "gdp_pc_usd"),
            height=390,
            xaxis=dict(gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_k3a, use_container_width=True)
    with col_k3b:
        df_k3b = df_m[["combined_profile", CA_PCT]].dropna()
        df_k3b = df_k3b[df_k3b[CA_PCT].abs() < 25]
        fig_k3b = _kw_box(CA_PCT, "combined_profile", _comb_order, _COMB_C)
        fig_k3b.add_hline(y=0, line_color="#6890B0", line_dash="dot", line_width=1)
        _pl(fig_k3b,
            title=t(STRINGS, "k3b_title"),
            yaxis_title=t(STRINGS, "ca_pct_title"),
            height=390,
            xaxis=dict(gridcolor="#304870"),
            yaxis=dict(gridcolor="#304870"))
        st.plotly_chart(fig_k3b, use_container_width=True)
    finding(t(STRINGS, "k3_finding"))
    thin_divider()

    # K.1/K.2 Combined & Structural Score scatter
    _question(t(STRINGS, "k1k2_question"))
    col_k1a, col_k1b = st.columns(2)
    with col_k1a:
        df_k1 = df_m[["country_name", GDP_pc, "combined_score",
                       "combined_profile"]].dropna()
        df_k1 = df_k1[df_k1[GDP_pc] > 0]
        _x = np.log10(df_k1[GDP_pc].values)
        _y = df_k1["combined_score"].values
        _poly = np.polyfit(_x, _y, 1)
        _x_line = np.linspace(_x.min(), _x.max(), 100)
        _y_line = np.polyval(_poly, _x_line)
        fig_k1 = _scatter(df_k1, GDP_pc, "combined_score",
                          "combined_profile", _COMB_C,
                          t(STRINGS, "k1_chart_title"),
                          t(STRINGS, "gdp_pc_log"), t(STRINGS, "combined_score_title"),
                          xlog=True, height=390, x_fmt="usd")
        fig_k1.add_trace(go.Scatter(
            x=10 ** _x_line, y=_y_line, mode="lines", name="OLS",
            line=dict(color="#60A5FA", dash="dash", width=2),
            hoverinfo="skip",
        ))
        st.plotly_chart(fig_k1, use_container_width=True)
    with col_k1b:
        tc_cs = _top_correlates("combined_score", n=6, flag_filter="[E]")
        tc_ss = _top_correlates("structural_score", n=6, flag_filter="[E]")
        tab_cs, tab_ss = st.tabs([t(STRINGS, "combined_score_title"), t(STRINGS, "structural_score_title")])
        with tab_cs:
            st.plotly_chart(_corr_bar(tc_cs, f"{t(STRINGS, 'combined_score_title')} × [E]", height=280),
                            use_container_width=True)
        with tab_ss:
            st.plotly_chart(_corr_bar(tc_ss, f"{t(STRINGS, 'structural_score_title')} × [E]", height=280),
                            use_container_width=True)
    finding(t(STRINGS, "k1_finding"))
    thin_divider()

    # K hierarchy
    section_label(t(STRINGS, "k_hier_label"))
    st.markdown(
        f"<p style='font-size:13px;color:#7090B0;'>{t(STRINGS, 'k_hier_intro')}</p>",
        unsafe_allow_html=True,
    )
    _e_rank = (df_sp[df_sp["ind_flag"] == "[E]"]
               .groupby("ind_label")["abs_rho"].mean()
               .sort_values(ascending=True))
    _e_colors = ["#10B981" if v > 0.15 else "#F59E0B" if v > 0.08 else "#6890B0"
                 for v in _e_rank]
    fig_hier = go.Figure(go.Bar(
        x=_e_rank.values, y=_e_rank.index,
        orientation="h", marker_color=_e_colors,
        text=_e_rank.values.round(3), textposition="outside",
        hovertemplate="<b>%{y}</b><br>avg |ρ| = %{x:.3f}<extra></extra>",
    ))
    _pl(fig_hier,
        title=t(STRINGS, "k_hier_chart_title"),
        xaxis_title="avg |ρ|", height=340,
        margin=dict(l=160, r=80, t=50, b=30),
        xaxis=dict(gridcolor="#304870", range=[0, 0.5]))
    st.plotly_chart(fig_hier, use_container_width=True)
    finding(t(STRINGS, "k_hier_finding"))

    # K.5 Two dimensions
    thin_divider()
    section_label(t(STRINGS, "k5_label"))
    finding(t(STRINGS, "k5_finding1"))
    finding(t(STRINGS, "k5_finding2"))
    finding(t(STRINGS, "k5_finding3"))

    with st.expander(t(STRINGS, "k_general_expander")):
        finding(t(STRINGS, "k_general_finding1"))
        finding(t(STRINGS, "k_general_finding2"))
        finding(t(STRINGS, "k_general_finding3"))

thin_divider()
