"""
Design system — ETL Portfolio  v4
===================================
Dark-but-readable navy theme
Accent: emerald #10B981
Font: IBM Plex Sans / Mono (loaded via Google Fonts)
Arrow fix: font override targets only text elements, NOT div/span
"""

import streamlit as st

FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
"""

STYLES = """
<style>

/* ── ARROW FIX: target only text elements, never div/span/svg ── */
p, h1, h2, h3, h4, h5, h6, li, td, th, label, a,
[data-testid="stMarkdownContainer"] p { font-family: 'IBM Plex Sans', sans-serif !important; }
code, pre                             { font-family: 'IBM Plex Mono', monospace !important; }
.material-icons                       { font-family: 'Material Icons' !important; }

/* ── App shell ───────────────────────────────────────────────── */
.stApp                                         { background: #162035; }
section[data-testid="stSidebar"]               { background: #1A2840 !important; }
.stMarkdown p                                  { color: #C4D5E8; font-size: 15px; line-height: 1.75; }
h1 { color: #EEF4FB !important; font-size: 2.2rem !important; font-weight: 700 !important; }
h2 { color: #EEF4FB !important; }
h3 { color: #C4D5E8 !important; }
.stCaption                                     { color: #7090B0 !important; font-size: 14px !important; }

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"]                   { color: #A0BAD0 !important; font-size: 14px; }
.stTabs [aria-selected="true"]                 { color: #10B981 !important; }
.stTabs [data-baseweb="tab-highlight"]         { background: #10B981 !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: #10B981; color: #fff; border: none;
    border-radius: 8px; font-size: 14px; font-weight: 500;
    padding: 0.5rem 1.25rem; transition: all .2s;
}
.stButton > button:hover                       { background: #059669; }

/* ── Form elements ───────────────────────────────────────────── */
.stSelectbox label, .stNumberInput label       { color: #A0BAD0 !important; font-size: 13px !important; }
.stProgress > div > div                        { background: #10B981 !important; }
.stDataFrame                                   { border-radius: 8px; overflow: hidden; }

/* ── Multiselect — show full indicator names in tags ─────────── */
[data-baseweb="tag"]                           { height: auto !important; white-space: normal !important; }
[data-baseweb="tag"] span:first-child          { white-space: normal !important; overflow: visible !important;
                                                 text-overflow: unset !important; max-width: none !important; }

/* ── Expanders ───────────────────────────────────────────────── */
[data-testid="stExpander"]                     { border: 1px solid #304870 !important; border-radius: 10px !important; background: #1A2840 !important; }
[data-testid="stExpander"] summary             { color: #8AABB0 !important; font-size: 14px !important; }
[data-testid="stExpander"] summary:hover       { color: #10B981 !important; }

/* ── Alerts ──────────────────────────────────────────────────── */
div[data-testid="stSuccess"] { background: #072F1E !important; border-left: 4px solid #10B981 !important; }
div[data-testid="stWarning"] { background: #231A07 !important; border-left: 4px solid #F59E0B !important; }
div[data-testid="stError"]   { background: #270C0C !important; border-left: 4px solid #EF4444 !important; }
div[data-testid="stInfo"]    { background: #0D2040 !important; border-left: 4px solid #3B82F6 !important; }

/* ── Page eyebrow ────────────────────────────────────────────── */
.page-eyebrow {
    font-size: 12px; font-weight: 600; color: #10B981;
    text-transform: uppercase; letter-spacing: .10em; margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Thin divider ────────────────────────────────────────────── */
.thin-div { border: none; border-top: 1px solid #304870; margin: 1.25rem 0; }

/* ── Section label ───────────────────────────────────────────── */
.section-label {
    font-size: 15px; font-weight: 600; color: #A8C0D8;
    letter-spacing: .02em;
    margin: 1.75rem 0 0.6rem 0; padding-bottom: 8px;
    border-bottom: 2px solid #10B981;
    display: inline-block;
}

/* ── KPI card ────────────────────────────────────────────────── */
.kpi-card {
    background: #1F3050; border: 1px solid #304870;
    border-top: 3px solid #304870; border-radius: 12px;
    padding: 1rem 1.25rem 0.9rem; text-align: center; transition: border-top-color .2s;
}
.kpi-card.teal  { border-top-color: #10B981; }
.kpi-card.blue  { border-top-color: #3B82F6; }
.kpi-card.amber { border-top-color: #F59E0B; }
.kpi-label { font-size: 11px; font-weight: 600; color: #6890B0; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.kpi-value { font-size: 30px; font-weight: 700; color: #EEF4FB; line-height: 1.1; }
.kpi-value.teal  { color: #10B981; }
.kpi-value.blue  { color: #60A5FA; }
.kpi-value.amber { color: #F59E0B; }

/* ── Challenge / Solution cards ─────────────────────────────── */
.challenge-box {
    background: #1F3050; border: 1px solid #304870;
    border-left: 3px solid #F59E0B; border-radius: 0 10px 10px 0;
    padding: 1rem 1.1rem; height: 100%;
}
.solution-box {
    background: #1F3050; border: 1px solid #304870;
    border-left: 3px solid #10B981; border-radius: 0 10px 10px 0;
    padding: 1rem 1.1rem; height: 100%;
}
.box-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 8px; }
.challenge-box .box-title { color: #F59E0B; }
.solution-box  .box-title { color: #10B981; }
.box-body { font-size: 15px; color: #A8C0D8; line-height: 1.8; }
.box-body strong { color: #D8ECF8; }
.box-body em     { color: #8AABB0; }
.box-body code   { background: #253B5C; color: #93C5FD; padding: 1px 6px; border-radius: 4px; font-size: 13px; }

/* ── Decision block ─────────────────────────────────────────── */
.decision-block {
    background: #1A2C48; border: 1px solid #304870;
    border-left: 3px solid #818CF8; border-radius: 0 10px 10px 0;
    padding: 0.75rem 1.1rem; margin-top: 8px;
    font-size: 14px; color: #A8C0D8; line-height: 1.75;
}
.decision-block strong { color: #C7D2FE; }
.decision-block code   { background: #253B5C; color: #93C5FD; padding: 1px 5px; border-radius: 4px; font-size: 13px; }

/* ── Finding block ───────────────────────────────────────────── */
.finding-block {
    background: #1F3050; border: 1px solid #304870;
    border-left: 3px solid #10B981; border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem; margin-bottom: 6px;
    font-size: 15px; color: #A8C0D8; line-height: 1.8;
}
.finding-block strong { color: #D8ECF8; }
.finding-block code   { background: #253B5C; color: #93C5FD; padding: 1px 5px; border-radius: 4px; font-size: 13px; }

/* ── Badge ──────────────────────────────────────────────────── */
.badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 99px; font-family: 'IBM Plex Mono', monospace; }
.badge-done    { background: #083D25; color: #34D399; border: 1px solid #0D6B40; }
.badge-partial { background: #2A1F07; color: #FCD34D; border: 1px solid #92400E; }
.badge-failed  { background: #2D0C0C; color: #FCA5A5; border: 1px solid #991B1B; }
.badge-pending { background: #1F3050; color: #6890B0; border: 1px solid #304870; }
.badge-active  { background: #083D25; color: #34D399; border: 1px solid #0D6B40; }
.badge-planned { background: #2A1F07; color: #FCD34D; border: 1px solid #92400E; }
.badge-ext     { background: #0F2240; color: #93C5FD; border: 1px solid #1E4080; }
.badge-corr    { background: #1E1240; color: #C4B5FD; border: 1px solid #5B21B6; }

/* ── Dots ────────────────────────────────────────────────────── */
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot-done    { background: #10B981; box-shadow: 0 0 6px #10B98166; }
.dot-partial { background: #F59E0B; box-shadow: 0 0 6px #F59E0B66; }
.dot-failed  { background: #EF4444; box-shadow: 0 0 6px #EF444466; }
.dot-pending { background: #3A5070; }

/* ── Status rows ─────────────────────────────────────────────── */
.status-row {
    display: flex; align-items: center; gap: 12px; padding: 9px 14px;
    background: #1F3050; border: 1px solid #304870; border-radius: 8px;
    margin-bottom: 4px; font-size: 14px; transition: border-color .15s;
}
.status-row:hover { border-color: #4A6890; }
.stage-row {
    display: flex; align-items: center; gap: 12px; padding: 7px 14px 7px 28px;
    background: #1A2840; border-radius: 6px; margin-bottom: 3px; font-size: 14px;
}

/* ── Checkpoint line ─────────────────────────────────────────── */
.ck-line { display: flex; align-items: center; gap: 10px; padding: 4px 14px 4px 42px; font-size: 13px; color: #8AABB0; font-family: 'IBM Plex Mono', monospace; }
.ck-ok   { color: #34D399; }
.ck-fail { color: #FCA5A5; }
.ck-warn { color: #FCD34D; }

/* ── Backend row ─────────────────────────────────────────────── */
.backend-row {
    display: flex; align-items: center; gap: 14px; padding: 11px 16px;
    background: #1F3050; border: 1px solid #304870; border-radius: 10px;
    margin-bottom: 6px; font-size: 14px; transition: border-color .15s;
}
.backend-row:hover { border-color: #4A6890; }
.backend-name { font-weight: 600; color: #EEF4FB; min-width: 110px; }
.backend-desc { color: #8AABB0; flex: 1; }

/* ── Medallion cards ─────────────────────────────────────────── */
.med-card { border-radius: 14px; padding: 1.1rem 1rem; height: 100%; }
.med-bronze { background: #231800; border: 1.5px solid #D97706; }
.med-silver { background: #0F2240; border: 1.5px solid #3B82F6; }
.med-gold   { background: #082E18; border: 1.5px solid #10B981; }
.med-title  { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.med-bronze .med-title { color: #FCD34D; }
.med-silver .med-title { color: #93C5FD; }
.med-gold   .med-title { color: #34D399; }
.med-files  { font-size: 13px; line-height: 1.9; font-family: 'IBM Plex Mono', monospace; }
.med-bronze .med-files { color: #FBBf24; }
.med-silver .med-files { color: #BAD7F8; }
.med-gold   .med-files { color: #6EE7B7; }
.med-tag    { opacity: .5; font-size: 12px; }
.med-footer { margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.08); font-size: 12px; line-height: 1.7; }
.med-bronze .med-footer { color: #FCD34D; opacity: .8; }
.med-silver .med-footer { color: #93C5FD; opacity: .8; }
.med-gold   .med-footer { color: #34D399; opacity: .8; }

</style>
"""


def inject_styles() -> None:
    st.markdown(FONTS + STYLES, unsafe_allow_html=True)


def kpi(label: str, value: str, color: str = "") -> str:
    cls     = f"kpi-card {color}".strip()
    val_cls = f"kpi-value {color}".strip()
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="{val_cls}">{value}</div></div>'


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def thin_divider() -> None:
    st.markdown('<hr class="thin-div">', unsafe_allow_html=True)


def challenge_solution(challenge: str, solution: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="challenge-box"><div class="box-title">The challenge</div>'
            f'<div class="box-body">{challenge}</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="solution-box"><div class="box-title">The solution</div>'
            f'<div class="box-body">{solution}</div></div>',
            unsafe_allow_html=True)


def decision(text: str) -> None:
    st.markdown(
        f'<div class="decision-block">⚡ <strong>Key decision</strong> — {text}</div>',
        unsafe_allow_html=True)


def finding(text: str) -> None:
    st.markdown(f'<div class="finding-block">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "done") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def dot(status: str) -> str:
    return {"done": "done", "partial": "partial", "partial_errors": "partial",
            "failed": "failed", "running": "partial"}.get(status, "pending")


def format_value(x) -> str:
    """Format large numbers with T/B/M/K suffixes."""
    try:
        x = float(x)
        if abs(x) >= 1e12: return f"{x/1e12:.2f}T"
        if abs(x) >= 1e9:  return f"{x/1e9:.2f}B"
        if abs(x) >= 1e6:  return f"{x/1e6:.1f}M"
        if abs(x) >= 1e3:  return f"{x/1e3:.1f}K"
        return f"{x:.2f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_usd(v) -> str:
    """Format dollar value with $ prefix and K/B/T suffix for hover display."""
    try:
        v = float(v)
        if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
        if abs(v) >= 1e3:  return f"${v/1e3:.1f}K"
        return f"${v:.2f}"
    except (TypeError, ValueError):
        return "n/a"