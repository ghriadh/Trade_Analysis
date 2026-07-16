"""
Page 1 — Pipeline & Extraction  (v4)
Layout fixes + all accumulated corrections:
- Bronze/Silver/Gold horizontal layout (6/2/2)
- Arrow removed from expander labels
- min-height on sub-boxes
- st.container() for demo sections
- Status limited to ~10 lines + Show all expander
- Order: KPIs → Pipeline Flow → Challenges → Status → Demo → Run
- 16 indicators (not 17)
- Code snippets added to challenges B and C
- Font: body 15px, labels 12px
- Contrast fixes
"""

import os
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from utils.styles import (
    inject_styles, kpi, section_label, thin_divider,
    challenge_solution, decision, finding, dot
)
from utils.etl_status_reader import load_etl_status, ETLStatus

inject_styles()

etl: ETLStatus = load_etl_status()

STAGE_LABELS = {
    "countries":         "Countries",
    "products":          "Products",
    "trade_flow":        "Trade flows",
    "indicators":        "Indicators",
    "indicators_values": "Indicator values",
    "trade_data":        "Trade data",
}


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="page-eyebrow">ETL · Page 1 of 2</div>', unsafe_allow_html=True)
st.title("Pipeline & Extraction")
st.caption(
    "Async API extraction · thread pool · rate-limit handling · "
    "checkpointing · missing country detection · resumable runs"
)
thin_divider()


# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════

bronze = etl.pipelines.get("bronze")
silver = etl.pipelines.get("silver")
gold   = etl.pipelines.get("gold")


def _fmt(pipeline, exclude_zero: bool = False) -> str:
    if not pipeline:
        return "—"
    rows = sum(
        s.rows for s in pipeline.stages
        if not (exclude_zero and s.rows == 0)
    )
    if rows == 0:
        return "—"
    return (f"{rows/1_000_000:.1f}M" if rows >= 1_000_000
            else f"{rows/1_000:.0f}K" if rows >= 1_000 else str(rows))


last_run = (etl.last_updated or "Not run yet").replace(" UTC", "")

cols = st.columns(4)
cols[0].markdown(kpi("Bronze rows", _fmt(bronze, exclude_zero=True), "amber"), unsafe_allow_html=True)
cols[1].markdown(kpi("Silver rows", _fmt(silver), "blue"),  unsafe_allow_html=True)
cols[2].markdown(kpi("Gold rows",   _fmt(gold),   "teal"),  unsafe_allow_html=True)
cols[3].markdown(kpi("Last run",    last_run, ""),           unsafe_allow_html=True)

thin_divider()


# ════════════════════════════════════════════════════════════════
# PIPELINE FLOW — Bronze large + Silver/Gold stacked on right
# ════════════════════════════════════════════════════════════════

section_label("Pipeline Flow")

col_bronze, col_silver, col_gold = st.columns([6, 2, 2])

# ── Bronze ────────────────────────────────────────────────────
with col_bronze:
    st.markdown("""
    <div style="background:#111827; border:1px solid #92400E55;
                border-radius:14px; padding:.85rem 1rem 1rem 1rem;">
      <div style="font-size:11px; font-weight:700; color:#F59E0B;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        🟡 Bronze Layer — Raw Extraction
      </div>
    </div>
    """, unsafe_allow_html=True)

    def _sub(icon, title, lines, footer):
        items = "".join(f"<div style='margin-bottom:2px;'>· {l}</div>" for l in lines)
        return f"""
        <div style="background:#1A2840; border:1px solid #2C3F58;
                    border-radius:10px; padding:.8rem .9rem;
                    height:160px; display:flex; flex-direction:column;">
          <div style="font-size:12px; font-weight:600; color:#FCD34D;
                      margin-bottom:8px; flex-shrink:0;">{icon} {title}</div>
          <div style="font-size:13px; color:#A8C0D8; line-height:1.75; flex:1;">{items}</div>
          <div style="margin-top:6px; font-size:11px; color:#D97706;
                      font-family:'IBM Plex Mono',monospace; flex-shrink:0;">{footer}</div>
        </div>"""

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_sub("🌐", "APIs",
        ["UN Comtrade", "World Bank API"],
        "250 reporters · 16 indicators"), unsafe_allow_html=True)

    s2.markdown(_sub("⚡", "Async + Threads",
        ["asyncio event loop", "asyncio.to_thread()", "Configurable workers"],
        "~5,000 safe calls"), unsafe_allow_html=True)

    s3.markdown(_sub("📍", "Checkpoints",
        [".done per (reporter,flow)", "Resume after crash", "Per-indicator WB"],
        "Zero re-work on re-run"), unsafe_allow_html=True)

    s4.markdown(_sub("🔍", "Validation",
        ["failed/empty/with_data", "Code divergence", "Auto re-run cmd"],
        "Partitioned CSV + DB"), unsafe_allow_html=True)

# ── Silver ────────────────────────────────────────────────────
with col_silver:
    st.markdown("""
    <div style="background:#0F2240; border:1px solid #3B82F655;
                border-radius:14px; padding:.85rem .9rem;
                height:100%; min-height:260px;">
      <div style="font-size:11px; font-weight:700; color:#93C5FD;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        🔵 Silver
      </div>
      <div style="font-size:13px; color:#93C5FD; line-height:1.9; opacity:.85;">
        <div>· 6 transform modules</div>
        <div>· Null checks</div>
        <div>· Deduplication</div>
        <div>· Type casting</div>
      </div>
      <div style="margin-top:10px; font-size:11px; color:#60A5FA;
                  font-family:'IBM Plex Mono',monospace; opacity:.8;">
        Module registry<br>→ targeted re-runs
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Gold ──────────────────────────────────────────────────────
with col_gold:
    st.markdown("""
    <div style="background:#082E18; border:1px solid #10B98155;
                border-radius:14px; padding:.85rem .9rem;
                height:100%; min-height:260px;">
      <div style="font-size:11px; font-weight:700; color:#34D399;
                  text-transform:uppercase; letter-spacing:.09em;
                  margin-bottom:.7rem; font-family:'IBM Plex Mono',monospace;">
        🟢 Gold
      </div>
      <div style="font-size:13px; color:#6EE7B7; line-height:1.9; opacity:.85;">
        <div>· 4 dimension tables</div>
        <div>· 2 fact tables</div>
        <div>· Surrogate keys</div>
        <div>· Dependency checker</div>
        <div>· Divergence detection</div>
      </div>
      <div style="margin-top:10px; font-size:11px; color:#34D399;
                  font-family:'IBM Plex Mono',monospace; opacity:.8;">
        316 entities
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# A — API CALL MANAGEMENT
# ════════════════════════════════════════════════════════════════

section_label("A · Managing API Calls — Comtrade")

challenge_solution(
    challenge="""
Comtrade requires one API call per <strong>(reporter × flow × year)</strong> combination.
Starting from <strong>250 countries</strong>, 2 flows, and 10 years that's
<strong>~5,000 calls</strong> per product. Sequential execution would take
days and guarantee hitting API quota bans
(<em>"Quota will be replenished in 00:36:18"</em>).
""",
    solution="""
<strong>Async event loop + thread pool</strong> via <code>asyncio.to_thread()</code>:
each synchronous API call runs in a dedicated thread while the async event loop
coordinates concurrency. Workers are configurable via <code>--workers</code>.
When the API returns <strong>429 (rate limit)</strong>, only that worker pauses —
others continue. A 1-second courtesy pause avoids triggering bans.
"""
)

decision(
    "Use <code>asyncio.to_thread()</code> to run blocking API calls in a thread pool — "
    "combines async coordination (event loop) with true parallelism (threads), "
    "without rewriting the synchronous Comtrade SDK"
)

with st.expander("See the async + thread pool code"):
    st.code("""
async def run_task(semaphore, flow, year, reporter, product, partition_dir):
    if is_done(partition_dir, reporter, flow):   # skip checkpointed
        return None

    async with semaphore:
        # asyncio.to_thread() dispatches to ThreadPoolExecutor
        df, status = await asyncio.to_thread(
            fetch_comtrade, flow, year, reporter, product
        )
        await asyncio.sleep(1)             # courtesy pause — non-blocking

        for delay in RETRY_DELAYS:         # [10, 20] seconds
            if status != 429:
                break
            await asyncio.sleep(delay)     # only THIS thread waits
            df, status = await asyncio.to_thread(fetch_comtrade, ...)

        if df is None:
            return {"reporter": reporter, "flow": flow, "status": "failed"}
        if df.empty:
            return {"reporter": reporter, "flow": flow, "status": "empty"}

        save_reporter(df, partition_dir, reporter, flow)
        return {"reporter": reporter, "flow": flow, "status": "with_data"}

# Bounded concurrency — configurable via --workers
semaphore = asyncio.Semaphore(workers)
results   = await asyncio.gather(*tasks)
    """, language="python")


# ════════════════════════════════════════════════════════════════
# B — CHECKPOINT & RESUMABILITY
# ════════════════════════════════════════════════════════════════

section_label("B · Checkpoint & Resumability")

challenge_solution(
    challenge="""
With <strong>~5,000 API calls</strong> split across multiple sessions
(API quota: <em>"Quota will be replenished in 00:36:18"</em>),
a crash or quota ban mid-run means losing all progress.
The pipeline must resume from exactly where it stopped —
across sessions, days, or machines.
""",
    solution="""
Each successful <strong>(reporter, flow)</strong> pair writes a lightweight
<code>.done</code> marker file. On re-run, any pair with a marker is silently skipped.
<strong>Empty responses</strong> (countries with no trade data) are tracked separately
but <em>not</em> marked done — they remain eligible for future re-extraction.
<code>failed_reporters.json</code> captures only true API errors for targeted re-run.
"""
)

decision(
    "Three distinct outcomes per call: "
    "<code>with_data</code> (saved + marked done), "
    "<code>empty</code> (valid — not marked done, re-eligible), "
    "<code>failed</code> (API error — in failed_reporters.json). "
    "Only failures need a re-run"
)

with st.expander("See the checkpoint logic"):
    st.code("""
# Partition layout
data/bronze/trade_data/
    year=2022/product=TOTAL/flows=M_X/
        134_M.csv        ← data for reporter 134, import
        134_M.done       ← marker: extracted with data
        failed_reporters.json  ← true API failures only
        .checkpoints/    ← marker files directory

def is_done(partition_dir, reporter, flow) -> bool:
    return (partition_dir / ".checkpoints" / f"{reporter}_{flow}.done").exists()

def mark_done(partition_dir, reporter, flow) -> None:
    path = partition_dir / ".checkpoints" / f"{reporter}_{flow}.done"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

# On re-run — only pending pairs are fetched
all_pairs = [(f, r) for f in flows for r in reporters]
pending   = [(f, r) for f, r in all_pairs
             if not is_done(partition_dir, r, f)]
logger.info("Skipping %d already-done pairs", len(all_pairs) - len(pending))
    """, language="python")


# ════════════════════════════════════════════════════════════════
# C — MISSING COUNTRY DETECTION
# ════════════════════════════════════════════════════════════════

section_label("C · Missing Country Detection & Correction")

challenge_solution(
    challenge="""
Some countries use a <strong>different code in Comtrade</strong> vs the ISO standard
used by World Bank. A standard extraction using ISO codes returns no data
for these countries — they appear as <em>silently missing</em> with no error raised.
Additional runs may also produce new API failures requiring targeted correction.
""",
    solution="""
The Gold pipeline runs a <strong>code divergence check</strong> comparing reporter codes
between World Bank and Comtrade. Divergent countries are flagged and an
<strong>exact re-extraction command is auto-generated</strong>.
API failures produce a <code>failed_reporters.json</code> with the same correction pattern.
The <strong>module registry</strong> then allows only the affected Silver/Gold modules
to re-run — not the full pipeline.
"""
)

decision(
    "Two failure modes, two corrections — both auto-generate the exact CLI command needed: "
    "<strong>divergent codes</strong> → re-extract with Comtrade-specific codes; "
    "<strong>API failures</strong> → re-extract from <code>failed_reporters.json</code>. "
    "After correction: <code>run_silver --module trade_data</code> + "
    "<code>run_gold --module trade_data</code>"
)

with st.expander("See the divergence detection & correction flow"):
    st.code("""
# Gold — detect code divergence after merging WB and Comtrade
divergent = df_merged[df_merged["wb_code"] != df_merged["comtrade_code"]]

if not divergent.empty:
    codes = " ".join(divergent["country_code"].astype(int).astype(str).tolist())
    print(f"WARNING: {len(divergent)} countries with divergent codes")
    print(f"  → python -m src.bronze.extract_trade_data \\")
    print(f"         --year 2022 --product TOTAL --flows M X \\")
    print(f"         --reporters {codes}")

# API failures → re-run with failed_reporters.json
#   python -m src.bronze.extract_trade_data --reporters 134 276 512

# After any correction — only re-run affected modules
#   python -m src.pipelines.run_silver --module trade_data
#   python -m src.pipelines.run_gold   --module trade_data
    """, language="python")


# ════════════════════════════════════════════════════════════════
# D — WORLD BANK PAGINATION
# ════════════════════════════════════════════════════════════════

section_label("D · World Bank — Pagination & Resumability")

challenge_solution(
    challenge="""
The World Bank API paginates results, but the <strong>total number of pages is unknown
before the first call</strong>. A naive implementation either fetches only page 1
(incomplete data) or over-fetches with unnecessary calls.
With <strong>16 indicators</strong> to extract, a mid-run failure
means re-fetching already-completed indicators from scratch.
""",
    solution="""
The first API call reads <code>data[0]["pages"]</code> to discover the page count,
then loops through the remaining pages automatically — no hardcoded limit.
Each indicator is saved to a <strong>per-indicator checkpoint</strong> on success,
giving the same crash-resilience as Comtrade.
A <strong>session with automatic HTTP retry</strong> handles transient server
errors (5xx) with exponential backoff.
"""
)

decision(
    "Discover pagination dynamically from the first response — no hardcoded page count. "
    "Per-indicator checkpoints make re-runs incremental: "
    "already-extracted indicators are skipped with no redundant API calls"
)

with st.expander("See the World Bank pagination + retry logic"):
    st.code("""
# HTTP session — automatic retry on 5xx errors
session = requests.Session()
retry   = Retry(total=3, backoff_factor=2,      # 2s → 4s → 8s
                status_forcelist=[500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retry))

def extract_indicator(session, indicator, start_year, end_year):
    # First call — discover total page count from response metadata
    df_all, pages = fetch_page(session, indicator, start_year, end_year, page=1)

    for page in range(2, pages + 1):             # loop only known pages
        time.sleep(1)
        df_page, _ = fetch_page(session, indicator, start_year, end_year, page)
        df_all = pd.concat([df_all, df_page])
    return df_all

# Per-indicator checkpoint — skip already-done on re-run
for indicator in all_indicators:
    if is_done(indicator):
        df_final = pd.concat([df_final, load_from_checkpoint(indicator)])
        continue
    df = extract_indicator(session, indicator, start_year, end_year)
    mark_done(indicator, df)
    df_final = pd.concat([df_final, df])
    """, language="python")


# ════════════════════════════════════════════════════════════════
# PIPELINE RUN STATUS
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label("Pipeline Run Status")

if not etl.has_data:
    st.info("Pipeline has not been run yet — metrics will appear after the first run.")
else:
    for pipeline_name, pipeline in etl.pipelines.items():
        d       = dot(pipeline.status)
        bdg_cls = {"done": "done", "partial": "partial", "partial_errors": "partial",
                   "failed": "failed", "running": "partial"}.get(pipeline.status, "pending")

        st.markdown(f"""
        <div class="status-row">
          <div class="dot dot-{d}"></div>
          <span style="font-weight:600;font-size:15px;color:#F1F5F9;">{pipeline.label}</span>
          <span class="badge badge-{bdg_cls}">{pipeline.status}</span>
          {"<span style='margin-left:auto;font-size:13px;color:#8AABB0;"
           "font-family:IBM Plex Mono,monospace;'>"
           + pipeline.total_duration_fmt + " total</span>"
           if pipeline.total_duration_s > 0 else ""}
        </div>""", unsafe_allow_html=True)

        for stage in pipeline.stages:
            ds  = dot(stage.status)
            lbl = STAGE_LABELS.get(stage.name, stage.label)
            st.markdown(f"""
            <div class="stage-row">
              <div class="dot dot-{ds}" style="width:7px;height:7px;"></div>
              <span style="min-width:170px;color:#C4D5E8;">{lbl}</span>
              <span style="color:#8AABB0;min-width:90px;text-align:right;
                           font-family:'IBM Plex Mono',monospace;">{stage.rows_fmt} rows</span>
              <span style="color:#6890B0;min-width:70px;text-align:right;
                           font-family:'IBM Plex Mono',monospace;">{stage.duration_fmt}</span>
              {"<span style='color:#EF4444;margin-left:10px;font-size:12px;'>"
               + (stage.error or "") + "</span>" if stage.error else ""}
            </div>""", unsafe_allow_html=True)

        # Checkpoints — limited display
        if pipeline_name == "bronze":
            for stage in pipeline.stages:
                if not stage.checkpoints:
                    continue

                if stage.name == "trade_data":
                    runs = {k: v for k, v in stage.checkpoints.items()
                            if isinstance(v, dict) and "total" in v}
                    if not runs:
                        continue

                    sorted_runs = sorted(runs.items())
                    preview     = sorted_runs[:10]
                    remainder   = sorted_runs[10:]

                    def _ck_row(run_key, counts):
                        partition  = run_key.split("#")[0]
                        timestamp  = run_key.split("#")[1] if "#" in run_key else ""
                        with_data_n = counts.get("with_data", counts.get("done", 0))
                        empty_n    = counts.get("empty", 0)
                        done_n     = with_data_n + empty_n
                        total_n    = counts.get("total", 0)
                        failed_n   = counts.get("failed", 0)
                        run_type   = counts.get("type", "standard")
                        color      = "#10B981" if failed_n == 0 else "#F59E0B"
                        bdg        = "corr" if run_type == "correction" else "pending"
                        return f"""
                        <div class="ck-line">
                          <span style="min-width:180px;color:#8AABB0;">{partition}</span>
                          <span style="color:{color};font-weight:500;">{done_n}/{total_n} reporters</span>
                          {"<span class='ck-fail'>&nbsp;· " + str(failed_n) + " failed</span>" if failed_n else ""}
                          {"<span style='color:#8AABB0;'>&nbsp;· " + str(empty_n) + " empty</span>" if empty_n else ""}
                          {"<span style='color:#64748B;margin-left:6px;'>" + timestamp + "</span>" if timestamp else ""}
                          <span class="badge badge-{bdg}" style="margin-left:6px;">{run_type}</span>
                        </div>"""

                    for rk, rc in preview:
                        st.markdown(_ck_row(rk, rc), unsafe_allow_html=True)

                    if remainder:
                        with st.expander(f"Show {len(remainder)} more checkpoint(s)"):
                            for rk, rc in remainder:
                                st.markdown(_ck_row(rk, rc), unsafe_allow_html=True)

                elif stage.name == "indicators_values":
                    ck  = stage.checkpoints
                    pct = ck.get("progress_pct", 0)
                    st.progress(pct / 100)
                    failed_n = ck.get("failed", 0)
                    st.markdown(
                        f'<div class="ck-line">'
                        f'{ck.get("done",0)}/{ck.get("total",0)} indicators · {pct}%'
                        + (f'<span class="ck-fail"> · {failed_n} failed</span>' if failed_n else "")
                        + "</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LIVE API DEMO
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label("Live API Demo")
st.caption("Single-call extractions — real-time API connectivity, no pipeline run required.")

tab1, tab2 = st.tabs(["🌍  Comtrade — Trade Data", "📈  World Bank — Indicator"])

with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # Priority: gold entities (Comtrade codes) > bronze countries (ISO codes)
        gold_ent = ROOT / "data" / "gold" / "entities" / "entities.csv"
        bronze_c = ROOT / "data" / "bronze" / "countries" / "countries.csv"
        src_path = gold_ent if gold_ent.exists() else bronze_c

        if src_path.exists():
            df_c    = pd.read_csv(src_path)
            # Find best name/code columns available
            name_c  = next((c for c in ["country_name","reporter_desc","name"]        if c in df_c.columns), None)
            code_c  = next((c for c in ["reporter_code","comtrade_code","country_code"] if c in df_c.columns), None)
            if name_c and code_c:
                opts = (df_c[[name_c, code_c]].dropna()
                        .assign(label=lambda d: d[name_c] + " (" + d[code_c].astype(int).astype(str) + ")")
                        .sort_values(name_c))
                default = next((l for l in opts["label"].tolist() if "France" in l), None)
                sel = st.selectbox("Reporter country (Comtrade code)", opts["label"].tolist(),
                                   index=opts["label"].tolist().index(default) if default else 0,
                                   key="trade_country")
                rc = int(opts.loc[opts["label"] == sel, code_c].iloc[0])
            else:
                rc = st.number_input("Reporter code (Comtrade)", value=251, step=1)
        else:
            rc = st.number_input("Reporter code (Comtrade)", value=251, step=1)
    with c2:
        year = st.selectbox("Year", list(range(2024, 2014, -1)), index=0, key="trade_year")
    with c3:
        flow      = st.selectbox("Flow", ["M (Imports)", "X (Exports)"], index=0, key="trade_flow")
        flow_code = flow.split()[0]

    if st.button("▶  Extract trade data", key="btn_trade"):
        try:
            import comtradeapicall
            with st.spinner(f"Calling Comtrade API — reporter {rc}, {year}…"):
                t0     = time.time()
                df_out = comtradeapicall.previewFinalData(
                    typeCode="C", freqCode="A", clCode="HS",
                    period=str(year), reporterCode=str(rc),
                    cmdCode="TOTAL", flowCode=flow_code,
                    partnerCode=None, partner2Code=0,
                    customsCode=None, motCode=0,
                    maxRecords=500, format_output="JSON", includeDesc=True,
                )
                elapsed = round(time.time() - t0, 2)
            if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                st.success(f"✅  {len(df_out)} rows fetched in {elapsed}s")
                show = [c for c in ["reporterDesc", "partnerDesc", "flowCode",
                                    "cmdCode", "primaryValue"] if c in df_out.columns]
                st.dataframe(df_out[show].head(10), width='stretch')
            else:
                st.warning("No data returned — try a different country or year.")
        except ImportError:
            st.error("comtradeapicall not installed.")
        except Exception as e:
            st.error(f"API error: {e}")

with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        ip = ROOT / "data" / "resources" / "indicators.csv"
        if ip.exists():
            df_i    = pd.read_csv(ip)
            col_n   = "indicator_name" if "indicator_name" in df_i.columns else "indicator_code"
            ind_opts = (df_i["indicator_code"] + " — " + df_i[col_n]).tolist()
            sel_ind  = st.selectbox("Indicator", ind_opts, index=0, key="wb_indicator")
            ind_code = sel_ind.split(" — ")[0]
        else:
            ind_code = st.text_input("Indicator code", value="NY.GDP.MKTP.KD.ZG")
    with c2:
        yr_range = st.selectbox("Period", ["2015–2024", "2015–2019", "2020–2024"],
                                index=0, key="wb_period")
        sy, ey = (int(y) for y in yr_range.split("–"))

    if st.button("▶  Extract indicator", key="btn_ind"):
        try:
            import requests as req
            url = (f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}"
                   f"?format=json&per_page=500&date={sy}:{ey}&page=1")
            with st.spinner(f"Calling World Bank API — {ind_code}…"):
                t0      = time.time()
                r       = req.get(url, timeout=30)
                r.raise_for_status()
                dat     = r.json()
                elapsed = round(time.time() - t0, 2)
            if dat and len(dat) > 1 and dat[1]:
                df_wb = pd.DataFrame(dat[1])
                df_wb["country_name"] = df_wb["country"].apply(
                    lambda x: x["value"] if isinstance(x, dict) else x)
                df_wb["date"]  = pd.to_numeric(df_wb["date"],  errors="coerce")
                df_wb["value"] = pd.to_numeric(df_wb["value"], errors="coerce")
                df_wb = df_wb.dropna(subset=["value"])
                total_pages = dat[0].get("pages", 1)
                st.success(
                    f"✅  {len(df_wb)} rows in {elapsed}s · "
                    f"{df_wb['country_name'].nunique()} countries · "
                    f"{total_pages} page(s)")
                sample = next(
                    (c for c in ["France", "Germany", "United States", "China", "Tunisia"]
                     if c in df_wb["country_name"].values),
                    df_wb["country_name"].iloc[0])
                df_s = (df_wb[df_wb["country_name"] == sample]
                        .sort_values("date")[["date", "value"]]
                        .rename(columns={"date": "Year", "value": ind_code}))
                st.markdown(f"**{sample}** — {ind_code}")
                st.line_chart(df_s.set_index("Year"), height=180)
                with st.expander("Preview — first 10 rows"):
                    st.dataframe(df_wb[["country_name", "date", "value"]].head(10),
                                 width='stretch')
            else:
                st.warning("No data returned — check indicator code.")
        except Exception as e:
            st.error(f"API error: {e}")


# ════════════════════════════════════════════════════════════════
# RUN PIPELINE — local dev only, distinct section
# ════════════════════════════════════════════════════════════════

thin_divider()
section_label("Run Pipeline — Local Development")

is_prod = not (ROOT / ".env").exists()

if is_prod:
    st.markdown("""
    <div style="background:#141E30; border:1px solid #243044; border-radius:10px;
                padding:.9rem 1.1rem; font-size:14px; color:#5A7A9A;">
      Pipeline runs locally — metrics above reflect the last execution.
      Trade data extraction runs separately via CLI.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#1C1407; border:1px solid #78350F44;
                border-left:3px solid #F59E0B; border-radius:0 10px 10px 0;
                padding:.9rem 1.1rem; font-size:14px; color:#D97706; margin-bottom:.75rem;">
      ⚠️  <strong style="color:#F59E0B;">Local execution only</strong>
      — Silver and Gold complete in seconds.
      Bronze skips already-extracted data (only new indicator values may take a few minutes).
      Trade data extraction runs separately via CLI and is not triggered here.
    </div>
    """, unsafe_allow_html=True)

    import subprocess

    for key, default in [
        ("bronze_proc",   None),
        ("bronze_result", None),
        ("silver_result", None),
        ("gold_result",   None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    bronze_running = (
        st.session_state.bronze_proc is not None
        and st.session_state.bronze_proc.poll() is None
    )
    bronze_just_done = (
        st.session_state.bronze_proc is not None and not bronze_running
    )

    if bronze_just_done:
        rc = st.session_state.bronze_proc.returncode
        st.session_state.bronze_proc   = None
        st.session_state.bronze_result = "ok" if rc == 0 else f"fail:{rc}"
        etl = load_etl_status()

    def _stage_row(pipeline_key, color):
        p = etl.pipelines.get(pipeline_key)
        if not p:
            return
        cols_s = st.columns(len(p.stages))
        for col, stage in zip(cols_s, p.stages):
            lbl = STAGE_LABELS.get(stage.name, stage.name)
            col.markdown(
                f"<div style='font-size:12px;color:{color};font-weight:600;'>{lbl}</div>"
                f"<div style='font-size:12px;color:#A8C0D8;'>{stage.rows_fmt} rows"
                f" · {stage.duration_fmt}</div>",
                unsafe_allow_html=True,
            )

    # ── Result messages — stack in pipeline order ────────────────
    if st.session_state.bronze_result is not None:
        res = st.session_state.bronze_result
        if res == "ok":
            st.success("✅  Bronze completed")
            _stage_row("bronze", "#34D399")
        else:
            st.error(f"❌  Bronze failed (exit code {res.split(':', 1)[1]}).")

    if st.session_state.silver_result is not None:
        res = st.session_state.silver_result
        if res == "ok":
            st.success("✅  Silver completed")
            _stage_row("silver", "#60A5FA")
        else:
            st.error(f"❌  Silver failed (exit code {res.split(':', 1)[1]}).")

    if st.session_state.gold_result is not None:
        res = st.session_state.gold_result
        if res == "ok":
            st.success("✅  Gold completed")
            _stage_row("gold", "#FBBF24")
        else:
            st.error(f"❌  Gold failed (exit code {res.split(':', 1)[1]}).")

    # ── Buttons ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    if c1.button("▶  Run Bronze", key="run_bronze", disabled=bronze_running):
        for k in ("bronze_result", "silver_result", "gold_result"):
            st.session_state[k] = None
        st.session_state.bronze_proc = subprocess.Popen(
            [sys.executable, "-m", "src.pipelines.run_bronze"], cwd=str(ROOT)
        )
        st.rerun()

    if bronze_running:
        c1.markdown(
            "<div style='font-size:13px;color:#F59E0B;margin-top:6px;'>"
            "⏳ Running…</div>",
            unsafe_allow_html=True,
        )

    if c2.button("▶  Run Silver", key="run_silver", disabled=bronze_running):
        with st.spinner("Running Silver pipeline…"):
            r = subprocess.run(
                [sys.executable, "-m", "src.pipelines.run_silver"],
                cwd=str(ROOT), capture_output=True, text=True
            )
        etl = load_etl_status()
        st.session_state.silver_result = "ok" if r.returncode == 0 else f"fail:{r.returncode}"
        if r.returncode != 0:
            st.error(f"Silver failed:\n{r.stderr[-500:]}")
        st.rerun()

    if c3.button("▶  Run Gold", key="run_gold", disabled=bronze_running):
        with st.spinner("Running Gold pipeline…"):
            r = subprocess.run(
                [sys.executable, "-m", "src.pipelines.run_gold"],
                cwd=str(ROOT), capture_output=True, text=True
            )
        etl = load_etl_status()
        st.session_state.gold_result = "ok" if r.returncode == 0 else f"fail:{r.returncode}"
        if r.returncode != 0:
            st.error(f"Gold failed:\n{r.stderr[-500:]}")
        st.rerun()

    # Auto-refresh every 3s while Bronze is running
    if bronze_running:
        time.sleep(3)
        st.rerun()
