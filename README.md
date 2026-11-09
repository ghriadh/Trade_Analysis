# Trade Analysis

🇫🇷 [Lire en français](README.fr.md)

Interactive trade and economic analysis platform built with Streamlit, powered by a medallion data pipeline (Bronze → Silver → Gold) that runs — by design — on **three interchangeable database backends**: SQL Server, DuckDB, and Databricks.

Data sources: **UN Comtrade** (bilateral trade flows) · **World Bank API** (macroeconomic indicators)

> This project was built as a skills portfolio: several parts of the pipeline are deliberately implemented more than once (pandas vs. pure SQL, three database backends) to demonstrate the same problem solved different ways rather than to avoid duplication.

---

## Live app

🔗 _Deployed app link — add once live on Streamlit Community Cloud._

The **Architecture** page includes a live demo that runs the exact same SQL query against DuckDB, Databricks and SQL Server simultaneously and shows the results side by side, to prove the pipeline produces identical data regardless of backend.

---

## Technical highlights

A few problems this project had to solve beyond "load CSV into a dashboard":

- **Concurrent API extraction.** UN Comtrade data (~200 reporters × multiple years × 2 flows) is fetched with `asyncio` + a bounded semaphore (configurable `--workers`, default 4) so dozens of calls run concurrently without blocking the event loop, with automatic backoff on HTTP 429 rate-limiting.
- **Resumable, checkpointed downloads.** Every `(reporter, flow)` pair (trade data) and every indicator (World Bank) writes a `.checkpoints/` marker on success. Re-running an interrupted extraction skips everything already done instead of re-downloading from scratch; failed reporters are logged to `failed_reporters.json` for targeted retries.
- **Retry-hardened HTTP.** The World Bank extractor uses a `requests` session wired with `urllib3`'s `Retry` adapter (exponential backoff on 500/502/503/504) instead of a bare `requests.get`.
- **One DataFrame, three databases.** A single `save_to_db()` abstraction (`src/db/loader.py`) writes the same DataFrame to SQL Server, DuckDB or Databricks through one SQLAlchemy connector interface, with per-backend tuning (DuckDB and Databricks get multi-row batch inserts with different row caps; SQL Server deliberately keeps `fast_executemany` instead, since the two batching strategies don't compose).
- **Two ETL implementations, same output.** The medallion pipeline (bronze → silver → gold) is implemented once in pandas (`src/bronze|silver|gold/`) and once in pure SQL (`src/sql/`, DuckDB-native `CREATE TABLE AS SELECT`). Both are validated to produce identical gold-layer tables (row counts, referential integrity, numeric divergence checks).
- **Two analysis implementations, same questions.** The trade/indicators analysis exists both as pandas notebooks (`Analysis/notebooks/`) and as pure-SQL notebooks using `jupysql` (`Analysis/notebooks_sql/`, `src/sql/analysis/`) — same questions, same answers, different tool.
- **Cross-domain analysis ("Link").** Trade-flow analysis and macro-indicator analysis are two separate notebooks on their own — they're combined in a third analysis layer (`Analysis/Analysis_files/Link/`, app pages *Link Signals* / *Link Strategy*) that asks which World Bank indicators actually explain trade-flow behavior (concentration, volatility, bilateral dependency) for a given country.
- **Cross-backend correctness checks.** Because the same gold tables exist on three engines, numeric comparisons between them can't use naive rounding — floating-point summation order differs by engine at trillion-dollar scale, so comparisons use relative tolerance (`numpy.allclose`), not decimal rounding.

---

## App pages

| Page | Description |
|---|---|
| Pipeline | Run bronze / silver / gold pipelines from the UI |
| Architecture | ETL architecture overview + live cross-backend demo |
| Trade Overview | Global trade summary and trends |
| Global Trade Dynamic | Trade flow evolution over time |
| Country Trade Profile | Exports, imports and trade balance by country |
| Cross-Country Comparison | Side-by-side country benchmarking |
| Bilateral Risk Analysis | Risk scoring between country pairs |
| Typology Scoring | Country classification by trade profile |
| Indicators Overview | World Bank indicators dashboard |
| Indicator Deep Dive | Deep analysis of a single indicator |
| Country Indicators Profile | Full indicator profile per country |
| Advanced Analysis | Forecasting and statistical models |
| Link Signals | Which indicators explain trade behavior |
| Link Strategy | Strategic country linkage analysis |

---

## Backends

| | DuckDB | Databricks | SQL Server |
|---|---|---|---|
| Server required | No (embedded) | Yes (cloud warehouse) | Yes |
| `.env` required | No | Yes (token) | Yes (Windows auth) |
| Works on Streamlit Cloud | Yes | Yes | No — local network only |

DuckDB is the default for local development and for the public Streamlit Cloud deployment. Databricks and SQL Server are fully wired and validated, shown live on the Architecture page when reachable.

---

## Project structure

```
Trade_Analysis/
├── app/
│   ├── Home.py                # Streamlit entry point
│   └── pages/                 # Analysis pages
├── src/
│   ├── bronze/ silver/ gold/  # pandas-based ETL (one implementation)
│   ├── sql/                   # pure-SQL ETL, DuckDB-native (second implementation)
│   ├── pipelines/              # run_bronze / run_silver / run_gold
│   ├── db/                     # connector factory + multi-backend loader
│   ├── bootstrap/              # Schema creation
│   └── config/                 # Settings, DB config, logging
├── Analysis/
│   ├── notebooks/              # pandas analysis (Trade, Indicators, Link)
│   └── notebooks_sql/          # pure-SQL analysis (jupysql)
├── data/
│   ├── bronze/ silver/         # raw / cleaned CSV (git-ignored, regenerable)
│   ├── gold/                   # star schema CSV (tracked)
│   └── resources/              # static reference files
├── setup.bat / setup.sh        # first-time setup
└── Makefile                    # dev workflow
```

---

## Quick start

**Windows**
```bat
.\setup.bat
```

**Linux / Mac**
```bash
chmod +x setup.sh && ./setup.sh
```

The setup script handles everything: virtual environment, dependencies, database schemas, and offers to run the pipeline or launch the app immediately.

---

## Manual setup

```bash
python -m venv myenv
source myenv/bin/activate          # Linux/Mac
# myenv\Scripts\activate           # Windows

pip install -r requirements.txt

python -c "from src.config.settings import ensure_directories; ensure_directories()"
python -m src.bootstrap.create_schemas --db duckdb   # skip for CSV-only

python -m src.pipelines.run_bronze --db duckdb
python -m src.bronze.extract_trade_data --consolidate --db duckdb
python -m src.pipelines.run_silver --db duckdb
python -m src.pipelines.run_gold   --db duckdb

streamlit run app/Home.py
```

---

## Makefile

```bash
make install              # Create venv + install dependencies
make pipeline DB=duckdb   # Full pipeline (bronze → silver → gold)
make pipeline             # Same, CSV-only mode
make bronze   DB=duckdb   # Bronze only
make silver   DB=duckdb   # Silver only
make gold     DB=duckdb   # Gold only
make app                  # Launch Streamlit app
```

---

## Trade data extraction

Trade data is extracted separately in batches (volume: ~200 reporters × multiple years), concurrently, with checkpointed resume:

```bash
# 1. Extract by year (repeat for each year needed) — bounded concurrency + checkpoints
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X --workers 6

# 2. Consolidate all batches → creates bronze trade_data
python -m src.bronze.extract_trade_data --consolidate --db duckdb

# 3. Re-run silver + gold to integrate trade_data
python -m src.pipelines.run_silver --module trade_data --db duckdb
python -m src.pipelines.run_gold   --db duckdb
```

Interrupting and re-running step 1 skips every `(reporter, flow)` pair already checkpointed — only the missing ones are fetched.

---

## Streamlit Cloud deployment

1. Push the repo to GitHub (gold-layer CSVs and the DuckDB file are tracked; raw bronze/silver and the pure-SQL duplicate DuckDB file are git-ignored and regenerated locally if needed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → set main file to `app/Home.py`
3. In the app's **Secrets**, set `DB_BACKEND = "duckdb"` so the public instance uses the embedded file (SQL Server is local-only by design; Databricks credentials can be added too if you want the live cross-backend demo to reach it from the cloud)

---

## Environment variables

Required only for SQL Server or Databricks. Create a `.env` file at the project root:

```
DB_BACKEND=sqlserver
DB_SERVER=...
DB_NAME=...

# or, for Databricks:
DB_BACKEND=databricks
DB_HOST=...
DB_HTTP_PATH=...
DB_TOKEN=...
DB_CATALOG=...
```
