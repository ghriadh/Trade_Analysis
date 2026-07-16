# Trade Analysis

Interactive trade and economic analysis platform built with Streamlit, powered by a medallion data pipeline (Bronze → Silver → Gold).

Data sources: **UN Comtrade** (trade flows) · **World Bank API** (economic indicators)

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

## App pages

| Page | Description |
|---|---|
| Pipeline | Run bronze / silver / gold pipelines from the UI |
| Architecture | ETL architecture overview + live DuckDB demo |
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
| Link Signals | Trade-linked signal detection |
| Link Strategy | Strategic country linkage analysis |

---

## Backends

| | CSV only | DuckDB | SQL Server |
|---|---|---|---|
| Server required | No | No | Yes |
| `.env` required | No | No | Yes |
| Streamlit Cloud | Yes | Yes | No |

DuckDB is recommended for local development and Streamlit Cloud deployment.

---

## Project structure

```
Trade_Analysis/
├── app/
│   ├── Home.py               # Streamlit entry point
│   └── pages/                # Analysis pages
├── src/
│   ├── bronze/               # Extraction scripts
│   ├── silver/               # Transformation scripts
│   ├── gold/                 # Dimension & fact builders
│   ├── pipelines/            # run_bronze / run_silver / run_gold
│   ├── bootstrap/            # Schema creation
│   └── config/               # Settings, DB config, logging
├── data/
│   ├── bronze/               # Raw CSV (git-ignored)
│   ├── silver/               # Cleaned CSV (git-ignored)
│   ├── gold/                 # Star schema CSV (tracked)
│   └── resources/            # Static reference files
├── setup.bat                 # Windows first-time setup
├── setup.sh                  # Linux / Mac first-time setup
└── Makefile                  # Dev workflow
```

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

Trade data is extracted separately in batches (volume: ~200 reporters × multiple years):

```bash
# 1. Extract by year (repeat for each year needed)
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X

# 2. Consolidate all batches → creates bronze trade_data
python -m src.bronze.extract_trade_data --consolidate --db duckdb

# 3. Re-run silver + gold to integrate trade_data
python -m src.pipelines.run_silver --module trade_data --db duckdb
python -m src.pipelines.run_gold   --db duckdb
```

---

## Streamlit Cloud deployment

1. Push the repo to GitHub (gold CSVs and DuckDB file are tracked)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → set main file to `app/Home.py`
3. No `.env` needed — the app auto-detects Cloud and uses DuckDB

---

## Environment variables

Required only for SQL Server or Databricks. Create a `.env` file at the project root:

```
DB_BACKEND=sqlserver
DB_SERVER=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```
