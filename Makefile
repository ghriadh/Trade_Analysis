.DEFAULT_GOAL := help

ifdef DB
  DB_ARG := --db $(DB)
else
  DB_ARG :=
endif

# Trade extraction args (passed through if set)
YEAR    ?=
PRODUCT ?= TOTAL
FLOWS   ?= M X

.PHONY: help install dirs database schemas \
        bronze silver gold consolidate pipeline \
        app

help:               ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  DB backends : duckdb | sqlserver | databricks"
	@echo "  Examples:"
	@echo "    make install"
	@echo "    make dirs"
	@echo "    make schemas      DB=duckdb"
	@echo "    make pipeline     DB=duckdb"
	@echo "    make bronze       DB=duckdb"
	@echo "    make consolidate  DB=duckdb"
	@echo "    make app"

# ── Environment ──────────────────────────────────────────────

install:            ## Create venv and install dependencies
	python -m venv myenv
	myenv/bin/pip install --upgrade pip -q
	myenv/bin/pip install -r requirements.txt

dirs:               ## Create data/ directory structure
	python -c "from src.config.settings import ensure_directories; ensure_directories()"

# ── Bootstrap (DB) ───────────────────────────────────────────

database:           ## Create the database file/instance (SQL Server only, DuckDB auto-creates)
ifndef DB
	$(error DB is required. Example: make database DB=sqlserver)
endif
	python -m src.bootstrap.create_database

schemas:            ## Create medallion schemas bronze/silver/gold (requires DB=...)
ifndef DB
	$(error DB is required. Example: make schemas DB=duckdb)
endif
	python -m src.bootstrap.create_schemas $(DB_ARG)

# ── Pipeline ─────────────────────────────────────────────────

bronze:             ## Run bronze pipeline             [DB=duckdb|sqlserver|databricks]
	python -m src.pipelines.run_bronze $(DB_ARG)
	python -m src.bronze.extract_trade_data --consolidate $(DB_ARG)

silver:             ## Run silver pipeline             [DB=duckdb|sqlserver|databricks]
	python -m src.pipelines.run_silver $(DB_ARG)

gold:               ## Run gold pipeline               [DB=duckdb|sqlserver|databricks]
	python -m src.pipelines.run_gold $(DB_ARG)

pipeline:           ## Full pipeline: dirs → [schemas] → bronze → silver → gold  [DB=duckdb|sqlserver]
	$(MAKE) dirs
ifdef DB
	$(MAKE) schemas DB=$(DB)
endif
	$(MAKE) bronze  $(if $(DB),DB=$(DB),)
	$(MAKE) silver  $(if $(DB),DB=$(DB),)
	$(MAKE) gold    $(if $(DB),DB=$(DB),)

# ── App ──────────────────────────────────────────────────────

app:                ## Launch Streamlit app
	streamlit run app/Home.py
