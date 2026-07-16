#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "============================================"
echo "  Trade Analysis -- First-time Setup"
echo "============================================"
echo ""

# -- 1. Virtual environment
if [[ -d myenv ]]; then
    echo "[1/3] Virtual environment already exists -- skipping creation."
    source myenv/bin/activate
else
    echo "[1/3] Creating virtual environment..."
    python3 -m venv myenv
    source myenv/bin/activate
fi

# -- 2. Install dependencies
echo ""
echo "[2/3] Installing / checking dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# -- 3. Backend choice
echo ""
echo "[3/3] Choose your backend:"
echo ""
echo "  1) CSV only    (no database, files saved to data/)"
echo "  2) DuckDB      (embedded, no server required)"
echo "  3) SQL Server  (requires .env credentials)"
echo ""
read -rp "  Your choice (1/2/3): " CHOICE

BACKEND=""
case "$CHOICE" in
  2) BACKEND="duckdb" ;;
  3) BACKEND="sqlserver" ;;
  *) echo "  CSV-only mode -- skipping schema creation." ;;
esac

if [[ -n "$BACKEND" ]]; then
    echo ""
    echo "  Creating data/ directories..."
    python -c "from src.config.settings import ensure_directories; ensure_directories()"
    echo "  Creating medallion schemas for backend: $BACKEND"
    python -m src.bootstrap.create_schemas --db "$BACKEND"
fi

# -- 4. What to do next
echo ""
echo "============================================"
echo "  Setup complete!  Backend: ${BACKEND:-CSV only}"
echo "============================================"
echo ""
echo "  What would you like to do next?"
echo ""
echo "  1) Launch Streamlit app"
echo "  2) Run full pipeline  (bronze + silver + gold)"
echo "  3) Exit"
echo ""
read -rp "  Your choice (1/2/3): " ACTION

case "$ACTION" in
  1)
    streamlit run app/Home.py
    ;;
  2)
    echo ""
    echo "  Running bronze pipeline..."
    if [[ -z "$BACKEND" ]]; then
        python -m src.pipelines.run_bronze
        python -m src.bronze.extract_trade_data --consolidate
    else
        python -m src.pipelines.run_bronze --db "$BACKEND"
        python -m src.bronze.extract_trade_data --consolidate --db "$BACKEND"
    fi
    echo ""
    echo "  Running silver pipeline..."
    [[ -z "$BACKEND" ]] && python -m src.pipelines.run_silver \
                        || python -m src.pipelines.run_silver --db "$BACKEND"
    echo ""
    echo "  Running gold pipeline..."
    [[ -z "$BACKEND" ]] && python -m src.pipelines.run_gold \
                        || python -m src.pipelines.run_gold --db "$BACKEND"
    echo ""
    echo "  Pipeline complete!"
    ;;
  *)
    ;;
esac

echo ""
