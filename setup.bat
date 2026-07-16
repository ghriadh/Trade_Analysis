@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Trade Analysis -- First-time Setup
echo ============================================
echo.

REM -- 1. Virtual environment
if exist myenv\ (
    echo [1/3] Virtual environment already exists -- skipping creation.
    call myenv\Scripts\activate
) else (
    echo [1/3] Creating virtual environment...
    python -m venv myenv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.11+ and retry.
        exit /b 1
    )
    call myenv\Scripts\activate
)

REM -- 2. Install dependencies
echo.
echo [2/3] Installing / checking dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)

REM -- 3. Backend choice
echo.
echo [3/3] Choose your backend:
echo.
echo   1) CSV only    (no database, files saved to data/)
echo   2) DuckDB      (embedded, no server required)
echo   3) SQL Server  (requires .env credentials)
echo.
set /p CHOICE="   Your choice (1/2/3): "

set BACKEND=
if "%CHOICE%"=="2" set BACKEND=duckdb
if "%CHOICE%"=="3" set BACKEND=sqlserver

if "%BACKEND%"=="" (
    echo.
    echo    CSV-only mode -- skipping schema creation.
    goto next_step
)

echo.
echo    Creating data/ directories...
python -c "from src.config.settings import ensure_directories; ensure_directories()"
echo    Creating medallion schemas for backend: %BACKEND%
python -m src.bootstrap.create_schemas --db %BACKEND%
if errorlevel 1 (
    echo ERROR: Schema creation failed. Check your .env configuration.
    exit /b 1
)

:next_step
REM -- 4. What to do next
echo.
echo ============================================
echo   Setup complete!  Backend: %BACKEND: =CSV only%
echo ============================================
echo.
echo   What would you like to do next?
echo.
echo   1) Launch Streamlit app
echo   2) Run full pipeline  (bronze + silver + gold)
echo   3) Exit
echo.
set /p ACTION="   Your choice (1/2/3): "

if "%ACTION%"=="1" (
    streamlit run app\Home.py
    goto end
)

if "%ACTION%"=="2" (
    echo.
    echo    Running full pipeline...
    echo.
    python -c "from src.config.settings import ensure_directories; ensure_directories()"
    python -m src.pipelines.run_bronze %BACKEND_ARG%
    echo.
    echo    Consolidating trade data...
    if "%BACKEND%"=="" (
        python -m src.bronze.extract_trade_data --consolidate
    ) else (
        python -m src.bronze.extract_trade_data --consolidate --db %BACKEND%
    )
    echo.
    echo    Running silver pipeline...
    if "%BACKEND%"=="" (
        python -m src.pipelines.run_silver
    ) else (
        python -m src.pipelines.run_silver --db %BACKEND%
    )
    echo.
    echo    Running gold pipeline...
    if "%BACKEND%"=="" (
        python -m src.pipelines.run_gold
    ) else (
        python -m src.pipelines.run_gold --db %BACKEND%
    )
    echo.
    echo    Pipeline complete!
    goto end
)

:end
echo.
endlocal
