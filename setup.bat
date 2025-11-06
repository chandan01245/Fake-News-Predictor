@echo off
echo ============================================
echo   Fake News Predictor - Setup Script
echo ============================================
echo.

echo Creating necessary directories...

if not exist "data" (
    mkdir data
    echo [+] Created: data/
) else (
    echo [OK] Directory exists: data/
)

if not exist "models" (
    mkdir models
    echo [+] Created: models/
) else (
    echo [OK] Directory exists: models/
)

if not exist "processed" (
    mkdir processed
    echo [+] Created: processed/
) else (
    echo [OK] Directory exists: processed/
)

echo.
echo Checking virtual environment...

if not exist "Virtual-env" (
    echo Creating virtual environment...
    python -m venv Virtual-env
    echo [+] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

echo.
echo Activating virtual environment...
call Virtual-env\Scripts\activate.bat

echo.
echo Installing/Updating dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Download dataset from Kaggle and place in data/ folder:
echo    - True.csv
echo    - Fake.csv
echo.
echo 2. Run the web application:
echo    run.bat
echo.
echo 3. Or run manually:
echo    python app.py
echo.
echo ============================================

pause
