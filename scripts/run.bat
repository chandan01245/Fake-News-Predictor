@echo off
echo ============================================
echo   Fake News Predictor - Web UI
echo ============================================
echo.

REM Activate virtual environment
if exist "Virtual-env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call Virtual-env\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found!
    echo Please create it first: python -m venv Virtual-env
    pause
    exit /b 1
)

REM Check if gradio is installed
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

echo.
echo Starting the web application...
echo.
echo The app will open in your browser at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
