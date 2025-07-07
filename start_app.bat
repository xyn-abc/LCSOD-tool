@echo off
echo Starting LCSOD Tool...
echo.

REM Check if virtual environment exists
if not exist "LCSOD" (
    echo Creating virtual environment 'LCSOD'...
    python -m venv LCSOD
    if errorlevel 1 (
        echo Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment 'LCSOD' already exists.
)

echo.
echo Activating virtual environment...
call LCSOD\Scripts\activate.bat

echo.
echo Installing/updating dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Starting Streamlit app...
python -m streamlit run app/main.py

REM Keep the window open if there's an error
if errorlevel 1 pause 