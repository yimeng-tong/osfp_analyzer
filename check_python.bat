@echo off
chcp 65001 >nul
echo ====================================================
echo Python Installation Checker
echo ====================================================
echo.

echo Checking Python installation methods...
echo.

REM Method 1: Check python command
echo [Method 1] Checking 'python' command:
python --version 2>nul
if errorlevel 1 (
    echo   Result: 'python' command NOT FOUND
) else (
    echo   Result: 'python' command FOUND
    python --version
)

echo.

REM Method 2: Check py command
echo [Method 2] Checking 'py' command:
py --version 2>nul
if errorlevel 1 (
    echo   Result: 'py' command NOT FOUND
) else (
    echo   Result: 'py' command FOUND
    py --version
)

echo.

REM Method 3: Check python3 command
echo [Method 3] Checking 'python3' command:
python3 --version 2>nul
if errorlevel 1 (
    echo   Result: 'python3' command NOT FOUND
) else (
    echo   Result: 'python3' command FOUND
    python3 --version
)

echo.
echo ====================================================
echo Diagnosis:
echo ====================================================

REM Determine what to do based on results
python --version >nul 2>&1
if not errorlevel 1 (
    echo ✓ Python is properly installed and in PATH
    echo   You can use: python app.py
    goto :check_packages
)

py --version >nul 2>&1
if not errorlevel 1 (
    echo ✓ Python is installed but 'python' command not in PATH
    echo   You can use: py app.py
    echo   Recommendation: Add Python to PATH for convenience
    goto :check_packages
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    echo ✓ Python3 is available
    echo   You can use: python3 app.py
    goto :check_packages
)

echo ✗ Python is NOT installed or not accessible
echo.
echo SOLUTION:
echo 1. Download Python from: https://python.org/downloads/
echo 2. Choose Python 3.8 or higher
echo 3. During installation, IMPORTANT: Check "Add Python to PATH"
echo 4. Restart your computer after installation
echo 5. Run this script again to verify
echo.
goto :end

:check_packages
echo.
echo Checking required packages...
python -c "import flask" >nul 2>&1 || py -c "import flask" >nul 2>&1 || python3 -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ✗ Flask package not installed
    echo   Run: pip install flask pandas numpy openpyxl matplotlib seaborn scikit-learn
) else (
    echo ✓ Flask package found
)

python -c "import pandas" >nul 2>&1 || py -c "import pandas" >nul 2>&1 || python3 -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo ✗ Pandas package not installed
    echo   Run: pip install pandas
) else (
    echo ✓ Pandas package found
)

echo.
echo ====================================================
echo Next Steps:
echo ====================================================
echo If Python is installed:
echo   1. Run: pip install -r requirements.txt
echo   2. Run: python app.py  (or py app.py)
echo   3. Open browser: http://localhost:5000
echo.
echo If Python is NOT installed:
echo   1. Download from: https://python.org
echo   2. Install with "Add to PATH" checked
echo   3. Restart computer
echo   4. Run this script again

:end
echo.
pause