@echo off
title OSFP Data Analyzer
echo Starting OSFP Data Analyzer...
echo.

REM Try different Python commands
echo Trying 'python' command...
python --version >nul 2>&1
if not errorlevel 1 (
    echo Found Python, starting application...
    python app.py
    goto :end
)

echo Trying 'py' command...
py --version >nul 2>&1
if not errorlevel 1 (
    echo Found Python via 'py' command, starting application...
    py app.py
    goto :end
)

echo Trying 'python3' command...
python3 --version >nul 2>&1
if not errorlevel 1 (
    echo Found Python3, starting application...
    python3 app.py
    goto :end
)

echo.
echo ERROR: Python not found!
echo.
echo Please install Python from: https://python.org
echo Make sure to check "Add Python to PATH" during installation
echo.
echo Or run 'check_python.bat' for detailed diagnosis
echo.

:end
pause