@echo off
chcp 65001 >nul
echo ====================================================
echo 800g OSFP Optical Module Production Data Analyzer
echo ====================================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in system PATH
    echo.
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    echo Alternative: Try running with 'py' command:
    py --version >nul 2>&1
    if errorlevel 1 (
        echo 'py' command also not found. Python needs to be installed.
        goto :error_exit
    ) else (
        echo Found Python via 'py' command, using that instead...
        set PYTHON_CMD=py
        goto :python_found
    )
) else (
    echo Python found successfully.
    set PYTHON_CMD=python
)

:python_found
echo.
echo Python version:
%PYTHON_CMD% --version
echo.

REM Check if required packages are installed
echo Checking required packages...
%PYTHON_CMD% -c "import flask, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    echo Upgrading pip and setuptools...
    %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install packages. Trying with Tsinghua mirror...
        %PYTHON_CMD% -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip setuptools wheel
        %PYTHON_CMD% -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
        if errorlevel 1 (
            echo Package installation failed. Please check your internet connection.
            goto :error_exit
        )
    )
    echo Packages installed successfully.
) else (
    echo Required packages are already installed.
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results

echo.
echo Starting the application...
echo Please open your browser and visit: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start the application
%PYTHON_CMD% app.py

goto :end

:error_exit
echo.
echo ====================================================
echo Installation Failed - Please follow these steps:
echo ====================================================
echo 1. Download Python from: https://python.org
echo 2. During installation, check "Add Python to PATH"
echo 3. Restart your computer after installation
echo 4. Run this script again
echo.
pause
exit /b 1

:end
pause