@echo off
echo Setting up OPI Python Environment

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Conda is available, creating new environment...
    conda create -n opi-env python=3.9 -y
    echo Activating opi-env environment...
    call conda activate opi-env
    echo Installing required packages...
    pip install -r requirements.txt
) else (
    echo Conda not found, attempting to install using pip...
    python --version
    if %ERRORLEVEL% NEQ 0 (
        echo Python not found. Please install Python 3.8 or higher.
        pause
        exit /b 1
    )
    echo Installing required packages using pip...
    pip install -r requirements.txt
)

echo.
echo Environment setup complete!
echo To test the installation, run:
echo   python verify_installation.py
echo.
pause