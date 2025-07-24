@echo off
REM West Nile Virus Predictor - Virtual Environment Setup (Windows)

echo Setting up Python virtual environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo Setup complete! Virtual environment is now active.
echo To activate the environment in the future, run: venv\Scripts\activate.bat
echo To deactivate, run: deactivate