#!/bin/bash
# West Nile Virus Predictor - Virtual Environment Setup
# Compatible with Unix/macOS/Linux and Windows Git Bash

echo "Setting up Python virtual environment..."

# Detect if we're on Windows (Git Bash/MSYS)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]]; then
    PYTHON_CMD="python"
    VENV_ACTIVATE="venv/Scripts/activate"
    echo "Detected Windows environment (Git Bash)"
else
    PYTHON_CMD="python3"
    VENV_ACTIVATE="venv/bin/activate"
    echo "Detected Unix/macOS/Linux environment"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_ACTIVATE

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
$PYTHON_CMD -m pip install -r requirements.txt

echo "Setup complete! Virtual environment is now active."
echo "To activate the environment in the future, run: source $VENV_ACTIVATE"
echo "To deactivate, run: deactivate"