#!/bin/bash
# West Nile Virus Predictor - Virtual Environment Setup (Unix/macOS/Linux)

echo "Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete! Virtual environment is now active."
echo "To activate the environment in the future, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"