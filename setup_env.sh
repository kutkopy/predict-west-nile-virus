#!/bin/bash

# Azure ML Conda Environment Setup Script
# This script creates a new conda environment and installs your requirements

set -e  # Exit on any error

# Configuration
ENV_NAME="predict-west-nile-virus"
PYTHON_VERSION="3.10"
REQUIREMENTS_FILE="requirements.txt"

echo "================================================"
echo "Creating Conda Environment: $ENV_NAME"
echo "================================================"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Warning: Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Aborted. Using existing environment."
        conda activate $ENV_NAME
        exit 0
    fi
fi

# Create conda environment using conda-forge channel directly
echo "Creating new conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION --channel conda-forge --override-channels -y

# Activate environment
echo "Activating environment..."
source /anaconda/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "✓ Environment activated successfully"
else
    echo "✗ Failed to activate environment"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing packages from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
    echo "✓ Packages installed successfully"
else
    echo "Warning: $REQUIREMENTS_FILE not found. Installing packages manually..."
    pip install pandas>=2.0.0 scikit-learn>=1.7.0 numpy>=1.25.2 imbalanced-learn>=0.14.0 dvc[azure]==3.59.0
fi

# Display installed versions
echo ""
echo "================================================"
echo "Installation Summary"
echo "================================================"
python --version
echo ""
echo "Key package versions:"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import imblearn; print(f'imbalanced-learn: {imblearn.__version__}')"
python -c "import dvc; print(f'dvc: {dvc.__version__}')"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To use this environment:"
echo "  1. In terminal: conda activate $ENV_NAME"
echo "  2. In Jupyter: Select kernel 'Python [conda env:$ENV_NAME]'"
echo ""
echo "Environment location:"
conda env list | grep $ENV_NAME
echo ""