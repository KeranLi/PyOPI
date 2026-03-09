#!/bin/bash

echo "Setting up OPI Python Environment"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda is available, creating new environment..."
    conda create -n opi-env python=3.9 -y
    echo "Activating opi-env environment..."
    conda activate opi-env
    echo "Installing required packages..."
    pip install -r requirements.txt
else
    echo "Conda not found, attempting to install using pip..."
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    echo "Installing required packages using pip..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Environment setup complete!"
echo "To test the installation, run:"
echo "  python verify_installation.py"
echo ""