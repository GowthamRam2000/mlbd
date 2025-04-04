#!/bin/bash

# Setup script for the Inventory Management System
echo "Setting up Inventory Management System..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Initialize database and generate data
python -c "from utils.data_loader import DataManager; DataManager().generate_synthetic_data()"

echo "Setup complete! To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the Streamlit app: streamlit run app.py"