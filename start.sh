#!/bin/bash

# Retail Anomaly Detection Startup Script

echo "🛒 Starting Retail Anomaly Detection Dashboard..."
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please configure your Azure credentials."
    echo "   Copy .env.example to .env and update with your Azure OpenAI details."
    exit 1
fi

# Generate sample data if it doesn't exist
if [ ! -f sample_retail_data.csv ]; then
    echo "📊 Generating sample retail dataset..."
    python generate_sample_data.py
fi

# Start the Streamlit application
echo "🚀 Starting Streamlit dashboard..."
echo "   The application will open in your browser at http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

streamlit run app.py --server.headless false
