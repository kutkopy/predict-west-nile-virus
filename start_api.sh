#!/bin/bash

echo "Starting West Nile Virus Prediction API..."
echo "========================================="

# Activate virtual environment
source venv/bin/activate

# Start the API server
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"

python api.py