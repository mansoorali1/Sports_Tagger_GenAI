#!/bin/bash

echo "Downloading model artifacts..."
python app/download_artifacts.py

echo "Starting FastAPI backend..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit frontend..."
streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
