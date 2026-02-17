#!/bin/bash
# Simulates production before you deploy

echo "=========================================="
echo "  PRODUCTION SIMULATION TEST"
echo "=========================================="

if [ ! -f ".env" ]; then
    echo "❌  .env not found. Run: cp .env.example .env"
    exit 1
fi
echo "✓  .env found"

if [ ! -f "data/processed/transcripts_with_embeddings.csv" ]; then
    echo "❌  CSV not found. Run embedding notebook first."
    exit 1
fi
echo "✓  Data file found"

pip install -r requirements.txt -q
echo "✓  Dependencies installed"

echo ""
echo "Starting API on http://localhost:8000"
echo "Open: http://localhost:8000/docs"
echo ""

export PORT=8000
uvicorn api.main:app --host 0.0.0.0 --port $PORT --reload