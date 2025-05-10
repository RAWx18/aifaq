#!/bin/bash
# Script to run all the multi-agent RAG system improvements

set -e  # Exit on error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CORE_DIR"

echo "========================================"
echo "Multi-Agent RAG System Improvement Script"
echo "========================================"
echo

# Install required dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed successfully."
echo

# Setup NLP dependencies
echo "Step 2: Setting up NLP dependencies..."
python scripts/setup_nlp_dependencies.py
echo

# Check vector database
echo "Step 3: Checking vector database..."
python scripts/check_vectordb.py

# Ask if user wants to reingest documents
read -p "Would you like to reingest documents into the vector database? (y/n): " reingest
if [[ "$reingest" == "y" ]]; then
    echo "Reingesting documents..."
    python scripts/reingest_vectordb.py --force
    echo "Reingestion completed."
fi
echo

# Run tests
echo "Step 4: Running agent coordination test..."
python scripts/test_agent_coordination.py
echo

echo "Step 5: Running mock test..."
python scripts/test_multi_agent_mock.py
echo

echo "Step 6: Running comprehensive tests..."
python scripts/test_comprehensive.py --mode mock
echo

# Run profiling
echo "Step 7: Running performance profiling..."
python scripts/profile_multi_agent.py
echo

# Restart the server if needed
read -p "Would you like to restart the API server? (y/n): " restart
if [[ "$restart" == "y" ]]; then
    echo "Checking for running API server..."
    api_pid=$(ps -ef | grep "python.*api.py" | grep -v grep | awk '{print $2}')
    
    if [[ -n "$api_pid" ]]; then
        echo "Stopping API server (PID: $api_pid)..."
        kill "$api_pid"
        sleep 2
    fi
    
    echo "Starting API server..."
    nohup python api.py > api.log 2>&1 &
    echo "API server started."
fi
echo

echo "========================================"
echo "Multi-Agent RAG System Improvements Complete!"
echo "Check the log files in the logs directory for details."
echo "========================================"
