#!/bin/bash
# Startup script for FinSense

set -e

echo "🚀 Starting FinSense Financial Intelligence System..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from config/env.example..."
    cp config/env.example .env
    echo "📝 Please edit .env file with your API keys"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$python_version" != "3.10" ]; then
    echo "⚠️  Warning: Python 3.10 recommended. Found: $python_version"
fi

# Create necessary directories
mkdir -p data models logs chatbot/vector_store chatbot/knowledge_base

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    exec "$@"
else
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies if needed
    if [ ! -f "venv/.installed" ]; then
        echo "📥 Installing dependencies..."
        pip install -r requirements.txt
        touch venv/.installed
    fi
    
    # Start services based on argument
    case "$1" in
        backend)
            echo "🔧 Starting backend API..."
            python run_backend.py
            ;;
        dashboard)
            echo "📊 Starting Streamlit dashboard..."
            streamlit run dashboard/app.py
            ;;
        both)
            echo "🔧 Starting backend API..."
            python run_backend.py &
            BACKEND_PID=$!
            sleep 2
            echo "📊 Starting Streamlit dashboard..."
            streamlit run dashboard/app.py
            kill $BACKEND_PID
            ;;
        *)
            echo "Usage: ./start.sh [backend|dashboard|both]"
            echo "  backend   - Start backend API only"
            echo "  dashboard - Start dashboard only"
            echo "  both      - Start both services"
            exit 1
            ;;
    esac
fi













