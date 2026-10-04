#!/bin/bash
# Quick start script for connecting hardware to the AI pipeline + backend

set -e

echo "🩺 CardioGlasses Hardware Setup"
echo "================================"
echo ""

# Check if backend is running
echo "1. Checking backend..."
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running at http://127.0.0.1:8000"
else
    echo "❌ Backend is NOT running"
    echo ""
    echo "Start the backend in another terminal:"
    echo "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    read -p "Press Enter once backend is started, or Ctrl+C to exit..."
fi

# List serial ports
echo ""
echo "2. Available serial ports:"
python -m tools.serial_check --list
echo ""

# Check if SERIAL_PORT is set in .env
if grep -q "^SERIAL_PORT=..*" .env 2>/dev/null; then
    CURRENT_PORT=$(grep "^SERIAL_PORT=" .env | cut -d'=' -f2)
    echo "📍 Current .env has: SERIAL_PORT=$CURRENT_PORT"
    read -p "Use this port? (y/n): " USE_CURRENT
    if [[ "$USE_CURRENT" != "y" ]]; then
        read -p "Enter serial port (e.g., /dev/cu.usbmodem14201 or COM5): " NEW_PORT
        # Update .env
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^SERIAL_PORT=.*|SERIAL_PORT=$NEW_PORT|" .env
        else
            sed -i "s|^SERIAL_PORT=.*|SERIAL_PORT=$NEW_PORT|" .env
        fi
        echo "✅ Updated .env with SERIAL_PORT=$NEW_PORT"
        CURRENT_PORT=$NEW_PORT
    fi
else
    read -p "Enter serial port (e.g., /dev/cu.usbmodem14201 or COM5): " NEW_PORT
    echo "SERIAL_PORT=$NEW_PORT" >> .env
    echo "✅ Added SERIAL_PORT=$NEW_PORT to .env"
    CURRENT_PORT=$NEW_PORT
fi

# Quick serial check
echo ""
echo "3. Testing serial stream (5 seconds)..."
timeout 5s python -m tools.serial_check --port "$CURRENT_PORT" 2>/dev/null || true
echo ""

# Ask about LLM mode
echo "4. Wording mode:"
echo "   a) Template mode (fast, no API key needed)"
echo "   b) LLM mode (Gemini wording, needs GEMINI_API_KEY in .env)"
read -p "Choose (a/b) [default: a]: " LLM_CHOICE

LLM_FLAG=""
if [[ "$LLM_CHOICE" != "b" ]]; then
    LLM_FLAG="--no-llm"
fi

# Get laptop IP for phone connection
echo ""
echo "5. Dashboard URLs:"
echo "   Laptop: http://localhost:8000"
if [[ "$OSTYPE" == "darwin"* ]]; then
    LAPTOP_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n1)
else
    LAPTOP_IP=$(hostname -I | awk '{print $1}')
fi
if [[ -n "$LAPTOP_IP" ]]; then
    echo "   Phone:  http://$LAPTOP_IP:8000"
fi
echo ""

# Start streaming
echo "================================"
echo "🚀 Starting live stream..."
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

python -m tools.live_stream $LLM_FLAG
