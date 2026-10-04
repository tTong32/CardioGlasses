#!/bin/bash
# Run the complete CardioGlasses demo

set -e

echo "🩺 CardioGlasses Demo"
echo "===================="
echo ""

# Kill any existing backend
echo "Cleaning up old processes..."
pkill -f "uvicorn backend.main:app" 2>/dev/null || true
sleep 1

# Start backend in background
echo "Starting backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 > /tmp/cardioglasses-backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Backend ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Backend failed to start"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 0.5
done

echo ""
echo "📊 Dashboard: http://localhost:8000"
echo "📁 Demo CSV: data/demo_2min_elevated_rest.csv"
echo ""
echo "Timeline (at speed 10 = 10x faster):"
echo "  0-7s:   Baseline calibration"
echo "  7-7.5s: Walking"
echo "  7.5-13s: Elevated heart rate building"
echo "  13.4s:  🚨 NOTIFY ALERT triggers"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8000 2>/dev/null || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8000 2>/dev/null || true
fi

sleep 2

# Run the demo with explanation mode
echo "▶️  Running demo with clinical reasoning..."
python3 -m ai.pipeline data/demo_2min_elevated_rest.csv --speed 10 --no-llm --explain

# Cleanup
echo ""
echo "Demo complete. Stopping backend..."
kill $BACKEND_PID 2>/dev/null || true
echo "✅ Done"
