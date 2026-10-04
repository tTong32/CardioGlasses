# Hardware Setup Guide

Quick guide for connecting the Arduino glasses to the AI pipeline and backend.

## Prerequisites

1. Arduino glasses connected via USB
2. Python environment set up (`pip install -r requirements.txt`)
3. `.env` file exists (copy from `.env.example` if needed)

## Quick Start (Automated)

### macOS/Linux:
```bash
./start_hardware.sh
```

### Windows:
```bash
python start_hardware.py
```

The script will:
1. Check if backend is running
2. List available serial ports
3. Help you configure SERIAL_PORT in .env
4. Test the serial connection
5. Start the live stream

## Manual Steps

If you prefer to do it manually:

### 1. Find your serial port
```bash
python -m tools.serial_check --list
```

### 2. Test the hardware stream
```bash
# Replace with your actual port
python -m tools.serial_check --port /dev/cu.usbmodem14201  # macOS
python -m tools.serial_check --port COM5                    # Windows
```

### 3. Set the port in .env
```bash
# Add or update this line in .env
SERIAL_PORT=/dev/cu.usbmodem14201
```

### 4. Start the backend (terminal 1)
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the hardware stream (terminal 2)
```bash
# Template mode (no LLM, faster)
python -m tools.live_stream --no-llm

# OR with LLM wording (needs GEMINI_API_KEY in .env)
python -m tools.live_stream
```

### 6. Open the dashboard

**On laptop:** http://localhost:8000

**On phone (same WiFi):**
```bash
# Find your laptop IP
ifconfig | grep "inet "        # macOS/Linux
ipconfig                        # Windows

# Then visit: http://[laptop-ip]:8000
```

## Expected Hardware Format

The Arduino must send JSON lines (Contract A) at ~50Hz:

```json
{"t": 1760000000123, "ppg": 51234, "ax": 0.02, "ay": -0.98, "az": 0.10, "gx": 0.5, "gy": 0.1, "gz": 0.0}
```

- `t`: Unix timestamp in milliseconds
- `ppg`: Photoplethysmogram raw ADC value
- `ax`, `ay`, `az`: Accelerometer in g (not m/s²)
- `gx`, `gy`, `gz`: Gyroscope in rad/s or deg/s

## Troubleshooting

### "Permission denied" on serial port
```bash
# macOS/Linux
sudo chmod 666 /dev/cu.usbmodem14201

# Or add yourself to dialout group (Linux)
sudo usermod -a -G dialout $USER
```

### "Can't reach backend"
Make sure uvicorn is running first:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### "PPG nearly flat"
- Sensor not touching skin
- Check hardware connections

### "accelerometer looks like m/s²"
Hardware is sending wrong units. Contract A expects `g` (where gravity ≈ 1.0, not 9.8).

### Phone can't connect
1. Make sure laptop and phone are on same WiFi
2. Check firewall isn't blocking port 8000
3. Try laptop hotspot instead of shared WiFi

## Recording Data for Testing

To record hardware data for later replay (no hardware needed):

```bash
# Record 2 minutes
python -m tools.serial_check --port COM5 --seconds 120 --out data/rec_test.csv

# Replay it later
python -m ai.pipeline data/rec_test.csv --speed 10
```
