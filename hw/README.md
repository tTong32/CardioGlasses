# Hardware (P1)

Pipeline spec, straight from the build log (Oct 3 2026): the glasses sample PPG and IMU
together at 100 Hz on an ESP32 and stream them over BLE; a laptop bridge decodes,
timestamps, and converts each sample to Contract A JSON.

```
hw/
  firmware/            # ESP32 (Arduino + PlatformIO)
    src/main.cpp
    platformio.ini       # board = esp32doit-devkit-v1
  bridge/               # laptop side (Python)
    live_bridge.py        # production: BLE -> Contract A JSON, live + saved for replay
    ble_logger.py          # prototype: BLE -> CSV, written once at the end of a run
    convert_firmware_csv.py # turns a ble_logger.py CSV into Contract A (csv + jsonl)
    plot_ppg.py             # analysis reference: bandpass, beat detection, spectral HR
    recordings/              # raw ble_logger.py captures (prototype format)
  requirements.txt       # bleak, for the bridge
```

## 1. Hardware and wiring

Both sensors run off the ESP32's 3.3 V rail and share SDA/SCL. Draw is about 120-160 mA at
5 V from a USB power bank.

| Part | Role | I2C address |
|---|---|---|
| ESP32 DevKit V1 (ESP32-D0WD-V3, CP2102 USB) | MCU + BLE radio | - |
| MAX30102 breakout | PPG (IR + red LEDs) | 0x57 |
| MPU6050 breakout | Accel + gyro | 0x68 (AD0 unconnected) |

| ESP32 pin | MAX30102 | MPU6050 |
|---|---|---|
| 3V3 | VIN | VCC |
| GND | GND | GND |
| D21 (SDA) | SDA | SDA |
| D22 (SCL) | SCL | SCL |

- A 10-100 µF capacitor across MAX30102 VIN-GND, at the sensor. Without it, high LED power
  browned the sensor out and it stopped sampling.
- Never wire a sensor to the ESP32's `VIN` pin -- on this board that's the 5 V USB rail, not 3.3 V.
- IMU mounted right beside the PPG sensor so it measures the same movement.
- Placement: ear crease behind the ear on the temple tip, constant pressure (foam pad +
  strap), light shielding.
- Flashing needs the USB cable (Windows CP210x driver, COM6 in testing). Running only needs
  the power bank.

## 2. Firmware (`firmware/`)

C++ on the Arduino framework, built with PlatformIO (`board = esp32doit-devkit-v1`).
`setup()` brings up serial, I2C, the PPG, the IMU, then BLE, in that order.

Libs (`platformio.ini`): `h2zero/NimBLE-Arduino@^2.1.0` (resolved 2.5.1),
`sparkfun/SparkFun MAX3010x Pulse and Proximity Sensor Library@^1.1.2`.

1. `Serial.begin(115200)` for debug output.
2. `Wire.begin(21, 22)` + `Wire.setClock(400000)`: I2C at 400 kHz on D21/D22.
3. MAX30102: `ppg.begin(Wire, I2C_SPEED_FAST)`, halts with "MAX30102 not found" if absent.
   Then `ppg.setup(0x7F, 4, 2, 400, 411, 16384)`, then `ppg.shutDown()`.
4. MPU6050, raw register writes (no library): `0x6B=0x00` wake (halts with "MPU6050 not
   found" if no ACK), `0x1A=0x03` low-pass ~44 Hz, `0x1B=0x08` gyro ±500 deg/s, `0x1C=0x08`
   accel ±4 g. Prints `WHO_AM_I` (`0x75`), then `0x6B=0x40` to sleep.
5. Expected boot log: `MAX30102 ready`, `MPU6050 ready (WHO_AM_I=0x68)`,
   `BLE advertising as CardioGlasses`. A clone IMU may report a different `WHO_AM_I`.

| `ppg.setup` argument | Value | Meaning |
|---|---|---|
| LED power | `0x7F` | ~25 mA; needed behind the ear (`0x24` was enough on a finger) |
| Sample average | `4` | 400 Hz internal / 4 = 100 Hz output |
| LED mode | `2` | Red + IR |
| Sample rate | `400` | Internal sampling, Hz |
| Pulse width | `411` µs | 18-bit resolution |
| ADC range | `16384` | Full scale; saturation at 262143 counts |

### BLE and power behavior

One BLE peripheral, one notify characteristic. Both sensors sleep until a client connects
and sleep again when it leaves.

| Item | Value |
|---|---|
| Stack | NimBLE (BLE only, no Classic/A2DP) |
| Device name | `CardioGlasses` |
| Service UUID | `6e400001-b5a3-f393-e0a9-e50e24dcca9e` |
| Data characteristic | `6e400003-b5a3-f393-e0a9-e50e24dcca9e`, NOTIFY |
| Requested MTU | 247 bytes (packets need ≥ 99) |

1. `NimBLEDevice::init("CardioGlasses")`, `setMTU(247)`, create server,
   `advertiseOnDisconnect(true)` so a client can reconnect without a reset.
2. Create the service + characteristic, set advertising name/service UUID, enable scan
   response, start advertising.
3. Each `loop()` checks `server->getConnectedCount()`.
4. On connect: wake IMU (`0x6B=0x00`), `ppg.wakeUp()`, `ppg.clearFIFO()`, reset `sampleIdx`
   and the packet buffer, log `Client connected - sensors ON`.
5. On disconnect: `ppg.shutDown()` (LEDs off), IMU sleep (`0x6B=0x40`), log
   `Client disconnected - sensors OFF`.
6. While idle: `delay(50)` and return; no sampling, LEDs dark.

### Sampling loop and packet format

Each PPG sample from the sensor FIFO triggers one IMU read; samples are counted, not
timestamped on-device, because the sensor's own clock keeps the 10 ms spacing exact.

1. `ppg.check()` pulls new samples from the MAX30102 FIFO into the library buffer.
2. For each available sample: `getFIFOIR()`, `getFIFORed()`, `nextSample()`.
3. Read the IMU: 14 bytes from register `0x3B`, keep accel X/Y/Z and gyro X/Y/Z, skip
   temperature.
4. Write a 24-byte record into the packet buffer; increment `sampleIdx`.
5. After 4 records: `setValue(pkt, 96)` + `notify()`, then reset the buffer. 4 records/notify,
   96 bytes, ~25 notifications/s.

Record layout, little-endian (Python `struct` format `<III6h`):

| Offset (bytes) | Field | Type | Raw unit |
|---|---|---|---|
| 0 | `idx` | uint32 | sample counter since connect |
| 4 | `ir` | uint32 | ADC counts, 0-262143 |
| 8 | `red` | uint32 | ADC counts, 0-262143 |
| 12 | `ax, ay, az` | int16 × 3 | 8192 LSB/g |
| 18 | `gx, gy, gz` | int16 × 3 | 65.5 LSB/(deg/s) |

Why binary over BLE instead of JSON: ~2.4 kB/s instead of ~9 kB/s, and the fixed record
size makes truncation detectable. JSON is produced on the laptop, not the ESP32.

## 3. Laptop bridge (`bridge/`)

The bridge connects by device name, subscribes to notifications, and turns each 24-byte
record into one sample with SI-style units and a Unix ms timestamp.

1. `BleakScanner.find_device_by_name("CardioGlasses", timeout=15)`; exits with a clear
   message if not found.
2. `BleakClient` connect; `start_notify(DATA_UUID, handler)`. Connecting is what turns the
   sensors on.
3. Handler: `len(data) % 24 != 0` -> count a bad packet (MTU too small). Unpack each record
   with `struct.Struct("<III6h")`.
4. Drop detection: `idx != last_idx + 1` -> add the gap to `dropped`.
5. Unit conversion: `accel = raw / 8192` (g); `gyro = raw / 65.5` (deg/s).
6. Timestamps: **t (Unix ms) = wall-clock ms at the first packet + idx × 10.** Never the
   per-packet arrival time as `t` -- BLE delivers in bursts and it corrupts beat intervals.
   (`ble_logger.py`, the prototype, instead logs `t_dev_ms = idx * 10` and
   `t_host_ms` = arrival wall clock, and leaves the Contract A conversion to
   `convert_firmware_csv.py` after the fact; `live_bridge.py` applies the real rule live.)
7. Status line every second: samples, rate (target ~100 Hz), dropped (target 0), bad
   (target 0), latest accel.
8. On exit: `stop_notify`, disconnect. The sensors turn off.

Environment: Python 3.11+ (3.12 is what this repo's own setup is tested on), `bleak` 1.1+
(`hw/requirements.txt`), Windows BLE via WinRT. A Mac hub needs Bluetooth permission for the
terminal.

## 4. Output: CSV (prototype) and Contract A JSON (this repo)

This repo emits one Contract A JSON object per sample, at 100 Hz, with the same values the
prototype CSV held.

Prototype CSV header (`ble_logger.py`, written once at the end of a run):
```
idx,t_dev_ms,t_host_ms,ir,red,ax,ay,az,gx,gy,gz
0,0,1791057400123,129650,118402,-0.1401,0.0102,0.9987,0.31,-0.15,0.08
```

Contract A JSON (`live_bridge.py`, one line per sample, streamed as it arrives):
```json
{"t": 1791057400123, "ppg": 129650, "red": 118402, "ax": -0.1401, "ay": 0.0102, "az": 0.9987, "gx": 0.31, "gy": -0.15, "gz": 0.08}
```

| CSV column | JSON key | Rule |
|---|---|---|
| `idx`, `t_dev_ms`, `t_host_ms` | `t` | Unix ms = first-packet wall clock + `idx × 10` |
| `ir` | `ppg` | Raw IR counts, integer, **unscaled** |
| `red` | `red` | Optional second PPG channel |
| `ax`, `ay`, `az` | same | g, 4 decimals |
| `gx`, `gy`, `gz` | same | deg/s, 2 decimals |

- Missing values are `null`, never omitted (Contract rule).
- `ppg`/`red` are raw ADC counts straight through, unscaled. The MAX30102 FIFO is 18-bit
  (0-262143); `ai/config.py`'s `PPG_SENSOR_MIN/MAX` matches that range for its clipping
  check (it used to assume a 16-bit sensor and read every real value as 100% clipped --
  fixed there, not by rescaling in the bridge).
- `red` is additive: `ai/contracts.py`'s `Sample.red` defaults to `None`, so Contract A's
  original 8-key shape still validates unchanged if a producer omits it. CONTRACTS.md
  itself is frozen and doesn't list `red` -- it's covered by that doc's own open note
  ("2nd PPG channel optional"), but say so if anyone's relying on the literal 8-key text.
- **Streamed live, not written at the end.** `live_bridge.py` writes to stdout as it goes;
  delivery method onward (stdout pipe, local websocket, or HTTP to the backend) is still
  open, see below.
- Every run also saves the same JSON lines to a file for replay (A-02):
  `python -m hw.bridge.live_bridge` -> `data/live_<timestamp>.jsonl` by default.

### Recorded sessions already in the repo

`bridge/recordings/*.csv` are raw `ble_logger.py` captures. Converted Contract A versions
are in `../data/` (`rec_ear_motion1`, `rec_imu_test`, `rec_finger_test`, each as `.csv` and
`.jsonl`) -- these are H-06's "3 labelled recordings" deliverable.

```bash
python -m hw.bridge.convert_firmware_csv hw/bridge/recordings/ear_motion1.csv --out data/rec_rest
python -m ai.pipeline data/rec_rest.csv --speed 5
python -m tools.serial_check --from-file data/rec_rest.jsonl
```

- `ear_motion1.csv` -- ear placement, with motion. Best quality window: HR ~83 bpm.
- `imu_test.csv` -- ear placement, steadier. Noisier pulse (best quality 0.46); good IMU
  reference.
- `finger_test.csv` -- finger placement, PPG only (no IMU). Best quality: HR ~73 bpm,
  degrades partway through (signal dropped, likely finger lifted off the sensor).

## 5. Running it

- `firmware/platformio.ini` targets `esp32doit-devkit-v1`; `pio run -t upload` from
  `firmware/`.
- `pip install -r hw/requirements.txt` for the bridge (`bleak`).
- `python -m hw.bridge.live_bridge` to stream live (needs the glasses powered and in range).
- `python hw/bridge/ble_logger.py --seconds 60 --out session.csv` for the older
  capture-then-convert path.
- `python hw/bridge/plot_ppg.py session.csv` for a quick sanity check before converting.

## 6. Analysis reference (`plot_ppg.py`)

These steps produced a 69.8 bpm beat count and 68.8 bpm spectral HR on a clean 10 s
ear-crease segment. `ai/processing.py` owns the production version (bandpass, beat
detection, quality score, activity) -- this is the reference it was built against.

1. Bandpass IR with a 3rd-order Butterworth, 0.7-3.0 Hz (42-180 bpm), zero-phase `filtfilt`.
2. Invert the filtered signal so beats point up (more blood = less reflected IR).
3. Beats: `find_peaks` with minimum spacing 0.5 s and prominence `0.5 × std`. The 0.5 s
   spacing caps HR at 120 bpm; relax it for exertion.
4. Median HR = `60 / median(beat interval)`.
5. Spectral HR: Welch PSD (`nfft = 2^14`), peak frequency between 0.7 and 3.0 Hz × 60. Use
   on short windows (~5-10 s); over 30 s with a noisy stretch it read 90.8 against a true
   ~70.
6. Motion: moving if gyro magnitude > 15 deg/s or `|accel magnitude - 1 g| > 0.08 g`;
   widened to any motion within 0.5 s.
7. Gating: a beat interval counts only if both its beats fall outside motion.

Observed limits, carried into `ai/processing.py`'s quality score: behind-ear pulse is about
0.3-0.5% of the DC level versus 1-3% on a finger; movement can shift the sensor and leave
the pulse 3-4× smaller afterward until contact is restored.

## Integration status

What's actually verified in this repo vs. what still needs a physical run with the glasses:

**Firmware**
- [x] `platformio.ini` targets `esp32doit-devkit-v1` with the right lib versions (read the code).
- [ ] Builds clean, boots with both sensors ready, advertises -- needs the board on hand.
- [ ] Halts cleanly if a sensor is missing -- needs the board on hand.

**Bridge**
- [x] Unpacks `<III6h>` records, detects dropped/bad packets, converts units (code path
      shared with `ble_logger.py`, exercised by `hw/bridge/convert_firmware_csv.py` against
      the three real recordings).
- [x] `t = origin + idx × 10` (the converter's batch equivalent of `live_bridge.py`'s live
      rule); consecutive samples 10 ms apart, verified on all three recordings.
- [x] Emits Contract A JSON lines with `t, ppg, red, ax, ay, az, gx, gy, gz`, missing = null
      -- validated against `tools/serial_check.py`'s own checker.
- [x] Saves a JSON-lines file per run for replay -- `live_bridge.py` and the converter both do.
- [ ] `live_bridge.py` itself hasn't been run against real hardware yet (no BLE device
      available in this environment) -- the record-unpacking/unit-conversion logic it shares
      with `ble_logger.py` has been exercised on real captures, but the live connect/stream
      loop is unverified end-to-end.
- [ ] Reconnect-after-restart without resetting the ESP32 -- needs the board on hand.

**System**
- [x] Real recorded sessions produce plausible HR through `ai/processing.py` (73-83 bpm
      across the three recordings, quality up to 1.0 in regular-rhythm mode).
- [ ] Same sessions *through the pipeline with the demo patient* are mostly not trusted: that
      patient has atrial fibrillation, so the pipeline uses irregular-rhythm quality scoring,
      and `ear_motion1.csv` never reaches the 0.6 trust threshold there (max 0.32). Regular-rhythm
      mode: 17 of 30 windows pass. Needs a P2 decision on whether AF mode is too strict for
      ear-crease PPG.
- [ ] `ear_motion1.csv`'s accelerometer reads a steady ~2 g magnitude from about 30 s onward
      while the gyro is moving. Looks like an IMU fault, not motion; check the sensor before
      trusting the IMU in that segment. PPG is unaffected.
- [ ] 5-minute live run on the power bank; visible pulse while worn; motion/PPG correlation;
      AirPods HR comparison -- all need the physical glasses and a person wearing them.
- [ ] Team informed of the BLE-not-serial, 100 Hz-not-50 Hz deviations from `CONTRACTS.md`'s
      literal Contract A text, and of the additive `red` field -- flagging it here; CONTRACTS.md
      itself is frozen and untouched.

## Open questions (unresolved, carried over from the build log)

- **Delivery method to P2/P3**: stdout pipe, local websocket, or HTTP to the backend?
  `live_bridge.py` writes to stdout today (simplest, pipeable into whatever gets decided) --
  not a final answer.
- **Shared Python version**: this repo's own setup is on 3.12; not yet a team-wide decision.
