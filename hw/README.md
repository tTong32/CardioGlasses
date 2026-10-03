# Hardware (P1)

Firmware is not in this scaffold. The glasses will stream one JSON object per line over USB serial at about 50 Hz. Those lines are Contract A in `../CONTRACTS.md`, and the replay CSVs in `../data/` use the same columns.

```json
{"t": 1760000000123, "ppg": 51234, "ax": 0.02, "ay": -0.98, "az": 0.10, "gx": 0.5, "gy": 0.1, "gz": 0.0}
```

- `t` is Unix milliseconds.
- Assumed baud rate is 115200 until the board is chosen. Set the port in `SERIAL_PORT` (see `.env.example`).
- Unknown values are `null`, never left out of the object.

TODO: firmware that reads the PPG sensor and IMU and writes these lines.
