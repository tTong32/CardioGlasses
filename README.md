# CardioGlasses

CardioGlasses is a 24-hour hackathon project: glasses with a PPG pulse sensor and an IMU stream raw samples to a laptop.
Python turns those samples into heart rate, activity, and signal quality, then compares them with a personal baseline and a synthetic clinical record.
It decides normal, monitor, notify, or escalate, speaks an ElevenLabs voice alert, and shows the explanation on a phone-friendly dashboard.

## Ownership

- `hw/` — P1 Hardware
- `ai/` — P2 AI
- `backend/` and `web/` — P3 Software

`CONTRACTS.md` is frozen. `ai/contracts.py` is the shared model source (the backend imports it). Don't change either without telling everyone.

## Setup

Python 3.11+. From the repo root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Run

Backend, so a phone on the same Wi-Fi can open `http://<laptop-ip>:8000/`:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Replay a Contract A recording:

```bash
python -m ai.replay data/sample_rest.csv --speedup 10
```

`python -m ai.replay --generate` rewrites `data/sample_rest.csv` and `data/sample_elevated.csv`. `python -m ai.pipeline data/sample_rest.csv --speedup 10` posts readings to the backend; the signal-processing stubs still return null.

## Git workflow

Stay on `main`. Make small commits, pull before you push, and only edit your own folder.
