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

Check every API key with one call each, then pre-generate the offline voice clips:

```bash
python -m backend.check_keys
python -m backend.voice --fallbacks
```

The patient comes from the FinchNode demo API (`patient-demo-polypharmacy`, no key needed). If FinchNode is down the backend uses the last cached copy, then `data/patient.json`, and the dashboard says so. Thresholds always come from `data/patient.json`. Alerts posted to `/alerts` are voiced with ElevenLabs and played on the dashboard; tap **Enable sound** on the phone first.

Run the tests with `python -m pytest`.

### Demo simulator

Stands in for the AI pipeline: posts scripted Readings and Alerts to the running backend so you can rehearse the dashboard and voice, or fall back to it on demo day.

```bash
python -m tools.demo_sim --list                      # elevated, escalate, poor-signal, dropout, normal, exercise, auto
python -m tools.demo_sim                             # rest -> exertion -> monitor -> notify -> recovery (~4 min)
python -m tools.demo_sim --scenario poor-signal --speed 3
```

### Serial checker (hardware)

Checks the glasses' USB stream against Contract A (rate, gaps, bad lines, g vs m/s², flat PPG) and can record it for replay.

```bash
python -m tools.serial_check --list
python -m tools.serial_check --port COM5 --seconds 120 --out data/rec_rest.csv
```

### AI pipeline

Turns samples into Readings every 2 s (10 s analysis window): heart rate and beat intervals from the PPG, an honest 0-1 quality score, resting/moving from the IMU, a resting baseline (frozen during episodes), recovery after exertion, and a rules-based alert level. Gemini (`GEMINI_MODEL`, default `gemini-flash-lite-latest`) writes the wording; templates take over if it's slow or unavailable.

```bash
python -m ai.pipeline --scenario elevated_rest --speed 10        # synthetic, posts to the backend
python -m ai.pipeline --scenario slow_recovery --dry-run          # print only, no backend
python -m ai.pipeline data/rec_rest.csv --speed 5                 # a recording
```

Scenarios: `rest`, `moving`, `noisy`, `normal_recovery` (no alerts), `elevated_rest` and `af_elevated_rest` (monitor -> notify -> escalate -> normal), `slow_recovery` and `calibration_then_slow` (notify, then escalate for a regular-rhythm patient). Add `--no-llm` for templated wording.

The FinchNode record shapes monitoring (`ai/clinical.py`): the risk tier sets how big and how long a rise must be before an alert; atrial fibrillation in the record switches beat detection to irregular-rhythm mode (otherwise half of an AF patient's readings would count as bad signal); rate-control medication picks the recovery comparison group and informs the wording; clinic heart-rate readings give a starting "usual" until the glasses calibrate, so a high reading at switch-on isn't learned as normal. The dashboard lists these under Health record.

### Caregiver safety net, activity, smartwatch comparison

- **Safety net:** a notify/escalate alert asks the wearer "Are you OK?" on the phone (60 s, `CHECKIN_TIMEOUT_S`). No answer or "I need help" notifies the caregiver on `/caregiver` (open it on a second phone) and by iMessage through Photon when `CAREGIVER_PHONE`, `PHOTON_IMESSAGE_ADDRESS` and `PHOTON_IMESSAGE_TOKEN` are set. Install the sender once: `cd tools/imessage && npm install`.
- **Activity:** the Activity button starts a coached exercise. The zone comes from the record (on metoprolol: usual + 20 to 30); the coach speaks only when the wearer stays above it, then checks the 1-minute recovery. Moving around with a heart rate above the usual resting rate for `ACTIVITY_DETECT_S` (default 120 s; set about 20 before starting the server to rehearse faster) starts a quiet activity instead: no voice, and the phone offers "Make it exercise". Rehearse with `python -m tools.demo_sim --scenario exercise` or `--scenario auto` (set `WALK_RECOVERY_S` lower when speeding the recovery up). `/walk/*` is the same as `/activity/start` and `/activity/stop`.
- **Why not a smartwatch:** `/compare.html`, one slide comparing a fixed smartwatch rule with CardioGlasses on the same 16 minutes. Regenerate the data with `python -m tools.compare`.

## Git workflow

Stay on `main`. Make small commits, pull before you push, and only edit your own folder.
