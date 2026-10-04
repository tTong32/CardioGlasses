# CardioGlasses

Smart glasses that monitor heart patients continuously, comparing real-time PPG (pulse) and IMU (motion) data against personal baselines and clinical records to provide contextual alerts via voice and phone dashboard.

**24-hour hackathon project** • Python + FastAPI + vanilla JS • ElevenLabs voice • FinchNode health records

## Quick Demo

**Single-command demo** (starts backend + runs 3-minute scenario with clinical reasoning):

```bash
./run_script/demo.sh
```

Opens `http://localhost:8000` and plays a realistic recovery alert scenario:
- **0-7s:** Baseline calibration (resting ~72 BPM)
- **7-10s:** Walking (~105 BPM)
- **10-18s:** Slow recovery phase (exponential decay, tau~90s)
- **~12.2s:** 🚨 **NOTIFY alert** triggers (slow recovery detected)

Shows: EMR → Clinical reasoning → Live monitoring → Alert with "Are you OK?" check-in

## What Works

✅ **PPG Processing:** Bandpass filtering, beat detection, IBI extraction, honest quality scoring
✅ **Activity Detection:** IMU → resting/moving classification (2s windows)
✅ **Baseline Calibration:** 60s continuous rest required, provisional clinic HR support
✅ **Recovery Model:** Exponential decay curve fitting (physics-based, not ML)
✅ **Clinical Adaptation:** Risk tier, atrial fibrillation mode, rate-control meds, clinic HR
✅ **Alert Levels:** Rules-based decision engine (normal/monitor/notify/escalate)
✅ **Voice Alerts:** ElevenLabs TTS with template fallbacks
✅ **Phone Dashboard:** Real-time WebSocket, color-coded panels, check-in flow
✅ **Safety Net:** "Are you OK?" prompts, iMessage notifications via Photon
✅ **FinchNode Integration:** Live health record API with local fallback
✅ **Tests:** 189/197 passing (7 LLM tests fail due to missing SDK in test env)

⚠️ **Recovery Cutoffs:** Synthetic data (n=200 mock episodes, not PhysioNet)
⚠️ **Live Hardware:** Firmware exists, BLE bridge works, but not integrated into backend pipeline

## Architecture

```
Hardware (ESP32 + MAX30102 + MPU6050) → Contract A (100Hz PPG+IMU JSON)
                    ↓
AI Pipeline (Python) → Contract B/C (Readings every ~2s, Alerts on change)
                    ↓
Backend (FastAPI) + Dashboard (Web) → Voice + Phone UI
                    ↑
              FinchNode API (EMR)
```

### Contracts (Frozen)

**Contract A** (Hardware → AI): `{"t": unix_ms, "ppg": int, "ax": g, "ay": g, "az": g, "gx": deg/s, "gy": deg/s, "gz": deg/s}` at ~100Hz

**Contract B** (AI → Software): `Reading` every ~2s with HR, activity, quality, baseline deviation, recovery metrics

**Contract C** (AI → Software): `Alert` on level change only (normal/monitor/notify/escalate)

**Contract D** (Software → AI): `PatientContext` with age, conditions, medications, risk tier, config

See `CONTRACTS.md` for full spec.

## Setup

Python 3.11+ required:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Optional API keys (system works without them via fallbacks):
- `ELEVENLABS_API_KEY`: Voice synthesis (falls back to pre-generated audio)
- `GEMINI_API_KEY`: Alert wording (falls back to templates)
- `PHOTON_IMESSAGE_ADDRESS` + `PHOTON_IMESSAGE_TOKEN`: Caregiver iMessage (screen-only without)

## Run

### Complete Demo (Recommended)

```bash
./run_script/demo.sh
```

Automated: starts backend, opens browser, runs demo CSV with clinical explanations, cleanup.

### Manual Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Dashboard: `http://localhost:8000`
Caregiver view: `http://localhost:8000/caregiver`

### Replay Recorded Data

```bash
# Real hardware recordings (3 available: ear_motion1, finger_test, imu_test)
python -m ai.pipeline data/rec_ear_motion1.csv --speed 5

# Synthetic scenarios
python -m ai.pipeline --scenario elevated_rest --speed 10
python -m ai.pipeline --scenario slow_recovery --dry-run --explain
```

Available scenarios: `rest`, `moving`, `noisy`, `elevated_rest`, `af_elevated_rest`, `slow_recovery`, `normal_recovery`

Add `--no-llm` for template wording (no Gemini API key needed)
Add `--explain` to show EMR and clinical decision-making process

### Demo Simulator (UI Testing)

Posts scripted Readings/Alerts to backend without running full AI pipeline:

```bash
python -m tools.demo_sim --list  # show available scenarios
python -m tools.demo_sim         # default: rest → exertion → monitor → notify → recovery
python -m tools.demo_sim --scenario exercise --speed 3
```

### Tests

```bash
python -m pytest                    # all tests (189 pass, 7 LLM tests fail)
python -m pytest ai/tests/ -v      # AI pipeline only (40 tests)
python -m pytest tests/ -v         # backend only
```

## Key Features

### Clinical Record Shapes Monitoring

From FinchNode API (`patient-demo-polypharmacy`: 78yo, atrial fibrillation, heart failure, on metoprolol):

**Risk Tier** → Alert Sensitivity:
- **High:** 2.0σ above baseline for 30s → alert
- **Medium:** 2.5σ for 45s
- **Low:** 3.0σ for 60s

**Atrial Fibrillation** → Irregular rhythm mode (6 bpm baseline floor vs 4 bpm, different beat detection)

**Rate-Control Meds** (metoprolol, etc.) → Recovery compared to "meds" cohort, exercise zone = resting + 20-30 bpm

**Clinic HR** → Provisional baseline until 60s rest calibrates real baseline (8 bpm floor)

### Alert Levels

**NORMAL:** Deviation < 1σ for 20s (default state)

**MONITOR:** Deviation ≥ trigger for 10s (early warning)
**NOTIFY:** Deviation ≥ trigger for persist_s (starts "Are you OK?" check-in)
**ESCALATE:** Deviation ≥ 2× trigger AND persist ≥ 3× persist_s (severe sustained elevation)

Recovery verdicts override timing:
- `slow` recovery → MONITOR immediately
- `very_slow` recovery → NOTIFY/ESCALATE immediately

Post-exertion grace period: 60s after movement (no alerts during normal recovery)

### Safety Net

Notify/escalate alerts trigger 60-second "Are you OK?" prompt:
- **"I'm OK" button:** Closes quietly, no notification
- **"I need help" button:** Notifies caregiver screen + iMessage (if configured)
- **No response:** Same as "I need help" after timeout
- **Return to normal:** Closes quietly

Head gestures (nod/shake) can answer via IMU if hardware connected.

### Recovery Model

Fits exponential decay: `HR(t) = baseline + (HR0 - baseline) × e^(-t/τ)`

**Not machine learning** — physics-based curve fitting with `scipy.optimize.curve_fit`

Compares tau (time constant) against population percentiles:
- **Cutoffs:** Currently synthetic (n=200 mock episodes from `ai/train/mock_data.py`)
- **Intended data:** PhysioNet frailty dataset (download failed, see `ai/model/recovery_cutoffs.json`)
- **Verdict logic:** slow = tau/reference ≥ 1.5, very_slow ≥ 2.5

Dual verdict: population percentile + personal reference tracking

## Hardware Status

**Firmware:** ESP32 with MAX30102 (PPG) + MPU6050 (IMU), PlatformIO build, BLE streaming
**Recordings:** 3 real hardware sessions captured (ear placement, 100Hz, Contract A format)
**Integration:** Firmware → BLE bridge → Contract A JSON works, but **not streaming to backend live**

Replay works (`python -m ai.pipeline data/rec_ear_motion1.csv`), live streaming planned but incomplete.

See `hw/README.md` for firmware details and `hw/bridge/` for BLE → Contract A conversion.

## File Organization

```
ai/                     # AI pipeline (PPG processing, activity, baseline, recovery, alerts)
├── pipeline.py         # Main orchestrator
├── processing.py       # PPG → HR + quality
├── activity.py         # IMU → resting/moving
├── baseline.py         # Calibration tracker
├── recovery.py         # Exponential decay model
├── decision.py         # Alert rules engine
├── explainer.py        # LLM/template wording
├── clinical.py         # Clinical record → monitoring rules
├── model/              # Recovery cutoffs (synthetic)
└── tests/              # 40 tests (37 pass, 3 LLM fail)

backend/                # FastAPI server
├── main.py             # API endpoints + WebSocket
├── finchnode.py        # FinchNode → PatientContext
├── voice.py            # ElevenLabs TTS
└── caregiver.py        # Safety net + iMessage

web/
└── index.html          # Phone dashboard (1,767 lines vanilla JS)

hw/                     # Hardware (ESP32 firmware + BLE bridge)
├── firmware/           # PlatformIO (C++)
└── bridge/             # Python/bleak BLE → Contract A

data/
├── patient.json        # Base config (thresholds)
├── finchnode_cache.json # FinchNode fallback
├── demo_2min_elevated_rest.csv # Demo scenario (18,000 samples)
├── rec_*.csv           # Real hardware recordings (3)
└── audio/              # TTS cache + fallbacks

run_script/
└── demo.sh             # One-command demo

tools/
├── live_stream.py      # Serial → AI → backend (planned)
├── demo_sim.py         # Scripted UI testing
└── imessage/           # Photon iMessage sender
```

## Clinical Decision Points

All locations where conditions, medications, or risk tier change thresholds:

1. **Risk tier sensitivity** (`ai/clinical.py:19-23`): High/medium/low → deviation trigger & persistence time
2. **Irregular rhythm mode** (`ai/clinical.py:24-27`): AF/flutter/arrhythmia → 6 bpm baseline floor vs 4 bpm
3. **Rate-control medications** (`ai/clinical.py:31-34`): metoprolol, etc. → recovery comparison group, exercise zone
4. **Provisional baseline** (`ai/clinical.py:29-30`): Clinic HR → 8 bpm floor until calibration
5. **Alert escalation** (`ai/decision.py:27-31`): Post-exertion grace (60s), monitor (10s), notify (30-60s), escalate (90-180s)
6. **Recovery cutoffs** (`ai/model/recovery_cutoffs.json`): Percentiles for slow/very_slow verdicts

## Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| PPG Processing | ✅ Working | Bandpass, peaks, IBI, quality |
| Activity Detection | ✅ Working | IMU → resting/moving |
| Baseline Calibration | ✅ Working | 60s rest, provisional clinic HR |
| Recovery Model | ⚠️ Partial | Fitting works, cutoffs synthetic (n=200) |
| Decision Engine | ✅ Working | Rules-based, 189/197 tests pass |
| Clinical Adaptation | ✅ Working | Risk tier, AF, meds all functional |
| FinchNode API | ✅ Working | Live with fallback chain |
| ElevenLabs Voice | ✅ Working | Live synthesis + fallbacks |
| Photon iMessage | ⚠️ Code ready | Needs env vars configured |
| Gemini Wording | ⚠️ Partial | Template fallback works (7 LLM tests fail) |
| Phone Dashboard | ✅ Working | Full UI, WebSocket, check-ins |
| Caregiver Safety Net | ✅ Working | Check-in flow, buttons, timeouts |
| Hardware Recordings | ✅ Working | 3 real sessions, replay tested |
| Live Hardware Stream | ❌ Planned | Firmware exists, backend integration incomplete |

## Contributing

This was a 24-hour hackathon project. Track ownership:
- `hw/` — Hardware team
- `ai/` — AI team
- `backend/` + `web/` — Software team

`CONTRACTS.md` is frozen. Don't change contracts without team announcement.

Git workflow: Stay on `main`, small commits, pull before push, edit only your folder.

## License

MIT (24-hour hackathon project)
