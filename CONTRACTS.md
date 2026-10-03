# CONTRACTS

Time = Unix ms

A — Hardware → AI: one JSON line per sample over USB serial, ~50 Hz; same columns in replay CSV
{"t": 1760000000123, "ppg": 51234, "ax": 0.02, "ay": -0.98, "az": 0.10, "gx": 0.5, "gy": 0.1, "gz": 0.0}

B — AI → Software, `Reading`, every ~2 s
{"t": 1760000002000, "hr": 78, "ibi_ms": [790, 772, 765], "activity": "resting|moving", "quality": 0.91, "baseline_hr": 68, "deviation": 1.8, "persist_s": 45}

C — AI → Software, `Alert`, only on state change
{"t": 1760000050000, "level": "normal|monitor|notify|escalate", "confidence": 0.8, "headline": "Elevated heart rate at rest", "reason": "…compared with your baseline and history…", "voice_text": "Your heart rate has stayed high while you're resting. Please sit down and check your phone.", "next_step": "Sit down and review", "reading": { … }}

D — Software → AI, `PatientContext`, loaded at start
{"patient_id": "demo-1", "age": 71, "conditions": ["…"], "medications": ["…"], "risk_tier": "high", "config": {"min_quality": 0.6, "persist_s": 30, "deviation_trigger": 2.0, "cooldown_s": 120}}

Rules:
- unknown/missing values = null, never omitted
- `level` is exactly one of the four strings
- low signal quality → no alert, and `quality` is reported honestly
- don't change a contract without telling everyone
