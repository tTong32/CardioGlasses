"""CardioGlasses hub: store readings and alerts, serve the dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai import clinical
from ai.contracts import Alert, PatientContext, Reading
from backend import finchnode, voice

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

WEB_DIR = ROOT / "web"
PATIENT_PATH = ROOT / "data" / "patient.json"
PATIENT_CACHE_PATH = ROOT / "data" / "finchnode_cache.json"
VOICE_TIMEOUT_S = 10
AUDIO_NAME = re.compile(r"(tts_[0-9a-f]{16}|fallback_[a-z]+|test)\.mp3")

log = logging.getLogger(__name__)

FALLBACK_PATIENT = {
    "patient_id": "demo-1",
    "age": 71,
    "conditions": ["hypertension", "type 2 diabetes"],
    "medications": ["metoprolol", "metformin"],
    "risk_tier": "high",
    "config": {
        "min_quality": 0.6,
        "persist_s": 30,
        "deviation_trigger": 2.0,
        "cooldown_s": 120,
    },
}


def local_patient() -> PatientContext:
    """The checked-in patient (or the hardcoded one). Its config holds the base thresholds."""
    if PATIENT_PATH.is_file():
        return PatientContext.model_validate_json(PATIENT_PATH.read_text(encoding="utf-8"))
    return PatientContext.model_validate(FALLBACK_PATIENT)


def with_record_config(context: PatientContext, base: PatientContext) -> PatientContext:
    """Contract D thresholds: the base file's settings, tuned to this record's risk tier."""
    return context.model_copy(update={"config": clinical.tuned_config(base.config, context.risk_tier)})


class PatientState:
    """Current Contract D context and where it came from: finchnode, cache, or local."""

    def __init__(self) -> None:
        self.context: PatientContext = local_patient()
        self.source = "local"
        self.error: str | None = None
        self.loaded_at = time.time()

    def load(self) -> None:
        local = local_patient()
        if os.environ.get("FINCHNODE_ENABLED", "1") == "0":
            self.context, self.source, self.error = with_record_config(local, local), "local", "FinchNode disabled"
        else:
            try:
                self.context = with_record_config(finchnode.fetch_patient_context(local.config), local)
                self.source, self.error = "finchnode", None
                PATIENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                PATIENT_CACHE_PATH.write_text(self.context.model_dump_json(indent=2), encoding="utf-8")
            except Exception as exc:
                log.warning("FinchNode unavailable, using fallback patient: %s", exc)
                self.error = f"FinchNode unavailable: {exc}"
                cached = self._cached(local)
                if cached is not None:
                    self.context, self.source = cached, "cache"
                else:
                    self.context, self.source = with_record_config(local, local), "local"
        self.loaded_at = time.time()

    @staticmethod
    def _cached(local: PatientContext) -> PatientContext | None:
        if not PATIENT_CACHE_PATH.is_file():
            return None
        try:
            cached = PatientContext.model_validate_json(PATIENT_CACHE_PATH.read_text(encoding="utf-8"))
        except ValueError:
            return None
        # Thresholds always come from the local file (tuned by tier), even for a cached record.
        return with_record_config(cached, local)


patient = PatientState()


def database_path() -> Path:
    url = os.environ.get("DATABASE_URL", "sqlite:///data/app.db")
    raw = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else "data/app.db"
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(database_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    database_path().parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for websocket in list(self.clients):
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.clients.discard(websocket)


hub = Hub()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    await asyncio.to_thread(patient.load)
    yield


app = FastAPI(title="CardioGlasses", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _insert(table: str, t: int, payload: str) -> None:
    if table not in {"readings", "alerts"}:
        raise ValueError(table)
    with connect() as conn:
        conn.execute(f"INSERT INTO {table} (t, payload) VALUES (?, ?)", (t, payload))


def _recent(table: str, limit: int) -> list[dict]:
    if table not in {"readings", "alerts"}:
        raise ValueError(table)
    with connect() as conn:
        rows = conn.execute(
            f"SELECT payload FROM {table} ORDER BY t DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/readings", response_model=Reading)
async def post_reading(reading: Reading) -> Reading:
    _insert("readings", reading.t, reading.model_dump_json())
    await hub.broadcast({"type": "reading", "data": reading.model_dump(mode="json")})
    return reading


async def _alert_audio(alert: Alert) -> tuple[str | None, str]:
    try:
        name, source = await asyncio.wait_for(
            asyncio.to_thread(voice.audio_for, alert.voice_text, alert.level),
            timeout=VOICE_TIMEOUT_S,
        )
    except Exception as exc:
        log.warning("Voice step failed: %s", exc)
        name = voice.fallback_name(alert.level)
        if not (voice.AUDIO_DIR / name).is_file():
            return None, "none"
        source = "fallback"
    return (f"/audio/{name}" if name else None), source


@app.post("/alerts", response_model=Alert)
async def post_alert(alert: Alert) -> Alert:
    _insert("alerts", alert.t, alert.model_dump_json())
    # Voice first so the sound and the card arrive together.
    audio_url, audio_source = await _alert_audio(alert)
    await hub.broadcast(
        {
            "type": "alert",
            "data": alert.model_dump(mode="json"),
            "audio_url": audio_url,
            "audio_source": audio_source,
        }
    )
    return alert


@app.get("/patient", response_model=PatientContext)
def get_patient() -> PatientContext:
    return patient.context


@app.post("/patient/refresh", response_model=PatientContext)
async def refresh_patient() -> PatientContext:
    await asyncio.to_thread(patient.load)
    await hub.broadcast({"type": "status", "data": status()})
    return patient.context


@app.get("/status")
def status() -> dict:
    """Where the patient came from and whether voice is set up, for the dashboard."""
    fallbacks = [level for level in voice.FALLBACK_TEXT if (voice.AUDIO_DIR / voice.fallback_name(level)).is_file()]
    return {
        "patient_source": patient.source,
        "patient_error": patient.error,
        "patient_loaded_at": int(patient.loaded_at * 1000),
        "record_effects": clinical.effects(patient.context),
        "voice_configured": voice.is_configured(),
        "fallback_audio_levels": fallbacks,
    }


@app.get("/readings", response_model=list[Reading])
def list_readings(limit: int = Query(50, ge=1, le=1000)) -> list[Reading]:
    return [Reading.model_validate(item) for item in _recent("readings", limit)]


@app.get("/alerts", response_model=list[Alert])
def list_alerts(limit: int = Query(50, ge=1, le=1000)) -> list[Alert]:
    return [Alert.model_validate(item) for item in _recent("alerts", limit)]


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)


@app.get("/audio/{name}")
def get_audio(name: str) -> FileResponse:
    if not AUDIO_NAME.fullmatch(name) or not (voice.AUDIO_DIR / name).is_file():
        raise HTTPException(status_code=404)
    return FileResponse(voice.AUDIO_DIR / name, media_type="audio/mpeg")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
