"""CardioGlasses hub: store readings and alerts, serve the dashboard."""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai.contracts import Alert, PatientContext, Reading

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

WEB_DIR = ROOT / "web"
PATIENT_PATH = ROOT / "data" / "patient.json"

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


@app.post("/alerts", response_model=Alert)
async def post_alert(alert: Alert) -> Alert:
    _insert("alerts", alert.t, alert.model_dump_json())
    await hub.broadcast({"type": "alert", "data": alert.model_dump(mode="json")})
    return alert


@app.get("/patient", response_model=PatientContext)
def get_patient() -> PatientContext:
    # FinchNode replaces this file later.
    if PATIENT_PATH.is_file():
        return PatientContext.model_validate_json(PATIENT_PATH.read_text(encoding="utf-8"))
    return PatientContext.model_validate(FALLBACK_PATIENT)


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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
