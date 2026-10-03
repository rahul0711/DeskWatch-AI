"""FastAPI app for the new attendance system (SCRFD + ByteTrack + AdaFace +
FAISS). Separate from app/web/server.py (the legacy YOLO/facenet/ArcFace
multi-detector app) on purpose -- that app's lifespan spins up a subprocess
per detector per camera on startup, which is unrelated GPU/CPU load we don't
want fighting this system's own SCRFD/AdaFace sessions for VRAM while
developing/demoing this one.

Run (from the project root, with the venv active):
    source .venv/bin/activate
    uvicorn app.web.attendance_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.attendance.config import load_attendance_config
from app.attendance.db import init_db
from app.attendance.events import event_bus
from app.attendance.pipeline import get_detector, get_embedder, get_index, init_pipeline
from app.attendance.worker import RecognitionWorker
from app.web.attendance_routes import router as attendance_router
from app.web.auth import DEMO_TOKEN, check_credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("attendance_web")

ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_attendance_config()
    logger.info("Initializing database at %s", cfg.database_url.rsplit("@", 1)[-1] if cfg.database_url else cfg.db_path)
    init_db(cfg)
    logger.info("Loading detection/recognition models (this can take a few seconds)...")
    init_pipeline(cfg)

    broadcast_task = asyncio.create_task(event_bus.run_broadcast_loop())
    worker = RecognitionWorker(cfg, get_detector(), get_embedder(), get_index())
    worker.start()
    logger.info("Attendance server ready.")

    yield

    worker.stop()
    broadcast_task.cancel()


app = FastAPI(title="Attendance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginRequest) -> dict:
    if not check_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": DEMO_TOKEN}


app.include_router(attendance_router)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket, token: str | None = None):
    if token != DEMO_TOKEN:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    event_bus.add_client(websocket)
    try:
        while True:
            await websocket.receive_text()  # client sends nothing meaningful; just detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.remove_client(websocket)

# Serve only the image folders -- mounting all of data/ exposed attendance.db publicly.
_cfg = load_attendance_config()
for _dir in (_cfg.enrollment_dir, _cfg.attendance_crops_dir):
    _dir.mkdir(parents=True, exist_ok=True)
    app.mount(f"/data/{_dir.name}", StaticFiles(directory=_dir), name=f"data_{_dir.name}")

if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info("Serving built frontend from %s", _FRONTEND_DIST)
else:
    logger.info("No frontend build at %s -- run `npm run build` in frontend/, or use `npm run dev` separately", _FRONTEND_DIST)
