"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import create_all
from .routers import sessions, trials, triggers, rotations, participants

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    create_all()
    yield


app = FastAPI(
    title="RPS Social Cognition",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(sessions.router, prefix="/api")
app.include_router(trials.router, prefix="/api")
app.include_router(triggers.router, prefix="/api")
app.include_router(rotations.router, prefix="/api")
app.include_router(participants.router, prefix="/api")


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}
