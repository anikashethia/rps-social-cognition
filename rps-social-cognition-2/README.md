# RPS Social Cognition (v2)

Full-stack web app for the Rock-Paper-Scissors social cognition task. React + jsPsych 8 frontend, FastAPI + SQLite backend.

## Setup

### Backend

```bash
cd backend
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
```

The API server starts on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Modes

| Mode | Trials/block | Scanner triggers | Use case |
|------|-------------|------------------|----------|
| **dev** | 5 | No | Fast local testing |
| **behavioral** | 40 | No | Full Buergi et al. timings, behavioral session outside scanner |
| **scanner** | 40 | Yes — F8 TR trigger listener, waiting-for-scanner screen, session-local clock anchored to first TR | In-scanner fMRI session |

## Architecture

```
Browser (React + jsPsych)  ──HTTP──▶  FastAPI  ──▶  SQLite (rps.db)
```

- Frontend sends trial data to the backend after each trial
- Backend writes every trial, trigger, and session event to SQLite
- No data lives only in the browser — everything is persisted server-side trial-by-trial

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions/{id}` | Get session details |
| `PATCH` | `/api/sessions/{id}/anchor` | Set anchor timestamp (first TR) |
| `POST` | `/api/sessions/{id}/trials` | Log a trial |
| `POST` | `/api/sessions/{id}/triggers` | Log a scanner trigger |
