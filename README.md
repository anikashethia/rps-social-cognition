# RPS Social Cognition Task

A Rock-Paper-Scissors task for studying social cognition and mentalization. Participants play RPS against four social agents (A–D) and a random baseline, while computational modeling (CHASE) captures trial-by-trial belief updating about opponent strategy.

Based on the CHASE model from Buergi, Aydogan, Konovalov & Ruff (2026, *Nature Neuroscience*).

The task is a full-stack web app: a FastAPI backend (sessions, trials, scanner triggers, SQLite storage) and a React + jsPsych frontend.

---

## Repository structure

```
rps-social-cognition/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entry point, CORS, routers
│   │   ├── models.py           # SQLAlchemy models: Session, Trial, Trigger
│   │   ├── database.py         # SQLite engine/session setup
│   │   ├── routers/            # sessions, trials, triggers, rotations endpoints
│   │   └── rotations/
│   │       └── rotation.json   # Counterbalanced avatar/condition rotation table (placeholder)
│   ├── rps.db                  # SQLite database — all session/trial/trigger data
│   └── .env.example            # DATABASE_URL, FRONTEND_ORIGIN
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Landing page: participant ID, mode, online/scanner launch
│   │   ├── timeline.ts         # jsPsych timeline builder (welcome → blocks → end)
│   │   ├── agents.ts           # Social agent behavior (CHASE-based) + RNG baseline
│   │   ├── api.ts              # Typed fetch wrappers for backend endpoints
│   │   ├── plugins/            # jsPsych plugins: RpsChoice, Feedback, Fixation
│   │   └── index.css           # Styling — black intro screens, white task screens
│   └── public/avatars/         # Avatar PNGs (16 placeholders, 2 sets × 4 ethnicities × 2 genders)
├── analysis/
│   ├── behavioral.py           # Model-free measures (win rate, entropy, WSLS)
│   ├── model_fitting.py        # MLE fitting pipeline, AIC model comparison
│   ├── parameter_recovery.py   # Simulation: how many trials/block for stable CHASE recovery?
│   ├── belief_updates.py       # CHASE belief update timeseries utilities
│   ├── data_io.py              # SQLite read helpers
│   └── plots.py                # Figures (in progress)
├── models/
│   ├── chase.py                # Full CHASE model
│   └── alternatives.py         # RL, FP, EWA, EWA-S, ToMk (stubs)
├── docs/
│   └── task_design.md
├── results/                    # Analysis outputs go here (gitignored)
└── requirements.txt            # Python deps for analysis/ (numpy, pandas, scipy, matplotlib)
```

---

## Ongoing Tasks

### MRI Compatibility
- [ ] TR trigger sync (using F8)
- [ ] Onset logging: every trial onset, response, and feedback event should be logged with a session-local t_ms
- [ ] Jittered ITIs
- [ ] Ensure current trial count and duration is around ~ 10 minutes or so
- [ ] Ensure task is fully keypoard-operable (not mouse)
- [ ] Event logging should be via SQLite DB: every trial needs trial onset t_ms, agent choice, outcome, and response time --> feeds CHASE model

### Buergi paper / CHASE model
- [ ] Confirm what output is needed for CHASE model and GLM
- [x] Confirm number of trials: 6 blocks of 40 trials

---

## Running the task

### Backend

```bash
cd backend
cp .env.example .env       # DATABASE_URL + FRONTEND_ORIGIN, defaults are fine for local dev
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

This creates `rps.db` (SQLite) on first run and serves the API at `http://localhost:8000/api`.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api → http://localhost:8000/api
```

### Using the landing page

1. Enter a **Participant ID**.
2. Choose **Mode**: `Test` runs 5 trials per block (for quick local checks), `Full` runs 35 trials per block.
3. Enter a **config index** (1–16, picks the avatar/condition rotation for that participant) and click **Start online version** or **Start scanner version**.
   - Online and scanner versions are functionally identical except the scanner version waits for an F8 trigger before starting and anchors trial timing to it.
4. Use keys **1**, **2**, **3** to play (Rock, Paper, Scissors) — the task is fully keyboard-operable.

---

## Task design

- **5 blocks**, one per agent (A, B, C, D, RNG), order counterbalanced across participants via Latin square
- **35 trials per block** in Full mode (~4–5 min per block)
- **Points**: +3 win, −3 lose, 0 draw, starting from 100
- **Agents**: all social agents (A–D) play at the same CHASE level (k=1) with calibrated noise; RNG is purely random
- Agent order is determined by the rotation config index, so the same participant/index always gets the same order

---

## Data output

All session, trial, and trigger data is written directly to the SQLite database at `backend/rps.db` (no CSV download). Tables:

**`sessions`** — `id`, `participant_id`, `session_number`, `mode`, `config_index`, `created_at`, `anchor_t_ms`

**`trials`** — `id`, `session_id`, `block`, `agent`, `trial_in_block`, `trial_global`, `participant_choice`, `agent_choice`, `outcome`, `points_delta`, `points_cumulative`, `rt_ms`, `onset_ms`, `iti_duration_ms`, `block_onset_ms`, `condition`

**`triggers`** — `id`, `session_id`, `tr_number`, `t_ms` (scanner TR pulses, scanner mode only)

---

## Analysis

Install dependencies:
```bash
pip install -r requirements.txt
```

**Behavioral (model-free):**
```bash
python analysis/behavioral.py --db_path backend/rps.db --output_dir results/behavioral/
```
Computes win rate, choice entropy, win-stay/lose-shift, and lag-1 autocorrelation per participant × agent, plus a monotonic gradient test (A > B > C > D > RNG).

**Model fitting (CHASE + alternatives):**
```bash
python analysis/model_fitting.py --db_path backend/rps.db --output_dir results/models/ --model chase,rl,fp
```
Fits models via MLE with random restarts, outputs per-participant parameters, trial-level CHASE estimates (belief updates, APE, choice values), and AIC model comparison.

**Parameter recovery (how many trials/block are needed?):**
```bash
python -m analysis.parameter_recovery --n_sims 100 --output_dir results/recovery/
```
Simulates synthetic CHASE data at trial counts `[10, 15, 20, 25, 30, 40]`, fits CHASE back to it, and reports recovery correlation/RMSE per parameter — used to decide the real trial count per block. Must be run as a module (`python -m analysis.parameter_recovery`, not `python analysis/parameter_recovery.py`) so `models/` is importable from repo root.

---

## Avatar images

Avatars live in `frontend/public/avatars/` and are referenced directly by filename (e.g. `s1f1.png`) from the rotation config — no separate path config needed. The current set is 16 placeholders (`s{1,2}{f,m}{1-4}.png`, 2 sets × 2 genders × 4 ethnicities — see `backend/app/rotations/rotation.json` for the naming convention) and will be replaced once the mentor's finalized rotation table is ready.

---

## Citation

Buergi, Aydogan, Konovalov & Ruff (2026). *Nature Neuroscience.*
