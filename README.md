# RPS Social Cognition Task

A Rock-Paper-Scissors task for studying social cognition and mentalization. Participants play RPS across 6 blocks against two agents they've had prior conversations with (one "friendly", one "neutral", assigned via IOS self-other overlap score from the companion chat task). Each agent plays at a different CHASE reasoning level per block (levels 0, 1, 2), counterbalanced across participants. Computational modeling (CHASE) captures trial-by-trial belief updating about opponent strategy.

Based on the CHASE model from Buergi, Aydogan, Konovalov & Ruff (2026, *Nature Neuroscience*). Bot behavior matches Buergi's `mn_RPS_task.m` exactly — verified to machine epsilon (1.1e-16) by direct comparison of probability vectors and attraction tracker states across all levels and trials.

The task is a full-stack web app: a FastAPI backend (sessions, trials, scanner triggers, SQLite storage) and a React + jsPsych frontend.

---

## Repository structure

```
rps-social-cognition/
├── config.py                   # Single source of truth for all Python constants
│                               # (mirrors frontend/src/config.ts and Buergi's mn_RPS_config.m)
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entry point, CORS, routers
│   │   ├── models.py           # SQLAlchemy models: Session, Trial, Trigger
│   │   ├── database.py         # SQLite engine/session setup
│   │   ├── routers/            # sessions, trials, triggers, rotations, participants endpoints
│   │   └── rotations/
│   │       └── rotation.json   # 6 counterbalanced block-order configs (2 agents × 3 CHASE levels)
│   ├── test_e2e.py             # Backend e2e test (88/88 checks) — run from backend/
│   └── .env.example            # DATABASE_URL, FRONTEND_ORIGIN
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Landing page: participant ID, avatar registration, mode, launch
│   │   ├── timeline.ts         # jsPsych timeline builder (welcome → blocks → end)
│   │   ├── agents.ts           # CHASEAgent bot: level-k reasoning + adaptive WSLS noise
│   │   ├── api.ts              # Typed fetch wrappers for backend endpoints
│   │   ├── config.ts           # Single source of truth for all TypeScript constants
│   │   │                       # (twin of config.py — keep both in sync)
│   │   └── plugins/            # jsPsych plugins: RpsChoice, Feedback, Fixation
│   └── public/avatars/         # 16 named avatar PNGs (r{rack}_{gender}_{name}.png)
├── analysis/
│   ├── verify_agent.py         # 19-check behavioral verification of CHASEAgent vs expected CHASE behavior
│   ├── compare_ts_matlab.mjs   # Node.js: runs agents.ts math on fixed choice sequence → ts_probs.json
│   ├── compare_ts_matlab.m     # MATLAB: runs mn_RPS_task.m math on same sequence → matlab_probs.json
│   ├── compare_ts_matlab.py    # Python: asserts ts_probs.json == matlab_probs.json to 1e-10
│   ├── plot_bot_comparison.py  # Figure: TypeScript vs MATLAB attraction tracker values
│   ├── run_buergi_recovery.m   # MATLAB: parameter recovery using Buergi's mn_sim + mn_fit pipeline
│   ├── export_for_matlab.py    # SQLite → CSV export in Buergi's behavioral_data.mat format
│   ├── load_for_buergi.m       # CSV → behavioral_data.mat for mn_fit
│   └── generate_recovery_data.py  # (unused) Python CHASEBot recovery — superseded by MATLAB pipeline
├── models/
│   └── chase.py                # Python CHASE model (reference only — analysis uses Buergi's MATLAB)
└── requirements.txt            # Python deps (numpy, pandas, scipy, matplotlib)
```

---

## Running the task

### Backend

```bash
cd backend
cp .env.example .env       # DATABASE_URL + FRONTEND_ORIGIN, defaults are fine for local dev
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Creates `rps.db` (SQLite) on first run. API at `http://localhost:8000/api`.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api → http://localhost:8000/api
```

### Using the landing page

1. Enter a **Participant ID**.
2. Enter the **friendly** and **neutral avatar IDs** from the IOS chat task (e.g. `r1_f_quinn`, `r2_m_charlie`) and click **Register**.
3. Choose **Mode**: `Test` (5 trials/block, quick dev check) or `Full` (40 trials/block, real session).
4. Click **Start behavioral** or **Start scanner**.
   - Scanner version waits for an F8 trigger before starting and anchors trial timing to it.
5. Use keys **1**, **2**, **3** to play (Rock, Paper, Scissors).

---

## Task design

- **6 blocks** per participant: one friendly agent and one neutral agent (assigned via IOS self-other overlap from the chat task), each played at CHASE levels 0, 1, and 2 across three blocks each.
- **40 trials per block** in Full mode. At current timing (3s response window + ITI jittered 0–6s + 2s feedback ≈ 8s/trial average), that's ~32 minutes total.
- **Points**: +1 win, −1 lose, 0 draw.
- **Block order**: counterbalanced across 6 configs (`rotation.json`). All configs satisfy Buergi's constraints: first block is never level 2; no consecutive blocks share the same level.
- **Agents**: two agents per participant (friendly and neutral), each playing all three CHASE levels. Attractions accumulate continuously across all blocks for the same agent — no reset between blocks — matching Buergi's `mn_RPS_task.m`.

---

## Bot behavior (CHASEAgent)

The bot (`frontend/src/agents.ts`) replicates Buergi et al.'s `mn_RPS_task.m` exactly. Verified by direct numerical comparison: given an identical 40-trial choice sequence, TypeScript and MATLAB produce the same probability vectors and attraction tracker states to machine epsilon (max diff = 1.1e-16) at all three levels.

- **Level-k reasoning**: k=0 plays softmax over own attraction history; k=1 best-responds to the participant's attraction history (no initial softmax, matching CHASE paper); k=2 adds one more recursion.
- **RW-freq attraction update**: dual trackers (`attr` for bot's own choices, `pAttr` for participant's choices), uniform initialization [1/3, 1/3, 1/3], delta rule with α=0.9. Accumulates continuously across all 6 blocks per agent.
- **Adaptive WSLS noise**: bot goes "noisy" (beta drops from 10 → 1e-3) on lose/tie streaks or sustained win streaks, matching `get_noise_level()` in Buergi's code. Noise-breaker suppresses repeated noisy actions.
- **Parameters** (from `config.py` / `config.ts`): α=0.9, β=10, λ=1.0, noise β=1e-3, time horizon=5, success criterion=0.5, skewness=1.3.
- **Seeded RNG** (mulberry32): reproducible bot behavior per session.

---

## Analysis pipeline

Participant data is fit using Buergi et al.'s MATLAB pipeline (`mn_fit` / `mn_fitModel` with `fminunc`). The Python `models/chase.py` is kept as reference only.

### Fitting workflow (post-data-collection)

```
backend/rps.db
  → python3 analysis/export_for_matlab.py   # SQLite → analysis/buergi_export.csv
  → analysis/load_for_buergi.m (MATLAB)     # CSV → buergi_chase/data/behavioral_data.mat
  → mn_fit (MATLAB, Buergi repo)            # fit CHASE per participant per condition
```

Fit friendly and neutral blocks separately (3 blocks × 40 trials = 120 trials per fit). Primary analysis: compare kappa between friendly and neutral conditions.

### Parameter recovery

Recovery was run using Buergi's fMRI participants' fitted parameters (48 subjects, kappa 0–2, 120 trials per agent) to validate the 3-block design:

| Parameter | r (our design) | r (Buergi 240-trial) | Notes |
|-----------|---------------|----------------------|-------|
| kappa     | 1.00          | 1.00                 | Primary measure — excellent |
| beta      | 0.86          | 0.88                 | Good |
| alpha     | 0.74          | 0.80                 | Good |
| gamma     | 0.40          | 0.73                 | Weak — kappa=2 only (n=48) |
| lambda    | −0.21         | 0.86                 | Unreliable — collinear with beta at 120 trials |

Lambda and gamma are not interpreted per-participant. Analysis focuses on kappa, beta, and alpha.

---

## Data output

All session, trial, and trigger data is written to `backend/rps.db` (SQLite). Writes happen incrementally per trial so data survives a mid-session browser crash.

**`sessions`** — `id`, `participant_id`, `session_number`, `mode`, `created_at`, `anchor_t_ms`

**`trials`** — `id`, `session_id`, `block`, `agent`, `trial_in_block`, `trial_global`, `participant_choice`, `agent_choice`, `outcome`, `points_delta`, `points_cumulative`, `rt_ms`, `onset_ms`, `feedback_onset_ms`, `iti_duration_ms`, `block_onset_ms`, `condition`, `level`

Per-trial bot state (for model validation):
- `is_noisy` — was this a WSLS noisy trial?
- `noise_trigger` — which streak triggered noise: `"lose"` / `"tie"` / `"win"` / `null`
- `success_rate` — participant win rate over last 5 trials at time of noise decision
- `noise_breaker` — did the noise-breaker suppress a repeated noisy action?
- `bot_attr_r/p/s` — bot's own attraction vector [Rock, Paper, Scissors] before choice
- `p_attr_r/p/s` — bot's estimate of participant's attraction vector before choice

**`triggers`** — `id`, `session_id`, `tr_number`, `t_ms` (scanner TR pulses only)

### Clock model

All timestamps are session-local milliseconds (`performance.now() − anchor`). In scanner mode, the first F8 pulse is `t = 0`; in online mode, the anchor is set at task start. Trial events and TR pulses share the same zero point, so aligning behavioral data to scanner pulses is direct subtraction.

---

## MRI compatibility

- [x] TR trigger sync (F8) — `tr_number` starts at 0 on first pulse, increments cleanly
- [x] Onset logging — `onset_ms`, `rt_ms`, `feedback_onset_ms`, `block_onset_ms` per trial
- [x] Jittered ITIs (uniform random within `iti_min`/`iti_max` per mode)
- [x] Fully keyboard-operable — all task screens use jsPsych keyboard plugins

---

## Citation

Buergi, Aydogan, Konovalov & Ruff (2026). *Nature Neuroscience.*
