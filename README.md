# RPS Social Cognition Task

A Rock-Paper-Scissors task for studying social cognition and mentalization. Participants play RPS across 6 blocks against two agents they've had prior conversations with (one "friendly", one "neutral", assigned via IOS self-other overlap score from the companion chat task). Each agent plays at a different CHASE reasoning level per block (levels 0, 1, 2), counterbalanced across participants. Computational modeling (CHASE) captures trial-by-trial belief updating about opponent strategy.

Based on the CHASE model from Buergi, Aydogan, Konovalov & Ruff (2026, *Nature Neuroscience*). Bot behavior matches Buergi's `mn_RPS_task.m` exactly: same level-k reasoning hierarchy, same RW-freq attraction update rule, same adaptive WSLS noise structure.

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
│   │   ├── routers/            # sessions, trials, triggers, rotations endpoints
│   │   └── rotations/
│   │       └── rotation.json   # 6 counterbalanced block-order configs (2 agents × 3 CHASE levels)
│   ├── rps.db                  # SQLite database — all session/trial/trigger data
│   └── .env.example            # DATABASE_URL, FRONTEND_ORIGIN
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Landing page: participant ID, mode, launch
│   │   ├── timeline.ts         # jsPsych timeline builder (welcome → blocks → end)
│   │   ├── agents.ts           # CHASEAgent bot: level-k reasoning + adaptive WSLS noise
│   │   ├── api.ts              # Typed fetch wrappers for backend endpoints
│   │   ├── config.ts           # Single source of truth for all TypeScript constants
│   │   │                       # (twin of config.py — keep both in sync)
│   │   ├── plugins/            # jsPsych plugins: RpsChoice, Feedback, Fixation
│   │   └── index.css           # Styling — black intro screens, white task screens
│   └── public/avatars/         # Avatar PNGs
├── models/
│   └── chase.py                # CHASE participant model: simulate() + fit() with two-stage grid search
├── analysis/
│   ├── verify_agent.py         # Smoke test: runs one block of CHASEBot, checks output
│   ├── parameter_recovery_new_design.py  # Recovery sim for 6-block 2-agent design
│   ├── behavioral.py           # Model-free measures (win rate, entropy, WSLS)
│   └── data_io.py              # SQLite read helpers
├── reference/
│   └── buergi_chase_matlab/    # Buergi et al.'s original MATLAB source (reference only)
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
2. Choose **Mode**: `Test` (5 trials/block, quick dev check) or `Full` (40 trials/block, real session).
3. Enter a **config index** (1–6, picks the block-order counterbalancing for that participant) and click **Start online version** or **Start scanner version**.
   - Scanner version waits for an F8 trigger before starting and anchors trial timing to it.
4. Use keys **1**, **2**, **3** to play (Rock, Paper, Scissors).

---

## Task design

- **6 blocks** per participant: one friendly agent and one neutral agent (assigned via IOS self-other overlap from the chat task), each played at CHASE levels 0, 1, and 2 across three blocks each.
- **40 trials per block** in Full mode. At current timing (4s response window + ITI jittered 0–6s + 2s feedback ≈ 9s/trial average), that's ~36 minutes total.
- **Points**: +1 win, −1 lose, 0 draw.
- **Block order**: counterbalanced across 6 configs (`rotation.json`). All configs satisfy Buergi's constraints: first block is never level 2; no repeated level at the agent-boundary transition.
- **Agents**: two agents per participant (friendly and neutral), each playing all three CHASE levels. Attractions accumulate continuously across all blocks for the same agent — no reset between blocks — matching Buergi's `mn_RPS_task.m`.

---

## Bot behavior (CHASEAgent)

The bot (`frontend/src/agents.ts`) replicates Buergi et al.'s `mn_RPS_task.m` exactly:

- **Level-k reasoning**: k=0 plays softmax over own attraction history; k=1 best-responds to the participant's attraction history (no initial softmax, matching CHASE paper); k=2 adds one more recursion.
- **RW-freq attraction update**: dual trackers (`attr` for bot's own choices, `pAttr` for participant's choices), uniform initialization [1/3, 1/3, 1/3], delta rule with α=0.9. Accumulates continuously across all 6 blocks per agent.
- **Adaptive WSLS noise**: bot goes "noisy" (beta drops from 10 → 1e-3) on lose/tie streaks or sustained win streaks, matching `get_noise_level()` in Buergi's code. Noise-breaker suppresses repeated noisy actions.
- **Parameters** (from `config.py` / `config.ts`): α=0.9, β=10, λ=1.0, noise β=1e-3, time horizon=5, success criterion=0.5, skewness=1.3.
- **Seeded RNG** (mulberry32): reproducible bot behavior per session.

---

## CHASE participant model (`models/chase.py`)

Used post-hoc to fit participant behavior. Key implementation details:

- **Two-stage fitting** matching Buergi's `mn_fitModel.m`: Cartesian grid search (5^4 = 625 combinations per kappa) to identify promising starting points, then BFGS optimization in transformed parameter space (logit for α, log for β/γ/λ).
- **Parameter bounds**: α ∈ (0,1), β ∈ [0,100], γ ∈ [0.01,20], λ ∈ [0,100], κ ∈ {0,1,2,3,4}.
- **Full attraction history**: `CHASEResult` contains `own_attractions` and `opp_attractions` as T×3 arrays — complete trial-by-trial record of both trackers, matching Buergi's `f_mat_own`/`f_mat_other`.

---

## Data output

All session, trial, and trigger data is written to `backend/rps.db` (SQLite). Writes happen incrementally per trial so data survives a mid-session browser crash.

**`sessions`** — `id`, `participant_id`, `session_number`, `mode`, `config_index`, `created_at`, `anchor_t_ms`

**`trials`** — `id`, `session_id`, `block`, `agent`, `trial_in_block`, `trial_global`, `participant_choice`, `agent_choice`, `outcome`, `points_delta`, `points_cumulative`, `rt_ms`, `onset_ms`, `feedback_onset_ms`, `iti_duration_ms`, `block_onset_ms`, `condition`, `level`

Per-trial bot state (for model validation and analysis):
- `is_noisy` — was this a WSLS noisy trial?
- `noise_trigger` — which streak triggered noise: `"lose"` / `"tie"` / `"win"` / `null`
- `success_rate` — participant win rate over last 5 trials at time of noise decision
- `noise_breaker` — did the noise-breaker suppress a repeated noisy action?
- `bot_attr_r/p/s` — bot's own attraction vector [Rock, Paper, Scissors] before choice
- `p_attr_r/p/s` — bot's estimate of participant's attraction vector before choice

**`triggers`** — `id`, `session_id`, `tr_number`, `t_ms` (scanner TR pulses only)

### Clock model

All timestamps are session-local milliseconds (`performance.now() − anchor`). In scanner mode, the first F8 pulse is `t = 0`; in online mode, the anchor is set at task start. Trial events and TR pulses share the same zero point, so aligning behavioral data to scanner pulses is direct subtraction. `feedback_onset_ms` is computed as `onset_ms + response_window_ms` (deterministic — the choice screen always holds for the full response window).

---

## MRI compatibility

- [x] TR trigger sync (F8) — `tr_number` starts at 0 on first pulse, increments cleanly
- [x] Onset logging — `onset_ms`, `rt_ms`, `feedback_onset_ms`, `block_onset_ms` per trial
- [x] Jittered ITIs (uniform random within `iti_min`/`iti_max` per mode)
- [x] Fully keyboard-operable — all task screens use jsPsych keyboard plugins; mouse handlers only on the experimenter landing page

---

## Citation

Buergi, Aydogan, Konovalov & Ruff (2026). *Nature Neuroscience.*
