# RPS Social Cognition Task

A Rock-Paper-Scissors task for studying social cognition and mentalization. Participants play RPS across 8 blocks against agents they've had prior conversations with (in the companion chat task, split into friendly/neutral conditions) and control agents they haven't spoken with, while computational modeling (CHASE) captures trial-by-trial belief updating about opponent strategy.

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
│   │   ├── agents.ts           # Social agent behavior (CHASE-based); all 8 agents share strategy — the manipulation is condition (friendly/neutral/control), not agent skill
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
- [x] TR trigger sync (using F8) — verified in a live scanner-mode test: `tr_number` starts at 0 on the first pulse (the anchor), increments cleanly with no gaps/duplicates on repeated F8 presses
- [x] Onset logging: every trial onset, response, and feedback event is logged with a session-local t_ms (`onset_ms`, `rt_ms`, `feedback_onset_ms` columns on `trials`)
- [x] Jittered ITIs (uniform random, see `iti_min`/`iti_max` in `timeline.ts`)
- [ ] Ensure current trial count and duration is around ~ 10 minutes or so — current Full-mode setting (35 trials × 8 blocks) runs ~42 minutes. Parameter recovery results point to **25 trials/block (~30–33 min total)** as the best tradeoff (see below) — still over the ~10 min target, but not yet changed in code pending final sign-off
- [x] Ensure task is fully keyboard-operable (not mouse) — confirmed: every task screen uses jsPsych keyboard plugins or custom `keydown` listeners; the only mouse (`onClick`) handlers anywhere in the frontend are on the experimenter-facing landing page, not the task itself
- [x] Event logging via SQLite DB: every trial logs onset t_ms, agent choice, outcome, and response time (see [Data output](#data-output) below)

### Buergi paper / CHASE model
- [ ] Confirm what output is needed for CHASE model and GLM
- [x] Confirm trial count per block — **25 trials/block (~30–33 min total)** is the recommended setting; not yet applied in `timeline.ts` pending sign-off

`models/chase.py` originally used a single shared "attraction" history for both players wherever it needed to predict opponent behavior at a given reasoning level. Cross-checking against Buergi et al.'s original MATLAB source (`reference/buergi_chase_matlab/`) confirmed the published model tracks **two separate histories** (`f_mat_own` / `f_mat_other`) — one updated from the participant's own choices, one from the opponent's — and seeds even/odd reasoning levels from each respectively. Fixed in `models/chase.py` and `analysis/parameter_recovery.py`.

Parameter recovery (100 sims × 6 trial counts), before → after the fix:

| n_trials | alpha | beta | gamma | **kappa** | lambda |
|---|---|---|---|---|---|
| 10 | 0.61 → 0.55 | 0.47 → 0.58 | 0.13 → 0.07 | 0.40 → **0.70** | 0.31 → 0.40 |
| 15 | 0.55 → 0.50 | 0.53 → 0.69 | 0.05 → 0.10 | 0.38 → **0.54** | 0.06 → 0.33 |
| 20 | 0.54 → 0.65 | 0.55 → 0.66 | 0.12 → 0.31 | 0.57 → **0.76** | 0.34 → 0.29 |
| **25** | 0.76 → 0.76 | 0.64 → 0.57 | 0.06 → 0.33 | 0.61 → **0.84** | 0.30 → 0.39 |
| 30 | 0.70 → 0.76 | 0.54 → 0.70 | 0.43 → 0.25 | 0.59 → **0.76** | 0.30 → 0.46 |
| 40 | 0.58 → 0.54 | 0.65 → 0.71 | 0.20 → 0.30 | 0.53 → **0.79** | 0.30 → 0.31 |

(Pearson r between true and recovered parameters; see `results/recovery/` for raw output, `results/recovery_pre_fix/` for the before-fix run.)

- **kappa** — the parameter of primary interest — improved substantially (r=0.40–0.61 → r=0.70–0.84), with 25 trials/block now hitting r=0.84, 75% exact matches, 96% within one level.
- **gamma** went from statistical noise to a weak-but-significant signal (still well short of the original paper's reported r=0.73–1.0 for all parameters).
- **alpha**, **beta**, **lambda** all improved modestly and fairly consistently.

Gamma and kappa's remaining ceiling is likely driven by task design, not implementation: in Buergi et al.'s task, the bot's reasoning level genuinely varies across blocks (0/1/2, rotated), giving `gamma` (sensitivity to evidence about the opponent's level) something real to track. In this task, every agent plays at a constant CHASE level 1 (`agents.ts`) — the manipulation here is social/friendliness framing, not opponent sophistication. Whether to introduce varying opponent levels (and how to do so without confounding it with the friendly/neutral/control conditions) is an open design question, not a code fix.

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

- **8 blocks** per participant: 4 agents the participant had a prior conversation with in the companion chat task (2 "friendly", 2 "neutral"), plus 4 "control" agents they haven't spoken with. Block order and avatar assignment come from `backend/app/rotations/rotation.json`, keyed by config index — the same config index always gets the same order for a given participant.
- **35 trials per block** in Full mode. At current timing (4s response window + ITI jittered 0–6s + 2s feedback ≈ 9s/trial average), that's ~42 minutes total across 8 blocks — well over the ~10 minute MRI target (see Ongoing Tasks above). `Test` mode (5 trials/block, ~6 min total) is for local/dev checks only, not a real session length.
- **Points**: +3 win, −3 lose, 0 draw, starting from 100
- **Agents**: all 8 agents play at the same CHASE level (k=1) with calibrated noise — the friendly/neutral/control manipulation is social framing carried over from the chat task, not a difference in RPS strategy. There is no random/RNG baseline agent.

---

## Data output

All session, trial, and trigger data is written directly to the SQLite database at `backend/rps.db` (no CSV download). Tables:

**`sessions`** — `id`, `participant_id`, `session_number`, `mode`, `config_index`, `created_at`, `anchor_t_ms`

**`trials`** — `id`, `session_id`, `block`, `agent`, `trial_in_block`, `trial_global`, `participant_choice`, `agent_choice`, `outcome`, `points_delta`, `points_cumulative`, `rt_ms`, `onset_ms`, `feedback_onset_ms`, `iti_duration_ms`, `block_onset_ms`, `condition`

**`triggers`** — `id`, `session_id`, `tr_number`, `t_ms` (scanner TR pulses, scanner mode only)

### Clock model

All three tables sit on **one shared, session-local clock** measured in milliseconds — not wall-clock time. In scanner mode, the instant the *first* F8 pulse is detected is `t = 0` (the anchor, stored as `sessions.anchor_t_ms = 0`); in online mode, the anchor is set as soon as the session starts running. Every later timestamp — a trial's `onset_ms`/`feedback_onset_ms`, or a trigger's `t_ms` — is just `performance.now() − anchor` at the moment it happens, sent to the backend, and stored with no further transformation. `feedback_onset_ms` is computed deterministically as `onset_ms + response_window_ms`, since the choice screen always holds for the full response window regardless of when (or whether) the participant responds.

Because trial events and TR pulses share the same clock and the same zero point, aligning behavioral data to scanner pulses at analysis time is direct subtraction — no separate reconciliation step needed. `triggers.tr_number` is 0-indexed, so `tr_number = 0` is always the anchor pulse itself.

Writes happen incrementally as the task runs (each trial/trigger is its own request) rather than being batched and sent at the end, so data already collected survives a browser crash mid-session.

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
Computes win rate, choice entropy, win-stay/lose-shift, and lag-1 autocorrelation per participant × agent block, plus a monotonic gradient test (friendly > neutral > control).

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
