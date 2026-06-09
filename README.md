# RPS Social Cognition Task

A Rock-Paper-Scissors task for studying social cognition and mentalization. Participants play RPS against four social agents (A–D) and a random baseline, while computational modeling (CHASE) captures trial-by-trial belief updating about opponent strategy.

Based on the CHASE model from Buergi, Aydogan, Konovalov & Ruff (2026, *Nature Neuroscience*).

---

## Repository structure

```
rps-social-cognition/
├── task/
│   └── rps_task.html        # jsPsych task — open in browser to run
├── stimuli/
│   └── avatars/             # Drop avatar PNGs here (optional)
├── analysis/
│   ├── behavioral.py        # Model-free measures (win rate, entropy, WSLS)
│   ├── model_fitting.py     # MLE fitting pipeline, AIC model comparison
│   ├── belief_updates.py    # CHASE belief update timeseries utilities
│   └── plots.py             # Figures (in progress)
├── models/
│   ├── chase.py             # Full CHASE model
│   └── alternatives.py      # RL, FP, EWA, EWA-S, ToMk (stubs)
├── utils/
│   ├── counterbalancing.py  # Latin-square block order
│   └── data_io.py           # CSV load/save helpers
├── data/                    # Output CSVs go here (gitignored)
├── results/                 # Analysis outputs go here (gitignored)
└── requirements.txt
```

---

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



## Running the task

### Local testing

1. Clone the repo and start a local server from the root:
   ```bash
   python3 -m http.server 8000
   ```
2. Open your browser and go to:
   ```
   http://localhost:8000/task/rps_task.html
   ```
3. Use keys **1**, **2**, **3** to play (Rock, Paper, Scissors).

By default the task runs **5 trials per block** for fast testing. At the end it downloads a CSV to your Downloads folder.

### URL parameters

Append these to the URL to configure the task:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pid` | `TEST001` | Participant ID |
| `session` | `1` | Session number |
| `trials` | `5` | Trials per block (use `35` for real runs) |
| `practice` | off | Add `&practice=1` to include a practice block |
| `seed` | random | Add `&seed=42` for a reproducible agent sequence |

Example full run:
```
http://localhost:8000/task/rps_task.html?pid=SUB001&session=1&trials=35&practice=1
```

### Online (Prolific + fly.io)

Host the repo on fly.io and point your Prolific study URL to:
```
https://your-app.fly.dev/task/rps_task.html?pid={{%PROLIFIC_PID%}}&session=1&trials=35
```

---

## Task design

- **5 blocks**, one per agent (A, B, C, D, RNG), order counterbalanced across participants via Latin square
- **35 trials per block** (~4–5 min per block)
- **Points**: +3 win, −3 lose, 0 draw, starting from 100
- **Agents**: all social agents (A–D) play at the same CHASE level (k=1) with calibrated noise; RNG is purely random
- Agent order is determined by participant ID so the same ID always gets the same order

---

## Data output

Each run downloads a CSV with one row per trial:

| Column | Description |
|--------|-------------|
| `participant_id` | Participant ID |
| `session` | Session number |
| `block` | Block number (1–5) |
| `agent` | Agent ID (`agent_a` … `rng`) |
| `trial_in_block` | Trial number within block |
| `trial_global` | Trial number across whole task |
| `participant_choice` | 1=Rock, 2=Paper, 3=Scissors |
| `agent_choice` | Agent's choice |
| `outcome` | `win` / `lose` / `draw` |
| `points_delta` | Points earned this trial |
| `points_cumulative` | Running total |
| `rt` | Response time (ms) |
| `timestamp` | Unix timestamp |

Place CSV files in `data/` before running analysis scripts.

---

## Analysis

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

**Behavioral (model-free):**
```bash
python analysis/behavioral.py --data_dir data/ --output_dir results/behavioral/
```
Computes win rate, choice entropy, win-stay/lose-shift, and lag-1 autocorrelation per participant × agent, plus a monotonic gradient test (A > B > C > D > RNG).

**Model fitting (CHASE + alternatives):**
```bash
python analysis/model_fitting.py --data_dir data/ --output_dir results/models/ --model chase,rl,fp
```
Fits models via MLE with random restarts, outputs per-participant parameters, trial-level CHASE estimates (belief updates, APE, choice values), and AIC model comparison.

---

## Avatar images

To use custom avatars instead of emoji placeholders:

1. Add PNG files to `stimuli/avatars/`:
   ```
   stimuli/avatars/avatar_a.png
   stimuli/avatars/avatar_b.png
   stimuli/avatars/avatar_c.png
   stimuli/avatars/avatar_d.png
   stimuli/avatars/rng_icon.png
   ```
2. In `task/rps_task.html`, find the `AVATAR_PATHS` config at the top of the script and update:
   ```javascript
   AVATAR_PATHS: {
     agent_a: "stimuli/avatars/avatar_a.png",
     agent_b: "stimuli/avatars/avatar_b.png",
     agent_c: "stimuli/avatars/avatar_c.png",
     agent_d: "stimuli/avatars/avatar_d.png",
     rng:     "stimuli/avatars/rng_icon.png",
   }
   ```

---

## Citation

Buergi, Aydogan, Konovalov & Ruff (2026). *Nature Neuroscience.*
