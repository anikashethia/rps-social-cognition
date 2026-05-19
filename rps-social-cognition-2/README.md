# RPS Social Cognition Task

A Rock-Paper-Scissors (RPS) task designed to measure **theory of mind (ToM) toward AI agents** as part of a larger study on social connection and mind attribution. Participants play RPS against four AI agents (A–D) and a random-draw control, with computational modeling using the **CHASE framework** (Buergi et al., 2026) to extract trial-by-trial mentalizing estimates.

## Overview

This task is the social cognition component of a multi-session neuroimaging study. It is administered **in-scanner**, after the conversation phase and face-viewing task. The core hypothesis is that felt connection during prior interaction modulates how strongly participants mentalize toward each agent during RPS — captured both behaviorally and computationally.

**Key constructs measured:**
- Win rate, choice entropy, win-stay/lose-shift (behavioral)
- Strategic level *k*, belief update magnitude (KL divergence), convergence rate (CHASE model)

**Prediction:** friendly > neutral > RNG on all mentalizing measures, driven by prior connection rather than agent difficulty (agents are matched on strategic level).

---

## Repository Structure

```
rps-social-cognition/
├── task/                    # PsychoPy task scripts
│   ├── rps_task.py          # Main task runner
│   ├── config.py            # Task parameters
│   ├── agents.py            # AI agent strategy implementations
│   └── trial_structure.py   # Trial/block logic
├── models/                  # Computational models
│   ├── chase.py             # CHASE model (Buergi et al., 2026)
│   ├── alternatives.py      # RL, FP, EWA, EWA-S, ToMk
│   └── model_comparison.py  # Bayesian model comparison utilities
├── analysis/                # Analysis scripts
│   ├── behavioral.py        # Win rate, entropy, WSLS, autocorrelation
│   ├── model_fitting.py     # MLE fitting pipeline
│   ├── belief_updates.py    # KL divergence / BU extraction
│   └── plots.py             # Figures
├── stimuli/
│   └── avatars/             # Agent avatar images (A, B, C, D + RNG icon)
├── utils/
│   ├── data_io.py           # Data loading/saving helpers
│   └── counterbalancing.py  # Block order counterbalancing
├── tests/                   # Unit tests
├── docs/
│   └── task_design.md       # Full design specification
├── requirements.txt
└── README.md
```

---

## Task Design

### Agents & Conditions

| Condition | Identifier | Strategic Level | Prior Context |
|-----------|-----------|----------------|---------------|
| Friendly agent | `friendly` | k ≈ 1–2 | High-connection conversation |
| Neutral agent | `neutral` | k ≈ 1–2 | Low-connection conversation |
| Random draw | `rng` | k = 0 (random) | No opponent |

Both social agents play at the **same strategic level** — any mentalizing difference between conditions is driven by prior connection, not difficulty. The random draw control provides a non-social baseline.

### Structure

- **3 blocks** (friendly, neutral, RNG), ~30–40 trials each
- **Block order** counterbalanced across participants (all 6 permutations of 3 conditions)
- **~12–15 min** total
- Each block identified by a unique avatar face (social agents) or generic icon (RNG)

### Trial Structure (following Buergi et al., 2026)

```
1. Opponent display     — avatar shown, opponent identified
2. Choice               — Rock / Paper / Scissors (self-paced)
3. Outcome feedback     — Win / Lose / Draw + points update
4. ITI                  — fixation cross (jittered)
```

### Payoff Structure

Points awarded for wins, deducted for losses, to incentivize strategic play. Exact conversion rate TBD based on lab norms.

---

## CHASE Model

The **Cognitive Hierarchy Assessment with Sophistication Estimation** (CHASE; Buergi et al., 2026) is the primary computational model. It captures *adaptive mentalization* — the trial-by-trial updating of beliefs about an opponent's strategic sophistication — rather than assuming a fixed strategy.

### Core assumptions

**A1 — Level-0 play** is governed by a recency-weighted action frequency tracker (delta rule over actions):

$$A(a)_{t+1} = A(a)_t + \alpha \cdot (\mathbf{I}(a) - A(a)_t)$$

**A2 — Strategic play (k > 0)** applies recursive best-response reasoning up to level k:

$$P(a|k>0) = \sigma(\Pi \times \cdots \sigma(\Pi \times P(a|k=0))\cdots)$$

**A3 — Adaptive play (κ > 1)** maintains a belief distribution over opponent levels and updates it via Bayes' rule:

$$B(k|a)_{t+1} = \frac{L(k|a)_t \cdot B(k)_t}{\sum_k L(k|a)_t \cdot B(k)_t}$$

### Key outputs (per agent condition)

| Parameter | Description |
|-----------|-------------|
| `k` | Strategic level used by participant |
| `kappa` | Maximum sophistication level |
| `alpha` | Action frequency learning rate |
| `beta` | Softmax noise |
| `gamma` | Sensitivity to opponent-level evidence (Bayesian learning rate analog) |
| `lambda` | Loss sensitivity |
| `BU` | Belief update magnitude (KL divergence between successive belief distributions) |

### Behavioral measures (no model required)

| Measure | Description |
|---------|-------------|
| Win rate | Proportion wins (chance = 1/3) |
| Choice entropy | Randomness of choices (lower = more structured) |
| Win-stay rate | P(repeat action \| previous win) |
| Lose-shift rate | P(switch action \| previous loss) |
| Choice autocorrelation | Sequential dependencies in choices |

---

## Installation

```bash
git clone https://github.com/your-lab/rps-social-cognition.git
cd rps-social-cognition
pip install -r requirements.txt
```

### Dependencies

See `requirements.txt`. Core dependencies:
- `psychopy` — task presentation
- `numpy`, `scipy` — numerical computing
- `pandas` — data handling
- `matplotlib`, `seaborn` — plotting
- `pymc` or custom MLE — model fitting

---

## Running the Task

```bash
cd task/
python rps_task.py --participant_id SUB001 --session 1
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--participant_id` | Subject identifier | required |
| `--session` | Session number | `1` |
| `--n_trials` | Trials per block | `35` |
| `--practice` | Run practice block | `False` |
| `--fullscreen` | Fullscreen mode | `True` |
| `--seed` | RNG seed for block order | `None` |

Output is saved to `data/SUB001_session1_rps.csv`.

---

## Data Format

Each row is one trial. Key columns:

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | str | Subject ID |
| `session` | int | Session number |
| `block` | int | Block number (1–5) |
| `agent` | str | `friendly`, `neutral`, or `rng` |
| `trial` | int | Trial within block |
| `participant_choice` | int | 1=Rock, 2=Paper, 3=Scissors |
| `agent_choice` | int | 1=Rock, 2=Paper, 3=Scissors |
| `outcome` | str | `win`, `lose`, `draw` |
| `points` | int | Cumulative points |
| `rt` | float | Response time (s) |
| `timestamp` | float | Unix timestamp |

---

## Fitting the CHASE Model

```python
from models.chase import CHASEModel
from analysis.model_fitting import fit_participant

# Load data for one participant
data = load_participant_data("SUB001")

# Fit CHASE separately for each agent condition
results = {}
for agent in ["friendly", "neutral", "rng"]:
    agent_data = data[data["agent"] == agent]
    results[agent] = fit_participant(CHASEModel, agent_data)

# Extract belief updates (KL divergence per trial)
from analysis.belief_updates import extract_bu_timeseries
bu = extract_bu_timeseries(results["friendly"])
```

---

## Analysis Pipeline

```bash
# 1. Behavioral summary (per participant × agent)
python analysis/behavioral.py --data_dir data/ --output_dir results/behavioral/

# 2. Model fitting (all participants)
python analysis/model_fitting.py --data_dir data/ --model chase --output_dir results/models/

# 3. Model comparison (CHASE vs. alternatives)
python models/model_comparison.py --results_dir results/models/

# 4. Generate figures
python analysis/plots.py --results_dir results/ --output_dir figures/
```

---

## Task Placement Rationale

The RPS task is administered *after* all conversations and the face-viewing task because:

1. **Avoid priming mentalizing** — inserting a ToM task between conversations could artificially inflate or alter social dynamics during the conversation phase.
2. **Test persistence** — demonstrating that connection-induced mentalizing persists beyond the interaction is a stronger result than showing it only during conversation.
3. **Clean analytic separation** — conversation phase → social dynamics; face-viewing phase → RSA/neural representation; RPS phase → ToM. Connection during conversation is the bridge variable.

---

## Citation

If you use this task or the CHASE model, please cite:

```bibtex
@article{buergi2026neural,
  title={A neural signature of adaptive mentalization},
  author={Buergi, Niklas and Aydogan, G{\"o}khan and Konovalov, Arkady and Ruff, Christian C.},
  journal={Nature Neuroscience},
  volume={29},
  pages={934--944},
  year={2026},
  doi={10.1038/s41593-026-02219-x}
}
```

---

## Contact

[Your lab contact here]

*Created: 2026-03-16 | Task design session 22*
*Last updated: 2026-05-19*
