# task/config.py
"""
Task configuration for RPS Social Cognition Task.
All timing, payoff, and display parameters live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ── Timing (seconds) ──────────────────────────────────────────────────────────
OPPONENT_DISPLAY_DURATION = 1.0       # Time avatar is shown before choice
FEEDBACK_DURATION         = 1.5       # Win/Lose/Draw display
ITI_MIN                   = 0.5       # Minimum inter-trial interval
ITI_MAX                   = 2.0       # Maximum ITI (jittered uniformly)
RESPONSE_TIMEOUT          = 5.0       # Max wait for participant choice (self-paced but bounded)

# ── Trial / Block structure ───────────────────────────────────────────────────
N_TRIALS_PER_BLOCK   = 35            # ~4–5 min per block at ~7 s/trial
N_PRACTICE_TRIALS    = 10
AGENTS               = ["agent_a", "agent_b", "agent_c", "agent_d", "rng"]

# ── Payoff structure ──────────────────────────────────────────────────────────
POINTS_WIN   =  3
POINTS_LOSE  = -3
POINTS_DRAW  =  0
STARTING_POINTS = 100

# ── Game mapping ──────────────────────────────────────────────────────────────
# Actions: 1=Rock, 2=Paper, 3=Scissors
# Winner: action that is exactly one step ahead (mod 3)
ACTIONS       = [1, 2, 3]
ACTION_LABELS = {1: "Rock", 2: "Paper", 3: "Scissors"}
ACTION_KEYS   = {"1": 1, "2": 2, "3": 3}  # Keyboard mapping

# Outcome matrix: OUTCOME[participant][agent] -> "win" | "lose" | "draw"
def get_outcome(p_choice: int, a_choice: int) -> str:
    """Return outcome from participant's perspective."""
    if p_choice == a_choice:
        return "draw"
    # Rock(1) beats Scissors(3), Paper(2) beats Rock(1), Scissors(3) beats Paper(2)
    # i.e. choice x beats choice ((x % 3) + 1) - 1 ... simplest: wins if p = (a % 3) + 1
    if p_choice == (a_choice % 3) + 1:
        return "win"
    return "lose"

# ── Agent strategy levels (CHASE k-values) ───────────────────────────────────
# All agents play at the same difficulty; connection (not difficulty) drives any mentalizing gradient.
AGENT_STRATEGY_LEVELS: Dict[str, int] = {
    "agent_a": 1,   # Will vary trial-by-trial around this; agents are matched on difficulty
    "agent_b": 1,
    "agent_c": 1,
    "agent_d": 1,
    "rng":     0,   # Purely random; k=0 baseline
}

# Noise parameter for agent strategies (calibrated to pass Turing test; see Buergi et al. 2026)
AGENT_NOISE: Dict[str, float] = {
    "agent_a": 0.3,
    "agent_b": 0.3,
    "agent_c": 0.3,
    "agent_d": 0.3,
    "rng":     None,  # No strategy noise — purely random
}

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_SIZE        = (1920, 1080)
FULLSCREEN         = True
BACKGROUND_COLOR   = [0.1, 0.1, 0.1]   # Dark grey
TEXT_COLOR         = [0.95, 0.95, 0.95]
HIGHLIGHT_COLOR    = [0.2, 0.8, 0.4]   # Win green
LOSS_COLOR         = [0.9, 0.3, 0.3]   # Loss red
DRAW_COLOR         = [0.8, 0.8, 0.2]   # Draw yellow
FONT               = "Helvetica"
FONT_SIZE_LARGE    = 48
FONT_SIZE_MEDIUM   = 32
FONT_SIZE_SMALL    = 24

AVATAR_SIZE        = (200, 200)         # Pixels
RNG_ICON_PATH      = "stimuli/avatars/rng_icon.png"
AVATAR_PATHS: Dict[str, str] = {
    "agent_a": "stimuli/avatars/avatar_a.png",
    "agent_b": "stimuli/avatars/avatar_b.png",
    "agent_c": "stimuli/avatars/avatar_c.png",
    "agent_d": "stimuli/avatars/avatar_d.png",
    "rng":     RNG_ICON_PATH,
}

# ── MRI / Scanner settings ────────────────────────────────────────────────────
SCANNER_TRIGGER_KEY = "5"              # TTL trigger key from scanner
RESPONSE_KEYS       = ["1", "2", "3"] # Button box mapping

# ── Data output ───────────────────────────────────────────────────────────────
DATA_DIR          = "data/"
LOG_DIR           = "logs/"
OUTPUT_COLUMNS    = [
    "participant_id", "session", "block", "block_order",
    "agent", "trial", "trial_global",
    "participant_choice", "agent_choice",
    "outcome", "points_delta", "points_cumulative",
    "rt", "timestamp",
]
