"""
Task configuration — Python mirror of frontend/src/config.ts.
Mirrors Buergi et al.'s mn_RPS_config.m + BAKR_2024_CHASE_config.m.
All Python analysis and model files import constants from here.
"""

# ── Task ──────────────────────────────────────────────────────────────────────

N_TRIALS = 40   # per block  (task.n_trials)
N_BLOCKS = 6    # 2 agents × 3 CHASE levels  (task.n_blocks)

# ── Game (task.game) ──────────────────────────────────────────────────────────

N_ACTIONS = 3   # Rock / Paper / Scissors  (task.game.strat_space)
WIN  =  1       # task.game.win
LOSS = -1       # task.game.loss
TIE  =  0       # task.game.tie

# ── Bot (task.bot.params) ─────────────────────────────────────────────────────

BOT_ALPHA  = 0.9   # task.bot.params.alpha — 87th percentile human-human gameplay
BOT_BETA   = 10.0  # task.bot.params.beta  — 94th percentile human-human gameplay
BOT_LAMBDA = 1.0   # loss sensitivity; 1.0 = neutral

# ── Adaptive noise (get_noise_level in mn_RPS_task.m) ────────────────────────

NOISY_BETA         = 1e-3  # beta when noisy — near-uniform softmax
NOISE_TIME_HORIZON = 5     # trials back for participant win-rate estimate
NOISE_SUCCESS_CRIT = 0.5   # win-rate threshold to trigger win-streak check
NOISE_SKEWNESS     = 1.3   # exponent for win-streak noise probability

# ── CHASE participant model ───────────────────────────────────────────────────
# Fitting parameter bounds and starting values (BAKR_2024_CHASE_config.m).

PARAM_BOUNDS = {
    "alpha": (1e-4, 1.0 - 1e-4),  # learning rate
    "beta":  (0.0,  100.0),        # softmax inverse temperature — widened; 28% hit upper bound at 20
    "gamma": (0.01, 20.0),         # sensitivity to opponent-level evidence
    "lam":   (0.0,  100.0),        # loss sensitivity (lambda) — widened
    "kappa": (0,    4),            # max sophistication level (integer)
}

PARAM_INIT = {
    "alpha": 0.3,
    "beta":  3.0,
    "gamma": 2.0,
    "lam":   1.0,
    "kappa": 3,
}
