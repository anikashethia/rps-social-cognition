/**
 * Task configuration — mirrors Buergi et al.'s mn_RPS_config.m structure.
 * All task constants live here; agents.ts and timeline.ts import from this file.
 *
 * mn_RPS_config.m reference:
 *   task.n_trials = 40;  task.n_blocks = 6;
 *   task.game.{win=1, loss=-1, tie=0, strat_space=3}
 *   task.bot.params.{alpha=0.9, beta=10}
 *   task.bot.levels  ← handled by backend/app/rotations/rotation.json
 */

// ── Task ─────────────────────────────────────────────────────────────────────

export const N_TRIALS = 40;  // per block  (task.n_trials)
export const N_BLOCKS = 6;   // 2 agents × 3 CHASE levels  (task.n_blocks)

// ── Game (task.game) ─────────────────────────────────────────────────────────

export const GAME = {
  STRAT_SPACE: 3,   // task.game.strat_space — Rock / Paper / Scissors
  WIN:          1,  // task.game.win
  LOSS:        -1,  // task.game.loss
  TIE:          0,  // task.game.tie
} as const;

// ── Bot (task.bot.params) ─────────────────────────────────────────────────────
// Level sequence is in backend/app/rotations/rotation.json, counterbalanced
// across participants (mirrors task.bot.levels generation in mn_RPS_config.m).

export const BOT = {
  ALPHA:  0.9,   // task.bot.params.alpha — 87th percentile human-human gameplay
  BETA:   10,    // task.bot.params.beta  — 94th percentile human-human gameplay
  LAMBDA: 1.0,   // loss sensitivity; 1.0 = neutral (bot has no loss aversion)
} as const;

// ── Adaptive noise (get_noise_level in mn_RPS_task.m) ────────────────────────

export const NOISE = {
  BETA:              1e-3,  // beta when noisy — near-uniform softmax
  TIME_HORIZON:      5,     // trials back for participant win-rate estimate
  SUCCESS_CRITERION: 0.5,   // win-rate threshold to trigger win-streak check
  SKEWNESS:          1.3,   // exponent for win-streak noise probability
} as const;

// ── Timings (ms) ─────────────────────────────────────────────────────────────

export type Mode = "dev" | "behavioral" | "scanner";

export const TIMINGS: Record<Mode, {
  trials: number;
  iti_min: number;
  iti_max: number;
  response_window: number;
  post_response_min: number;
  post_response_max: number;
  feedback: number;
}> = {
  dev:        { trials: 5,  iti_min: 0,    iti_max: 1000, response_window: 3000, post_response_min: 100, post_response_max: 2000, feedback: 2000 },
  behavioral: { trials: 40, iti_min: 0,    iti_max: 6000, response_window: 3000, post_response_min: 100, post_response_max: 2000, feedback: 2000 },
  scanner:    { trials: 40, iti_min: 0,    iti_max: 6000, response_window: 3000, post_response_min: 100, post_response_max: 2000, feedback: 2000 },
} as const;

// ── Points (display) ─────────────────────────────────────────────────────────
// Matches GAME.WIN / GAME.LOSS so the displayed score is the raw game payoff.

export const POINTS = {
  WIN:   GAME.WIN,   //  +1
  LOSS:  GAME.LOSS,  //  -1
  TIE:   GAME.TIE,   //   0
  START: 0,
} as const;
