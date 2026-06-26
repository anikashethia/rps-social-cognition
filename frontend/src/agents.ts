/**
 * Agent strategy logic for the RPS task.
 * CHASEAgent exactly implements the CHASE model (Buergi et al., Nature Neuroscience 2026):
 * - Dual attraction trackers: own choices + participant's choices (RW-freq delta rule)
 * - Parity-based seeding: even levels seed from bot's own habits, odd from participant's habits
 * - Lambda scales loss payoffs in best-response computation
 * - Beta (softmax inverse temperature) is the only source of decision noise — no epsilon noise
 */

// ── Seeded RNG (mulberry32) ──────────────────────────────────────────────────

function mkRng(seed: number | null): () => number {
  if (seed === null) return () => Math.random();
  let s = seed;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function weightedChoice(rng: () => number, probs: number[]): number {
  const r = rng();
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i]!;
    if (r < c) return i;
  }
  return probs.length - 1;
}

// ── Constants ────────────────────────────────────────────────────────────────

const ACTIONS: readonly [1, 2, 3] = [1, 2, 3];

// ── Agent interface ──────────────────────────────────────────────────────────

export interface Agent {
  readonly id: string;
  choose(): number;
  update(own: number, opp: number): void;
  reset(): void;
}

// ── CHASEAgent ───────────────────────────────────────────────────────────────

interface CHASEConfig {
  level?: number;
  alpha?: number;   // attraction learning rate
  beta?: number;    // softmax inverse temperature (decision noise)
  lambda?: number;  // loss sensitivity (scales negative payoffs)
  seed?: number | null;
}

class CHASEAgent implements Agent {
  readonly id: string;
  private level: number;
  private alpha: number;
  private beta: number;
  private lambda: number;
  private rng: () => number;

  // Dual attraction trackers (RW-freq delta rule, matching BAKR_2024_CHASE_LR_update.m)
  private attr: number[];   // bot's own action frequencies
  private pAttr: number[];  // participant's action frequencies (as tracked by the bot)

  // Row-player payoff matrix: payoff[myAction][theirAction]
  // 0=Rock, 1=Paper, 2=Scissors (0-indexed internally)
  private readonly payoff = [
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
  ];

  constructor(id: string, config: CHASEConfig = {}) {
    this.id = id;
    this.level = config.level ?? 1;
    this.alpha = config.alpha ?? 0.3;
    this.beta = config.beta ?? 3.0;
    this.lambda = config.lambda ?? 1.0;
    this.rng = mkRng(config.seed ?? null);
    this.attr = [1 / 3, 1 / 3, 1 / 3];
    this.pAttr = [1 / 3, 1 / 3, 1 / 3];
  }

  private softmax(v: number[]): number[] {
    const s = v.map((x) => this.beta * x);
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  /** Best-response distribution against an opponent mixed strategy, with lambda on losses. */
  private br(op: number[]): number[] {
    const ev = this.payoff.map((row) =>
      row.reduce((s, v, j) => s + (v < 0 ? v * this.lambda : v) * op[j]!, 0),
    );
    return this.softmax(ev);
  }

  /**
   * Compute action probabilities using CHASE-style parity-based recursive best-response.
   *
   * Matches Buergi et al.'s two-tracker recursion:
   *   Even levels (0, 2, ...): seed from bot's own habit (this.attr)
   *   Odd levels  (1, 3, ...): seed from participant's habit (this.pAttr)
   *
   * Examples:
   *   level=0: softmax(attr)                    — plays own habit, ignores participant
   *   level=1: BR(softmax(pAttr))               — best-responds to participant's habit
   *   level=2: BR(BR(softmax(attr)))             — 2-step recursion from own habit
   */
  private probs(): number[] {
    const seed = this.level % 2 === 0 ? this.attr : this.pAttr;
    let p = this.softmax(seed);
    for (let k = 0; k < this.level; k++) p = this.br(p);
    return p;
  }

  choose(): number {
    return ACTIONS[weightedChoice(this.rng, this.probs())]!;
  }

  /**
   * Delta-rule update for both attraction trackers.
   * own  = bot's own choice (updates this.attr)
   * opp  = participant's choice (updates this.pAttr)
   * Matches BAKR_2024_CHASE_LR_update.m RW-freq rule.
   */
  update(own: number, opp: number): void {
    const ownIdx = ACTIONS.indexOf(own as 1 | 2 | 3);
    this.attr = this.attr.map(
      (a, i) => a + this.alpha * ((i === ownIdx ? 1 : 0) - a),
    );

    const oppIdx = ACTIONS.indexOf(opp as 1 | 2 | 3);
    this.pAttr = this.pAttr.map(
      (a, i) => a + this.alpha * ((i === oppIdx ? 1 : 0) - a),
    );
  }

  reset(): void {
    this.attr = [1 / 3, 1 / 3, 1 / 3];
    this.pAttr = [1 / 3, 1 / 3, 1 / 3];
  }
}

// ── Factory ──────────────────────────────────────────────────────────────────

export function buildAgent(avatarId: string, level: number, seed: number | null): Agent {
  return new CHASEAgent(avatarId, { level, seed });
}
