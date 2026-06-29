/**
 * CHASEAgent replicates Buergi et al.'s artificial opponent (mn_RPS_task.m):
 *   - alpha=0.9, beta=10 (87th/94th percentile of human-human gameplay)
 *   - Dual attraction trackers: own choices + participant choices (RW-freq delta rule)
 *   - Level-k generation: level 0 softmax(own attr); levels 1+ use raw attractions as
 *     input to expected-value computation (no initial softmax on seed), matching their code
 *   - Adaptive WSLS noise: after participant win/lose/tie streaks, plays non-optimal action
 *     (mn_RPS_task.m get_noise_level logic; Supplementary Methods section)
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
  /** own = bot's choice, opp = participant's choice, score = +1 win / -1 lose / 0 tie */
  update(own: number, opp: number, score?: number): void;
  reset(): void;
}

// ── CHASEAgent ───────────────────────────────────────────────────────────────

interface CHASEConfig {
  level?: number;
  alpha?: number;
  beta?: number;
  lambda?: number;
  seed?: number | null;
}

class CHASEAgent implements Agent {
  readonly id: string;
  private level: number;
  private alpha: number;
  private beta: number;
  private lambda: number;
  private rng: () => number;

  // Dual attraction trackers (RW-freq delta rule, BAKR_2024_CHASE_LR_update.m)
  private attr: number[];   // bot's own action frequencies
  private pAttr: number[];  // participant's action frequencies (tracked by bot)

  // Participant outcome history for adaptive noise (+1 win, -1 lose, 0 tie from participant's view)
  private scores: number[];

  // Row-player payoff: payoff[myAction][theirAction], 0=Rock 1=Paper 2=Scissors
  private readonly payoff = [
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
  ];

  constructor(id: string, config: CHASEConfig = {}) {
    this.id = id;
    this.level = config.level ?? 1;
    this.alpha = config.alpha ?? 0.9;  // 87th percentile human-human (Supplementary Methods)
    this.beta = config.beta ?? 10;     // 94th percentile human-human (Supplementary Methods)
    this.lambda = config.lambda ?? 1.0;
    this.rng = mkRng(config.seed ?? null);
    this.attr = [1 / 3, 1 / 3, 1 / 3];
    this.pAttr = [1 / 3, 1 / 3, 1 / 3];
    this.scores = [];
  }

  private softmax(v: number[]): number[] {
    const s = v.map((x) => this.beta * x);
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  /** Expected value of each action against a given opponent distribution. */
  private ev(dist: number[]): number[] {
    return this.payoff.map((row) =>
      row.reduce((s, v, j) => s + (v < 0 ? v * this.lambda : v) * dist[j]!, 0),
    );
  }

  /**
   * Compute action probabilities, matching mn_RPS_task.m exactly:
   *
   *   k=0: softmax(attr, β)
   *         — plays own habit via softmax, ignores participant
   *
   *   k=1: softmax(π @ pAttr, β)
   *         — raw participant attractions fed directly into EV (no softmax on pAttr)
   *         — matches: bot_resp_k(2,:) = softmax_fxn((pi_bot * f_subj)', beta)
   *
   *   k=2: softmax(π @ softmax(π @ attr, β), β)
   *         — bot predicts participant as level-1 (BR to raw attr), then best-responds
   *         — matches: bot_pred = softmax_fxn((pi_subj*f_bot)',beta);
   *                    bot_resp_k(3,:) = softmax_fxn((pi_bot*bot_pred')',beta)
   */
  private probs(): number[] {
    if (this.level === 0) {
      return this.softmax(this.attr);
    }
    // Levels 1+: raw attractions used as distribution seed (no initial softmax on seed)
    const seed = this.level % 2 === 0 ? this.attr : this.pAttr;
    let p = this.softmax(this.ev(seed));        // first BR against raw seed
    for (let k = 1; k < this.level; k++) {
      p = this.softmax(this.ev(p));             // subsequent BRs (intermediate p is already a distribution)
    }
    return p;
  }

  /**
   * Adaptive WSLS noise, matching mn_RPS_task.m get_noise_level():
   *
   * Returns true (→ play non-optimal action) when:
   *   - Participant lose or tie streak ≥ 3 or 4 (threshold randomised each trial)
   *   - Participant win rate ≥ 50% over last 5 trials AND win streak ≥ 1,
   *     with probability 1/(maxStreak+1−streak)^1.3; deterministic at streak 2 (k=0) or 3 (k>0)
   */
  private isNoisyTrial(): boolean {
    if (this.scores.length === 0) return false;

    const timeHorizon = 5;
    const successCriterion = 0.5;
    const skewness = 1.3;
    const maxWinStreak = this.level === 0 ? 2 : 3;         // streak_bounds[2]
    const streakMax = 3 + Math.floor(this.rng() * 2);       // randi([3 4]): 3 or 4

    // Success rate over last timeHorizon trials
    const recent = this.scores.slice(-timeHorizon);
    const success = recent.filter((s) => s > 0).length / recent.length;

    // Consecutive wins at end of history (bounded by maxWinStreak)
    let winStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && winStreak < maxWinStreak; i--) {
      if (this.scores[i]! > 0) winStreak++;
      else break;
    }

    // Consecutive losses at end (bounded by streakMax)
    let loseStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && loseStreak < streakMax; i--) {
      if (this.scores[i]! < 0) loseStreak++;
      else break;
    }

    // Consecutive ties at end (bounded by streakMax)
    let tieStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && tieStreak < streakMax; i--) {
      if (this.scores[i]! === 0) tieStreak++;
      else break;
    }

    // Lose or tie streak → deterministically noisy
    if (loseStreak >= streakMax || tieStreak >= streakMax) return true;

    // Win streak with sufficient overall success → increasingly likely noisy
    if (success >= successCriterion && winStreak >= 1) {
      const chanceLevel = Math.pow(1 / (maxWinStreak + 1 - winStreak), skewness);
      if (this.rng() < chanceLevel || winStreak >= maxWinStreak) return true;
    }

    return false;
  }

  choose(): number {
    let p = this.probs();

    if (this.isNoisyTrial()) {
      // Exclude the model-conform action (highest probability), sample from remaining two
      const maxIdx = p.indexOf(Math.max(...p));
      p = p.map((v, i) => (i === maxIdx ? 0 : v));
      const sum = p.reduce((a, b) => a + b, 0);
      p = p.map((v) => v / sum);
    }

    return ACTIONS[weightedChoice(this.rng, p)]!;
  }

  update(own: number, opp: number, score?: number): void {
    const ownIdx = ACTIONS.indexOf(own as 1 | 2 | 3);
    this.attr = this.attr.map(
      (a, i) => a + this.alpha * ((i === ownIdx ? 1 : 0) - a),
    );
    const oppIdx = ACTIONS.indexOf(opp as 1 | 2 | 3);
    this.pAttr = this.pAttr.map(
      (a, i) => a + this.alpha * ((i === oppIdx ? 1 : 0) - a),
    );
    if (score !== undefined) this.scores.push(score);
  }

  reset(): void {
    this.attr = [1 / 3, 1 / 3, 1 / 3];
    this.pAttr = [1 / 3, 1 / 3, 1 / 3];
    this.scores = [];
  }
}

// ── Factory ──────────────────────────────────────────────────────────────────

export function buildAgent(avatarId: string, level: number, seed: number | null): Agent {
  return new CHASEAgent(avatarId, { level, seed });
}
