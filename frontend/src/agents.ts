/**
 * CHASEAgent replicates Buergi et al.'s artificial opponent (mn_RPS_task.m).
 * All tunable constants come from config.ts (mirrors mn_RPS_config.m).
 */

import { BOT, GAME, NOISE } from "./config";

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
  /** Update the reasoning level for the next block without resetting state. */
  setLevel(level: number): void;
  choose(): number;
  /** own = bot's choice, opp = participant's choice, score = +1 win / -1 lose / 0 tie */
  update(own: number, opp: number, score?: number): void;
  reset(): void;
  /** Whether the most recent choose() was a noisy trial. */
  getLastNoisy(): boolean;
  /** Bot's own attraction vector [R,P,S] snapshotted before the most recent choose(). */
  getLastBotAttr(): [number, number, number];
  /** Participant's attraction vector [R,P,S] snapshotted before the most recent choose(). */
  getLastPAttr(): [number, number, number];
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
  private readonly alpha: number;
  private readonly beta: number;   // normal-trial beta
  private readonly lambda: number;
  private readonly rng: () => number;

  // Dual attraction trackers (RW-freq delta rule, BAKR_2024_CHASE_LR_update.m)
  private attr: number[];   // bot's own action frequencies
  private pAttr: number[];  // participant's action frequencies (tracked by bot)

  // Participant outcome history for adaptive noise (+1 win, -1 lose, 0 tie from participant's view)
  private scores: number[];

  // Per-trial history for noise_breaker (mn_RPS_task.m lines ~100-115)
  private botChoices: number[];    // bot's own past choices
  private noisyHistory: boolean[]; // whether each past trial was a noisy trial

  // Snapshots from the most recent choose() call — read by timeline.ts for DB logging
  private _lastNoisy = false;
  private _lastBotAttr: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
  private _lastPAttr: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];

  // Row-player payoff matrix (myAction × theirAction), R/P/S indexed 0–2
  private readonly payoff = [
    [GAME.TIE,  GAME.LOSS, GAME.WIN ],
    [GAME.WIN,  GAME.TIE,  GAME.LOSS],
    [GAME.LOSS, GAME.WIN,  GAME.TIE ],
  ];

  constructor(id: string, config: CHASEConfig = {}) {
    this.id = id;
    this.level = config.level ?? 1;
    this.alpha = config.alpha ?? BOT.ALPHA;
    this.beta  = config.beta  ?? BOT.BETA;
    this.lambda = config.lambda ?? BOT.LAMBDA;
    this.rng = mkRng(config.seed ?? null);
    this.attr = [1 / 3, 1 / 3, 1 / 3];
    this.pAttr = [1 / 3, 1 / 3, 1 / 3];
    this.scores = [];
    this.botChoices = [];
    this.noisyHistory = [];
  }

  setLevel(level: number): void {
    this.level = level;
  }

  private softmax(v: number[], beta: number): number[] {
    const s = v.map((x) => beta * x);
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  /** Expected value of each action against a given opponent distribution. */
  private ev(dist: number[]): number[] {
    return this.payoff.map((row) =>
      row.reduce((s: number, v, j) => s + (v < 0 ? v * this.lambda : v) * dist[j]!, 0),
    );
  }

  /**
   * Compute action probabilities at the given beta, matching mn_RPS_task.m exactly:
   *
   *   k=0: softmax(attr, β)
   *   k=1: softmax(π @ pAttr, β)          — raw pAttr fed into EV, no initial softmax
   *   k=2: softmax(π @ softmax(π @ attr, β), β)
   */
  private probs(beta: number): number[] {
    if (this.level === 0) {
      return this.softmax(this.attr, beta);
    }
    const seed = this.level % 2 === 0 ? this.attr : this.pAttr;
    let p = this.softmax(this.ev(seed), beta);
    for (let k = 1; k < this.level; k++) {
      p = this.softmax(this.ev(p), beta);
    }
    return p;
  }

  /**
   * Matches get_noise_level() in mn_RPS_task.m.
   * Returns { beta, isNoisy }: beta is 10 (normal) or 1e-3 (noisy trial).
   * Always consumes 1 RNG call (streakMax draw); may consume a 2nd (chance check).
   */
  private getNoisyLevel(): { beta: number; isNoisy: boolean } {
    if (this.scores.length === 0) return { beta: this.beta, isNoisy: false };

    const timeHorizon = NOISE.TIME_HORIZON;
    const successCriterion = NOISE.SUCCESS_CRITERION;
    const skewness = NOISE.SKEWNESS;
    const maxWinStreak = this.level === 0 ? 2 : 3;          // streak_bounds[2]
    const streakMax = 3 + Math.floor(this.rng() * 2);        // randi([3 4])

    // Consecutive wins at end of history, bounded by maxWinStreak
    let winStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && winStreak < maxWinStreak; i--) {
      if (this.scores[i]! > 0) winStreak++;
      else break;
    }

    // Consecutive losses at end, bounded by streakMax
    let loseStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && loseStreak < streakMax; i--) {
      if (this.scores[i]! < 0) loseStreak++;
      else break;
    }

    // Consecutive ties at end, bounded by streakMax
    let tieStreak = 0;
    for (let i = this.scores.length - 1; i >= 0 && tieStreak < streakMax; i--) {
      if (this.scores[i]! === 0) tieStreak++;
      else break;
    }

    // Success rate over last timeHorizon trials
    const recent = this.scores.slice(-timeHorizon);
    const success = recent.filter((s) => s > 0).length / recent.length;

    if (loseStreak >= streakMax || tieStreak >= streakMax) {
      return { beta: NOISE.BETA, isNoisy: true };
    }

    if (success >= successCriterion && winStreak >= 1) {
      if (winStreak >= maxWinStreak) return { beta: NOISE.BETA, isNoisy: true };
      const chanceLevel = Math.pow(1 / (maxWinStreak + 1 - winStreak), skewness);
      if (this.rng() < chanceLevel) return { beta: NOISE.BETA, isNoisy: true };
    }

    return { beta: this.beta, isNoisy: false };
  }

  getLastNoisy(): boolean { return this._lastNoisy; }
  getLastBotAttr(): [number, number, number] { return this._lastBotAttr; }
  getLastPAttr(): [number, number, number] { return this._lastPAttr; }

  choose(): number {
    // Assert no NaN in either attraction tracker (mn_RPS_task.m assertion)
    if (this.attr.some(isNaN) || this.pAttr.some(isNaN)) {
      throw new Error(
        `CHASEAgent ${this.id}: NaN in attractions — attr=[${this.attr}] pAttr=[${this.pAttr}]`,
      );
    }

    // Snapshot pre-choice attractions for DB logging
    this._lastBotAttr = [...this.attr] as [number, number, number];
    this._lastPAttr   = [...this.pAttr] as [number, number, number];

    const { beta, isNoisy } = this.getNoisyLevel();
    this._lastNoisy = isNoisy;

    // Compute action probs with the trial's beta (1e-3 when noisy → near-uniform)
    let p = this.probs(beta);

    if (isNoisy) {
      // Exclude model-conform action (highest prob), sample from remaining two
      const maxIdx = p.indexOf(Math.max(...p));
      p = p.map((v, i) => (i === maxIdx ? 0 : v));
      const sum = p.reduce((a, b) => a + b, 0);
      p = p.map((v) => v / sum);

      // noise_breaker (mn_RPS_task.m lines ~100-115): if recent noisy trials all
      // played the same action as the last one, exclude that action too.
      // n_max = randi([1 2]) — consumes 1 RNG call, matching MATLAB's draw order.
      const nMax = 1 + Math.floor(this.rng() * 2);
      const recentNoisy = this.noisyHistory.slice(-nMax);
      const recentChoices = this.botChoices.slice(-nMax);
      const allNoisy = recentNoisy.length === nMax && recentNoisy.every(Boolean);
      // ~any(diff(choices)): true when all elements equal (or only 1 element)
      const allSameAction =
        recentChoices.length === nMax &&
        (nMax === 1 || recentChoices.every((c) => c === recentChoices[0]));

      if (allNoisy && allSameAction && this.botChoices.length > 0) {
        const lastIdx = ACTIONS.indexOf(
          this.botChoices[this.botChoices.length - 1]! as 1 | 2 | 3,
        );
        if (lastIdx >= 0 && p[lastIdx]! > 0) {
          p[lastIdx] = 0;
          const s = p.reduce((a, b) => a + b, 0);
          if (s > 0) p = p.map((v) => v / s);
        }
      }
    }

    const choice = ACTIONS[weightedChoice(this.rng, p)]!;
    this.botChoices.push(choice);
    this.noisyHistory.push(isNoisy);
    return choice;
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
    this.botChoices = [];
    this.noisyHistory = [];
  }
}

// ── Factory ──────────────────────────────────────────────────────────────────

export function buildAgent(avatarId: string, level: number, seed: number | null): Agent {
  return new CHASEAgent(avatarId, { level, seed });
}
