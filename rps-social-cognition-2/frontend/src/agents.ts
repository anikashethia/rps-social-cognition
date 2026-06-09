/**
 * Agent strategy logic for the RPS task.
 * Ported from rps_task.html — mulberry32 seeded RNG, RandomAgent, CHASEAgent.
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

// ── RandomAgent ──────────────────────────────────────────────────────────────

class RandomAgent implements Agent {
  readonly id: string;
  private rng: () => number;

  constructor(id: string, seed: number | null) {
    this.id = id;
    this.rng = mkRng(seed);
  }

  choose(): number {
    return ACTIONS[Math.floor(this.rng() * 3)]!;
  }

  update(_own: number, _opp: number): void {}
  reset(): void {}
}

// ── CHASEAgent ───────────────────────────────────────────────────────────────
// Level-k recursive best-response with softmax + delta-rule attraction updates.

interface CHASEConfig {
  level?: number;
  alpha?: number;
  beta?: number;
  noise?: number;
  seed?: number | null;
}

class CHASEAgent implements Agent {
  readonly id: string;
  private level: number;
  private alpha: number;
  private beta: number;
  private noise: number;
  private rng: () => number;
  private attr: number[];

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
    this.noise = config.noise ?? 0.15;
    this.rng = mkRng(config.seed ?? null);
    this.attr = [1 / 3, 1 / 3, 1 / 3];
  }

  private softmax(v: number[]): number[] {
    const s = v.map((x) => this.beta * x);
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  /** Level-0 belief: softmax over own action attractions. */
  private l0(): number[] {
    return this.softmax(this.attr);
  }

  /** Best-response distribution against an opponent mixed strategy. */
  private br(op: number[]): number[] {
    const ev = this.payoff.map((row) =>
      row.reduce((s, v, j) => s + v * op[j]!, 0),
    );
    return this.softmax(ev);
  }

  /** Compute action probabilities at configured level. */
  private probs(): number[] {
    let p = this.l0();
    for (let k = 0; k < this.level; k++) p = this.br(p);
    return p;
  }

  choose(): number {
    const p = this.probs();
    const idx =
      this.rng() < this.noise
        ? Math.floor(this.rng() * 3)
        : weightedChoice(this.rng, p);
    return ACTIONS[idx]!;
  }

  update(own: number, _opp: number): void {
    const idx = ACTIONS.indexOf(own as 1 | 2 | 3);
    this.attr = this.attr.map(
      (a, i) => a + this.alpha * ((i === idx ? 1 : 0) - a),
    );
  }

  reset(): void {
    this.attr = [1 / 3, 1 / 3, 1 / 3];
  }
}

// ── Agent config (all social agents at same CHASE level) ─────────────────────

export const AGENT_CONFIG: Record<
  string,
  { level: number; noise: number | null }
> = {
  agent_a: { level: 1, noise: 0.15 },
  agent_b: { level: 1, noise: 0.15 },
  agent_c: { level: 1, noise: 0.15 },
  agent_d: { level: 1, noise: 0.15 },
  rng: { level: 0, noise: null },
};

// ── Factory ──────────────────────────────────────────────────────────────────

export function buildAgent(agentId: string, seed: number | null): Agent {
  if (agentId === "rng") return new RandomAgent(agentId, seed);
  const cfg = AGENT_CONFIG[agentId];
  if (!cfg) return new RandomAgent(agentId, seed);
  return new CHASEAgent(agentId, {
    level: cfg.level,
    noise: cfg.noise ?? 0.15,
    seed,
  });
}
