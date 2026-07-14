/**
 * compare_ts_matlab.mjs
 *
 * Tests whether agents.ts math matches Buergi's mn_RPS_task.m math.
 * Reimplements agents.ts probability + update equations in plain JS,
 * runs them on a fixed 40-trial choice sequence, and writes ts_probs.json.
 *
 * Run:
 *   node analysis/compare_ts_matlab.mjs
 *
 * Then run compare_ts_matlab.m in MATLAB to produce matlab_probs.json,
 * then compare_ts_matlab.py to assert the two match.
 */

import { writeFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Constants from config.ts
const ALPHA  = 0.9;
const BETA   = 10;
const LAMBDA = 1.0;  // bot has no loss aversion

// Payoff matrix [myAction][opponentAction], 0-indexed, R=0/P=1/S=2
// From agents.ts: payoff = [[TIE,LOSS,WIN],[WIN,TIE,LOSS],[LOSS,WIN,TIE]]
const PAYOFF = [
  [ 0, -1,  1],
  [ 1,  0, -1],
  [-1,  1,  0],
];

// Replicates agents.ts private softmax()
function softmax(v) {
  const s = v.map(x => BETA * x);
  const m = Math.max(...s);
  const e = s.map(x => Math.exp(x - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / sum);
}

// Replicates agents.ts private ev() — with lambda on negative payoffs
function ev(dist) {
  return PAYOFF.map(row =>
    row.reduce((s, v, j) => s + (v < 0 ? v * LAMBDA : v) * dist[j], 0)
  );
}

// Replicates agents.ts private probs() for a given level
function probs(attr, pAttr, level) {
  if (level === 0) return softmax(attr);
  const seed = level % 2 === 0 ? attr : pAttr;
  let p = softmax(ev(seed));
  for (let k = 1; k < level; k++) {
    p = softmax(ev(p));
  }
  return p;
}

// Fixed 40-trial sequence: [p_choice, bot_choice], 1-indexed (R=1, P=2, S=3)
// Shared with compare_ts_matlab.m — must be identical.
const SEQ = [
  [1,2],[1,3],[2,1],[3,2],[1,1],[2,3],[3,1],[1,2],[2,2],[3,3],
  [1,3],[2,1],[3,2],[1,1],[2,3],[3,3],[1,2],[2,1],[3,3],[1,2],
  [2,2],[3,1],[1,3],[2,2],[3,1],[1,1],[2,3],[3,2],[1,3],[2,1],
  [3,3],[1,2],[2,1],[3,1],[1,3],[2,2],[3,3],[1,1],[2,3],[3,2],
];

const results = [];

for (const level of [0, 1, 2]) {
  let attr  = [1/3, 1/3, 1/3];
  let pAttr = [1/3, 1/3, 1/3];

  for (let t = 0; t < SEQ.length; t++) {
    const [pChoice, botChoice] = SEQ[t];

    // Record pre-update state and probability vector
    results.push({
      trial:      t + 1,
      level,
      p_choice:   pChoice,
      bot_choice: botChoice,
      probs:      probs(attr, pAttr, level),
      attr:       [...attr],
      pAttr:      [...pAttr],
    });

    // Replicates agents.ts update(): 1-indexed choices → 0-indexed
    const botIdx = botChoice - 1;
    const pIdx   = pChoice   - 1;
    attr  = attr.map( (a, i) => a + ALPHA * ((i === botIdx ? 1 : 0) - a));
    pAttr = pAttr.map((a, i) => a + ALPHA * ((i === pIdx   ? 1 : 0) - a));
  }
}

const outPath = resolve(__dirname, 'ts_probs.json');
writeFileSync(outPath, JSON.stringify(results, null, 2));
console.log(`Written ${results.length} records → ${outPath}`);
