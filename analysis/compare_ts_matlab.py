#!/usr/bin/env python3
"""
compare_ts_matlab.py

Compares agents.ts vs mn_RPS_task.m probability vectors and tracker states.

Usage (run from repo root):
  python3 analysis/compare_ts_matlab.py

Expects:
  analysis/ts_probs.json    — produced by: node analysis/compare_ts_matlab.mjs
  analysis/matlab_probs.json — produced by running compare_ts_matlab.m in MATLAB
"""

import json
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_PATH     = os.path.join(ROOT, 'analysis', 'ts_probs.json')
MATLAB_PATH = os.path.join(ROOT, 'analysis', 'matlab_probs.json')
TOL = 1e-10

for path in (TS_PATH, MATLAB_PATH):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        print("Run the Node.js and MATLAB scripts first.")
        sys.exit(1)

with open(TS_PATH) as f:
    ts_data = json.load(f)
with open(MATLAB_PATH) as f:
    ml_data = json.load(f)

if len(ts_data) != len(ml_data):
    print(f"Length mismatch: TS has {len(ts_data)} records, MATLAB has {len(ml_data)}")
    sys.exit(1)

failures = []

for ts, ml in zip(ts_data, ml_data):
    if ts['trial'] != ml['trial'] or ts['level'] != ml['level']:
        print(f"Record order mismatch: TS trial={ts['trial']} level={ts['level']} "
              f"vs MATLAB trial={ml['trial']} level={ml['level']}")
        sys.exit(1)

    for key in ('probs', 'attr', 'pAttr'):
        ts_v = np.array(ts[key])
        ml_v = np.array(ml[key])
        if not np.allclose(ts_v, ml_v, atol=TOL, rtol=0):
            failures.append({
                'trial': ts['trial'], 'level': ts['level'], 'field': key,
                'ts': ts_v, 'matlab': ml_v, 'max_diff': float(np.abs(ts_v - ml_v).max()),
            })

n_checks = 3 * len(ts_data)

if not failures:
    print(f"All {n_checks} checks PASSED")
    print(f"TypeScript (agents.ts) and MATLAB (mn_RPS_task.m) match to {TOL:.0e}")
    print(f"  Checked: {len(ts_data)} records × 3 fields (probs, attr, pAttr)")
    print(f"  Levels tested: 0, 1, 2")
    print(f"  Trials: 40")
else:
    print(f"{len(failures)}/{n_checks} checks FAILED:\n")
    for f in failures:
        print(f"  trial={f['trial']} level={f['level']} field={f['field']}  "
              f"max_diff={f['max_diff']:.2e}")
        print(f"    TS:     {f['ts']}")
        print(f"    MATLAB: {f['matlab']}")
    sys.exit(1)
