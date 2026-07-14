#!/usr/bin/env python3
"""
Generate parameter recovery data using our Python CHASEBot + CHASE generative model.

For each simulated participant:
  1. Sample known CHASE parameters (alpha, beta, lambda, gamma, kappa)
  2. Run CHASEBot at levels 0, 1, 2 (one block each, 40 trials) → opponent sequences
  3. Simulate participant choices using the Python CHASE generative model
  4. Save trial-by-trial data → analysis/recovery_input.csv

The CSV is then loaded by analysis/run_buergi_recovery.m which runs Buergi's
mn_fit to recover parameters and checks recovery quality.

Run from repo root:
  python3 analysis/generate_recovery_data.py
  python3 analysis/generate_recovery_data.py --n_per_kappa 30  # more sims
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.parameter_recovery_new_design import generate_session, generate_block_order

DEFAULT_N_PER_KAPPA = 20   # 60 total (20 per kappa level)
KAPPAS              = [0, 1, 2]
RNG_SEED            = 42

PAYOFF = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def sample_params(kappa: int, rng: np.random.Generator) -> dict:
    """Sample true CHASE parameters uniformly within Buergi's fitting bounds."""
    return {
        "alpha": float(rng.uniform(0.10, 0.95)),
        "beta":  float(rng.uniform(1.0,  15.0)),
        "lam":   float(rng.uniform(0.1,   3.0)) if kappa >= 1 else 1.0,
        "gamma": float(rng.uniform(0.1,   5.0)) if kappa >= 2 else 1.0,
        "kappa": kappa,
    }


def run(n_per_kappa: int, out_path: str) -> None:
    rng    = np.random.default_rng(RNG_SEED)
    rows   = []
    sim_id = 0

    for kappa in KAPPAS:
        for i in range(n_per_kappa):
            sim_id += 1
            true_params = sample_params(kappa, rng)

            # One block per level (0, 1, 2) — random permutation, no starting at 2
            block_seed  = int(rng.integers(1_000_000_000))
            block_order = generate_block_order(block_seed)

            # generate_session returns 0-indexed choices
            session_seed = int(rng.integers(1_000_000_000))
            p_choices, opp_choices, trial_in_block = generate_session(
                true_params, block_order, seed=session_seed,
            )

            n_total = len(p_choices)
            n_trials_per_block = n_total // len(block_order)

            bot_level_arr = np.array([
                block_order[b]["level"]
                for b in range(len(block_order))
                for _ in range(n_trials_per_block)
            ])
            block_arr = np.array([
                b + 1
                for b in range(len(block_order))
                for _ in range(n_trials_per_block)
            ])

            scores = np.array([PAYOFF[p, o] for p, o in zip(p_choices, opp_choices)])

            for t in range(n_total):
                rows.append({
                    "sim_id":       sim_id,
                    "true_kappa":   kappa,
                    "true_alpha":   true_params["alpha"],
                    "true_beta":    true_params["beta"],
                    "true_lambda":  true_params["lam"],
                    "true_gamma":   true_params["gamma"],
                    # Convert 0-indexed → 1-indexed for MATLAB (1=R,2=P,3=S)
                    "choice_own":   int(p_choices[t])   + 1,
                    "choice_other": int(opp_choices[t]) + 1,
                    "score_own":    int(scores[t]),
                    "bot_level":    int(bot_level_arr[t]),
                    "trial":        int(trial_in_block[t]),
                    "block":        int(block_arr[t]),
                    "missing":      0,
                })

            if (i + 1) % 5 == 0:
                print(f"  kappa={kappa}: {i+1}/{n_per_kappa} sims done")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    n_sims = sim_id
    print(f"\nGenerated {n_sims} simulations ({n_per_kappa} per kappa level × 3 levels)")
    print(f"  Trials per simulation: {n_total}")
    print(f"  Total rows: {len(df)}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_kappa", type=int, default=DEFAULT_N_PER_KAPPA)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "recovery_input.csv"))
    args = parser.parse_args()

    print(f"Generating recovery data: {args.n_per_kappa} sims per kappa level...")
    for kappa in KAPPAS:
        print(f"  kappa={kappa}")

    run(args.n_per_kappa, args.out)
