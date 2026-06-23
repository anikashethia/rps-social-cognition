"""
Refit all 192 real parameter-recovery sequences from Buergi et al.'s
results/supplementary/parameter_recovery.mat using our own Python CHASE
model (with block-reset support), and compare against:
  - the ground truth used to generate the data (prec.params.gen)
  - their own MATLAB-recovered estimates (prec.params.est)

This is the real fidelity test: same data, same ground truth, our fitting
pipeline vs theirs.

Run from repo root: python3 reference/buergi_chase_matlab/refit_with_python.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.chase import CHASEModel

MAT_PATH   = "/tmp/parameter_recovery.mat"
OUT_CSV    = os.path.join(os.path.dirname(__file__), "refit_results.csv")
N_RESTARTS = 5

PARAM_NAMES = ["beta", "lam", "gamma", "alpha", "kappa"]  # matches gen/est column order

m    = sio.loadmat(MAT_PATH, simplify_cells=True)
prec = m["prec"]
sim  = m["sim"]

gen  = np.array(prec["params"]["gen"])   # (192, 5)
est  = np.array(prec["params"]["est"])   # (192, 5)
subj = sim["subj"]                        # list of 192 dicts

model = CHASEModel()
rows  = []

t_start = time.time()
for i, s in enumerate(subj):
    d   = s["data"]
    ch  = np.array(d["choice_own"])
    opp = np.array(d["choice_other"])
    tib = np.array(d["trial"])

    result = model.fit(ch, opp, n_restarts=N_RESTARTS, seed=i, trial_in_block=tib)

    row = {"sim_idx": i, "converged": result.converged}
    for j, name in enumerate(PARAM_NAMES):
        row[f"gen_{name}"]     = gen[i, j]
        row[f"matlab_{name}"]  = est[i, j]
    row["our_beta"]  = result.beta
    row["our_lam"]   = result.lam
    row["our_gamma"] = result.gamma
    row["our_alpha"] = result.alpha
    row["our_kappa"] = result.kappa
    rows.append(row)

    elapsed = time.time() - t_start
    if (i + 1) % 10 == 0 or i == len(subj) - 1:
        rate = elapsed / (i + 1)
        remaining = rate * (len(subj) - i - 1)
        print(f"{i+1}/{len(subj)}  elapsed={elapsed/60:.1f}min  "
              f"est. remaining={remaining/60:.1f}min", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")
