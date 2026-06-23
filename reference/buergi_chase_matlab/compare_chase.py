"""
Cross-validation script: run our Python CHASE model (models/chase.py) on the
same fixed trial sequence and parameters as compare_chase.m, to check the
two implementations produce identical trial-by-trial outputs.

Run from anywhere: python3 reference/buergi_chase_matlab/compare_chase.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.chase import CHASEModel

# 0-indexed: Rock=0, Paper=1, Scissors=2 (matches MATLAB's 1=Rock,2=Paper,3=Scissors minus 1)
choices     = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 0])
opp_choices = np.array([1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 0, 2])

params = {"alpha": 0.3, "beta": 3.0, "gamma": 2.0, "lam": 1.5, "kappa": 3}

model  = CHASEModel()
result = model.simulate(params, choices, opp_choices)

lik = result.action_probs[np.arange(len(choices)), choices]

np.set_printoptions(precision=4, suppress=True)

print("lik (P(chosen action) per trial):")
print(lik)
print()
print("SV (subjective value per trial):")
print(result.choice_values)
print()
print("APE (action prediction error per trial):")
print(result.ape)
print()
print("KL_div / BU (belief update per trial):")
print(result.belief_updates)
print()
print("beliefs (POSTERIOR after each trial, rows=trials, cols=levels 0..kappa-1):")
print(result.beliefs)
