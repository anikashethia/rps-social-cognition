"""
Recreate Buergi et al.'s Supplementary Figure 3 (CHASE parameter recovery)
directly from their own data file (results/supplementary/parameter_recovery.mat),
following the exact plotting logic in BAKR_2024_results_and_figures.m
(the "%% param recovery" section).

This uses ONLY their precomputed ground-truth (prec.params.gen) and recovered
(prec.params.est) values -- no refitting, no MATLAB needed. It's a sanity check
that we're reading/interpreting their data structure correctly, by checking we
reproduce their reported r = 0.80 / 0.88 / 0.86 / 0.73 / 1.00.
"""

import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

MAT_PATH = "/tmp/parameter_recovery.mat"

m    = sio.loadmat(MAT_PATH, simplify_cells=True)
prec = m["prec"]

# MATLAB column order: beta, lambda, gamma, alpha, kappa (1-indexed 1..5)
gen = np.array(prec["params"]["gen"])  # shape (192, 5)
est = np.array(prec["params"]["est"]).copy()

BETA, LAM, GAMMA, ALPHA, KAPPA = 0, 1, 2, 3, 4

# ── Apply their exact NaN-masking (matches lines 998-999 of their script) ──
est[est[:, KAPPA] < 2, GAMMA] = np.nan   # gamma has no effect if kappa < 2
est[est[:, KAPPA] == 0, LAM]  = np.nan   # lambda has no effect if kappa == 0

# Panel order + axis limits, matching their figure exactly
panel_order = [ALPHA, BETA, LAM, GAMMA, KAPPA]
panel_names = ["alpha", "beta", "lambda", "gamma", "kappa"]
param_lims  = {
    ALPHA: (0, 1),
    BETA:  (0.4, 4.2),
    LAM:   (0, 3.6),
    GAMMA: (0, 10),
    KAPPA: (-0.5, 3.5),
}


def pairwise_pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[mask], y[mask])[0], mask.sum()


fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
rng = np.random.default_rng(0)

for ax, p_idx, name in zip(axes, panel_order, panel_names):
    lims = param_lims[p_idx]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)

    x, y = gen[:, p_idx], est[:, p_idx]
    if p_idx == KAPPA:
        noise_x = rng.normal(0, 0.15, size=len(x))
        noise_y = rng.normal(0, 0.15, size=len(y))
        ax.scatter(x + noise_x, y + noise_y, alpha=0.25, s=40, color="steelblue", edgecolors="none")
    else:
        ax.scatter(x, y, alpha=0.25, s=40, color="steelblue", edgecolors="none")

    r, n = pairwise_pearson(x, y)
    ax.set_title(f"{name}\nr = {r:.2f}  (n={n})", fontsize=11)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Generating")
    if p_idx == ALPHA:
        ax.set_ylabel("Recovered")
    ax.set_aspect("equal")

    if p_idx == GAMMA:
        ax.add_patch(plt.Rectangle((0, 0), 2.5, 2.5, fill=False, edgecolor="black", linewidth=0.8))

plt.tight_layout()
plt.savefig("reference/buergi_chase_matlab/recreated_fig_param_recovery.png", dpi=150)
print("Saved -> reference/buergi_chase_matlab/recreated_fig_param_recovery.png")
print()

print("Reported in paper:   alpha=0.80  beta=0.88  lambda=0.86  gamma=0.73  kappa=1.00")
print("Reproduced here:     ", end="")
for p_idx, name in zip(panel_order, panel_names):
    r, n = pairwise_pearson(gen[:, p_idx], est[:, p_idx])
    print(f"{name}={r:.2f}", end="  ")
print()
print()
print(f"Fraction of generating gamma <= 2.5: {np.mean(gen[:, GAMMA] <= 2.5):.3f}  (paper reports 0.77)")
