"""
Three-way comparison on Buergi et al.'s real 192-sim parameter recovery data:
ground truth vs their MATLAB-recovered estimates vs our Python-recovered
estimates (refit with block-reset support, see refit_with_python.py).

Applies the same NaN-masking they use (gamma excluded where kappa<2, lambda
excluded where kappa==0) so each parameter's r is computed on the subset
where it's actually identifiable by the model's own design.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), "refit_results.csv")
df = pd.read_csv(CSV_PATH)

PARAMS = ["alpha", "beta", "lam", "gamma", "kappa"]
LIMS = {
    "alpha": (0, 1), "beta": (0.4, 4.2), "lam": (0, 3.6),
    "gamma": (0, 10), "kappa": (-0.5, 3.5),
}

# Apply their exact masking logic, separately for each estimator's own kappa-based validity
for est_prefix in ["matlab", "our"]:
    kappa_col = f"{est_prefix}_kappa"
    df.loc[df[kappa_col] < 2, f"{est_prefix}_gamma"] = np.nan
    df.loc[df[kappa_col] == 0, f"{est_prefix}_lam"]  = np.nan


def pairwise_pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan, mask.sum()
    return pearsonr(x[mask], y[mask])[0], mask.sum()


fig, axes = plt.subplots(2, 5, figsize=(18, 7))
rng = np.random.default_rng(0)

for row, est_prefix, label in [(0, "matlab", "MATLAB (theirs)"), (1, "our", "Python (ours)")]:
    for col, p in enumerate(PARAMS):
        ax = axes[row, col]
        lims = LIMS[p]
        ax.plot(lims, lims, "--", color="gray", linewidth=1)

        x = df[f"gen_{p}"].values
        y = df[f"{est_prefix}_{p}"].values

        if p == "kappa":
            nx = rng.normal(0, 0.15, size=len(x))
            ny = rng.normal(0, 0.15, size=len(y))
            ax.scatter(x + nx, y + ny, alpha=0.25, s=30, color="steelblue", edgecolors="none")
        else:
            ax.scatter(x, y, alpha=0.25, s=30, color="steelblue", edgecolors="none")

        r, n = pairwise_pearson(x, y)
        ax.set_title(f"{p}\nr = {r:.2f} (n={n})", fontsize=10)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(f"{label}\nRecovered")
        if row == 1:
            ax.set_xlabel("Generating")

fig.suptitle("All 192 sims (includes kappa=0 ground truth, which our model cannot represent)", fontsize=12)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "compare_refit_results.png")
plt.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}\n")

# ── Second figure: restricted to true kappa >= 1 (the regime we implement) ──
df_sub = df[df["gen_kappa"] >= 1].copy()
df_sub.loc[df_sub["matlab_kappa"] < 2, "matlab_gamma"] = np.nan
df_sub.loc[df_sub["our_kappa"]    < 2, "our_gamma"]    = np.nan
df_sub.loc[df_sub["matlab_kappa"] == 0, "matlab_lam"]  = np.nan
df_sub.loc[df_sub["our_kappa"]    == 0, "our_lam"]     = np.nan

fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7))
for row, est_prefix, label in [(0, "matlab", "MATLAB (theirs)"), (1, "our", "Python (ours)")]:
    for col, p in enumerate(PARAMS):
        ax = axes2[row, col]
        lims = LIMS[p]
        ax.plot(lims, lims, "--", color="gray", linewidth=1)

        x = df_sub[f"gen_{p}"].values
        y = df_sub[f"{est_prefix}_{p}"].values

        if p == "kappa":
            nx = rng.normal(0, 0.15, size=len(x))
            ny = rng.normal(0, 0.15, size=len(y))
            ax.scatter(x + nx, y + ny, alpha=0.25, s=30, color="steelblue", edgecolors="none")
        else:
            ax.scatter(x, y, alpha=0.25, s=30, color="steelblue", edgecolors="none")

        r, n = pairwise_pearson(x, y)
        ax.set_title(f"{p}\nr = {r:.2f} (n={n})", fontsize=10)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(f"{label}\nRecovered")
        if row == 1:
            ax.set_xlabel("Generating")

fig2.suptitle("Restricted to true kappa >= 1 (n=144) -- the regime our model actually implements", fontsize=12)
plt.tight_layout()
out_path2 = os.path.join(os.path.dirname(__file__), "compare_refit_results_kappa_geq1.png")
plt.savefig(out_path2, dpi=150)
print(f"Saved -> {out_path2}\n")

print("All 192 sims (includes kappa=0 ground truth, which our model cannot represent):")
print(f"{'param':<8}{'MATLAB r':>10}{'(n)':>6}{'Python r':>12}{'(n)':>6}")
for p in PARAMS:
    r_m, n_m = pairwise_pearson(df[f"gen_{p}"].values, df[f"matlab_{p}"].values)
    r_o, n_o = pairwise_pearson(df[f"gen_{p}"].values, df[f"our_{p}"].values)
    print(f"{p:<8}{r_m:>10.2f}{n_m:>6}{r_o:>12.2f}{n_o:>6}")

print()
print("Restricted to true kappa >= 1 (n=144) -- the regime our model actually implements:")
sub = df[df["gen_kappa"] >= 1].copy()
sub.loc[sub["matlab_kappa"] < 2, "matlab_gamma"] = np.nan
sub.loc[sub["our_kappa"]    < 2, "our_gamma"]    = np.nan
sub.loc[sub["matlab_kappa"] == 0, "matlab_lam"]  = np.nan
sub.loc[sub["our_kappa"]    == 0, "our_lam"]     = np.nan
print(f"{'param':<8}{'MATLAB r':>10}{'(n)':>6}{'Python r':>12}{'(n)':>6}")
for p in PARAMS:
    r_m, n_m = pairwise_pearson(sub[f"gen_{p}"].values, sub[f"matlab_{p}"].values)
    r_o, n_o = pairwise_pearson(sub[f"gen_{p}"].values, sub[f"our_{p}"].values)
    print(f"{p:<8}{r_m:>10.2f}{n_m:>6}{r_o:>12.2f}{n_o:>6}")
