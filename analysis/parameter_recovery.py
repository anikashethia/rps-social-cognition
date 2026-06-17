"""
CHASE parameter recovery simulation.

Answers: how many trials per block are needed for stable CHASE parameter
estimation? Generates synthetic data with known parameters, fits CHASE,
and measures how well the true parameters are recovered across a range of
trial counts.

Usage:
    python analysis/parameter_recovery.py
    python analysis/parameter_recovery.py --n_sims 200 --output_dir results/recovery/

Output:
    results/recovery/recovery_raw.csv       — one row per simulation
    results/recovery/recovery_summary.csv   — correlation + RMSE per param × trial count
    results/recovery/recovery_plots.png     — visualisation
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Optional

from models.chase import CHASEModel, PARAM_BOUNDS

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

TRIAL_COUNTS   = [10, 15, 20, 25, 30, 40]
N_SIMS_DEFAULT = 100        # simulations per trial count
N_RESTARTS     = 5          # MLE restarts per fit (keep low for speed)
CONTINUOUS_PARAMS = ["alpha", "beta", "gamma", "lam"]
ALL_PARAMS        = ["alpha", "beta", "gamma", "lam", "kappa"]


# ── Opponent: fixed CHASE agent (matching task implementation) ────────────────

class OpponentCHASE:
    """Level-1 CHASE opponent, matching agents.ts implementation."""

    PAYOFF = np.array([
        [ 0, -1,  1],
        [ 1,  0, -1],
        [-1,  1,  0],
    ], dtype=float)

    def __init__(self, seed: int):
        self.rng   = np.random.default_rng(seed)
        self.attr  = np.ones(3) / 3.0
        self.alpha = 0.3
        self.beta  = 3.0
        self.noise = 0.15

    def _softmax(self, x):
        e = np.exp(self.beta * x - np.max(self.beta * x))
        return e / e.sum()

    def choose(self) -> int:
        p0  = self._softmax(self.attr)
        ev  = self.PAYOFF @ p0
        p1  = self._softmax(ev)
        if self.rng.random() < self.noise:
            return int(self.rng.integers(3))
        return int(self.rng.choice(3, p=p1))

    def update(self, own_action: int):
        indicator        = np.zeros(3)
        indicator[own_action] = 1.0
        self.attr       += self.alpha * (indicator - self.attr)


# ── Data generation ───────────────────────────────────────────────────────────

def generate_choices(
    true_params: dict,
    n_trials:    int,
    seed:        int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic participant and opponent choices using CHASE.

    Returns
    -------
    participant_choices : np.ndarray, shape (n_trials,), 0-indexed
    opponent_choices    : np.ndarray, shape (n_trials,), 0-indexed
    """
    rng      = np.random.default_rng(seed)
    opponent = OpponentCHASE(seed=seed + 1)
    model    = CHASEModel()

    alpha = true_params["alpha"]
    beta  = true_params["beta"]
    gamma = true_params["gamma"]
    lam   = true_params["lam"]
    kappa = int(round(true_params["kappa"]))
    kappa = max(1, min(kappa, model.max_kappa))

    attractions = np.ones(3) / 3.0
    beliefs     = np.ones(kappa) / kappa

    participant_choices = np.zeros(n_trials, dtype=int)
    opponent_choices    = np.zeros(n_trials, dtype=int)

    payoff_scaled = model.PAYOFF.copy()
    payoff_scaled[payoff_scaled == -1] *= lam

    for t in range(n_trials):
        # ── Participant chooses ───────────────────────────────────────────────
        integrated_opp = np.zeros(3)
        for k in range(kappa):
            integrated_opp += beliefs[k] * model._recursive_probs(
                attractions, beta, k, lam
            )

        ev_p   = payoff_scaled @ integrated_opp
        probs  = model._softmax(ev_p, beta)
        p_choice = int(rng.choice(3, p=probs))

        # ── Opponent chooses ──────────────────────────────────────────────────
        opp_choice = opponent.choose()

        participant_choices[t] = p_choice
        opponent_choices[t]    = opp_choice

        # ── Update state ──────────────────────────────────────────────────────
        beliefs, _ = model._belief_update(
            beliefs, attractions, opp_choice, beta, gamma, lam, kappa
        )

        indicator      = np.zeros(3)
        indicator[p_choice] = 1.0
        attractions   += alpha * (indicator - attractions)

        opponent.update(opp_choice)

    return participant_choices, opponent_choices


# ── Single recovery run ───────────────────────────────────────────────────────

def sample_true_params(rng: np.random.Generator) -> dict:
    """Sample a realistic set of true parameters."""
    return {
        "alpha": rng.uniform(0.05, 0.8),
        "beta":  rng.uniform(0.5, 10.0),
        "gamma": rng.uniform(0.5, 10.0),
        "lam":   rng.uniform(0.2, 3.0),
        "kappa": int(rng.integers(1, 5)),   # 1, 2, 3, or 4
    }


def run_single(
    true_params: dict,
    n_trials:    int,
    sim_id:      int,
    seed:        int,
) -> Optional[dict]:
    """
    Generate data with true_params, fit CHASE, return recovery row.
    Returns None if fitting fails.
    """
    p_choices, opp_choices = generate_choices(true_params, n_trials, seed)

    model  = CHASEModel()
    result = model.fit(
        p_choices + 1,
        opp_choices + 1,
        n_restarts=N_RESTARTS,
        seed=seed,
    )

    if not result.converged:
        return None

    row = {
        "sim_id":   sim_id,
        "n_trials": n_trials,
    }
    for p in ALL_PARAMS:
        row[f"true_{p}"]      = true_params[p]
        row[f"recovered_{p}"] = getattr(result, p if p != "lam" else "lam")

    return row


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(
    n_sims:      int = N_SIMS_DEFAULT,
    trial_counts: list[int] = TRIAL_COUNTS,
    seed:        int = 42,
    output_dir:  str = "results/recovery/",
) -> pd.DataFrame:
    """Run full parameter recovery simulation and save results."""
    os.makedirs(output_dir, exist_ok=True)
    rng    = np.random.default_rng(seed)
    rows   = []
    total  = len(trial_counts) * n_sims

    print(f"Running {n_sims} simulations × {len(trial_counts)} trial counts = {total} fits")
    print(f"Parameters: {ALL_PARAMS}\n")

    done = 0
    for n_trials in trial_counts:
        converged = 0
        for sim_id in range(n_sims):
            sim_seed   = int(rng.integers(0, 2**31))
            true_params = sample_true_params(rng)

            row = run_single(true_params, n_trials, sim_id, sim_seed)
            if row is not None:
                rows.append(row)
                converged += 1

            done += 1
            if done % 50 == 0:
                pct = 100 * done / total
                print(f"  {done}/{total} ({pct:.0f}%)  — n_trials={n_trials}, "
                      f"converged={converged}/{sim_id+1}")

        print(f"n_trials={n_trials:3d}: {converged}/{n_sims} fits converged")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "recovery_raw.csv"), index=False)
    print(f"\nRaw results saved → {output_dir}recovery_raw.csv")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Compute correlation and RMSE per parameter × trial count."""
    records = []
    for n_trials, grp in df.groupby("n_trials"):
        for param in ALL_PARAMS:
            true = grp[f"true_{param}"].values
            rec  = grp[f"recovered_{param}"].values

            if len(true) < 3:
                continue

            if param == "kappa":
                # Kappa is discrete — use % exactly correct and % within 1
                pct_exact  = float(np.mean(true == rec) * 100)
                pct_within = float(np.mean(np.abs(true - rec) <= 1) * 100)
                r, p_val   = pearsonr(true, rec) if len(np.unique(true)) > 1 else (np.nan, np.nan)
                records.append({
                    "n_trials":   n_trials,
                    "param":      param,
                    "r":          r,
                    "p_val":      p_val,
                    "rmse":       float(np.sqrt(np.mean((true - rec) ** 2))),
                    "pct_exact":  pct_exact,
                    "pct_within1": pct_within,
                    "n":          len(true),
                })
            else:
                r, p_val = pearsonr(true, rec)
                records.append({
                    "n_trials":    n_trials,
                    "param":       param,
                    "r":           float(r),
                    "p_val":       float(p_val),
                    "rmse":        float(np.sqrt(np.mean((true - rec) ** 2))),
                    "pct_exact":   np.nan,
                    "pct_within1": np.nan,
                    "n":           len(true),
                })

    summary = pd.DataFrame(records).sort_values(["param", "n_trials"])
    summary.to_csv(os.path.join(output_dir, "recovery_summary.csv"), index=False)
    print(f"Summary saved → {output_dir}recovery_summary.csv")
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_recovery(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str):
    """
    Two-panel figure:
    Left:  Scatter plots of true vs recovered for each parameter at each trial count
    Right: Recovery correlation (r) vs trial count per parameter
    """
    n_params = len(ALL_PARAMS)
    n_counts = len(TRIAL_COUNTS)

    # ── Figure 1: correlation vs trial count (one line per param) ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.6, n_params))

    for param, color in zip(ALL_PARAMS, colors):
        sub   = summary[summary["param"] == param].sort_values("n_trials")
        label = f"kappa (% exact)" if param == "kappa" else param
        y     = sub["pct_exact"] / 100 if param == "kappa" else sub["r"]
        ax.plot(sub["n_trials"], y, marker="o", label=label, color=color)

    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="r = 0.8 threshold")
    ax.axvline(20,  color="black", linestyle=":",  linewidth=1.5, label="20 trials (planned)")
    ax.set_xlabel("Trials per block", fontsize=12)
    ax.set_ylabel("Recovery (Pearson r  |  % exact for κ)", fontsize=12)
    ax.set_title("CHASE Parameter Recovery vs Trial Count", fontsize=13, fontweight="bold")
    ax.set_xticks(TRIAL_COUNTS)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_correlation.png"), dpi=150)
    plt.close()
    print(f"Correlation plot saved → {output_dir}recovery_correlation.png")

    # ── Figure 2: scatter true vs recovered at 20 trials ─────────────────────
    df20    = df[df["n_trials"] == 20]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    for ax, param in zip(axes, ALL_PARAMS):
        true = df20[f"true_{param}"].values
        rec  = df20[f"recovered_{param}"].values

        ax.scatter(true, rec, alpha=0.4, s=20, color="steelblue")
        mn, mx = min(true.min(), rec.min()), max(true.max(), rec.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)

        if len(np.unique(true)) > 1:
            r, _ = pearsonr(true, rec)
            ax.set_title(f"{param}\nr = {r:.2f}", fontsize=11)
        else:
            ax.set_title(f"{param}", fontsize=11)

        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Recovered", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("True vs Recovered Parameters (20 trials/block)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_scatter_20trials.png"), dpi=150)
    plt.close()
    print(f"Scatter plot saved → {output_dir}recovery_scatter_20trials.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHASE parameter recovery simulation")
    parser.add_argument("--n_sims",      type=int, default=N_SIMS_DEFAULT,
                        help="Simulations per trial count (default: 100)")
    parser.add_argument("--output_dir",  default="results/recovery/")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    df      = run_simulation(args.n_sims, TRIAL_COUNTS, args.seed, args.output_dir)
    summary = summarise(df, args.output_dir)

    print("\n── Recovery summary (Pearson r) ──")
    pivot = summary.pivot(index="n_trials", columns="param", values="r").round(3)
    print(pivot.to_string())

    plot_recovery(df, summary, args.output_dir)
    print("\nDone.")
