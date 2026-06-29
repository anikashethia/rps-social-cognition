"""
Parameter recovery for the new 6-block study design.

Design: 2 agents (friendly + neutral) × 3 CHASE levels (0, 1, 2) = 6 blocks,
        40 trials/block, random block order (Buergi et al. constraints).

For each simulation:
  1. Sample ground-truth CHASE parameters (alpha, beta, gamma, lam, kappa)
  2. Generate a random 6-block order
  3. Run the CHASEBot at each block's level to produce opponent choices
  4. Run the CHASE generative model (with ground-truth params) to produce
     participant choices, resetting state at each block boundary
  5. Fit CHASE (with block resets) to recover parameters
  6. Report Pearson r per parameter and compare to Buergi's published values

Key differences from the old parameter_recovery.py:
  - Opponent is CHASEBot (alpha=0.9, beta=10, WSLS noise, dual-tracker)
    at the level assigned to each block -- not a fixed level-1 agent
  - Block structure: 6 blocks × 40 trials, not variable trial counts
  - kappa={0,1,2,3,4} (includes egocentric case)
  - trial_in_block passed to fit() so the model resets between blocks
  - NaN masking: gamma excluded where fitted kappa<2; lam excluded where kappa=0

Run from repo root (takes ~1 hour for 200 sims):
  python3 analysis/parameter_recovery_new_design.py
  python3 analysis/parameter_recovery_new_design.py --n_sims 50  # quick check
"""

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.chase import CHASEModel

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

N_SIMS_DEFAULT    = 200      # simulations (40 per kappa level × 5 levels)
N_BLOCKS          = 6
N_TRIALS_PER_BLOCK = 40
N_RESTARTS        = 5
ALL_PARAMS        = ["alpha", "beta", "gamma", "lam", "kappa"]

BUERGI_R = {"alpha": 0.80, "beta": 0.88, "lam": 0.86, "gamma": 0.73, "kappa": 1.00}


# ── CHASEBot (Python mirror of agents.ts) ─────────────────────────────────────

class CHASEBot:
    """Matches Buergi et al. mn_RPS_task.m exactly (see analysis/verify_agent.py)."""

    PAYOFF = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

    def __init__(self, level: int, alpha: float = 0.9, beta: float = 10.0,
                 lam: float = 1.0, seed=None):
        self.level  = level
        self.alpha  = alpha
        self.beta   = beta
        self.lam    = lam
        self.rng    = np.random.default_rng(seed)
        self.attr   = np.ones(3) / 3
        self.p_attr = np.ones(3) / 3
        self.scores: list[int] = []
        self.last_was_noisy = False

    def _softmax(self, v):
        z = self.beta * v; z -= z.max()
        e = np.exp(z); return e / e.sum()

    def _ev(self, dist):
        p = self.PAYOFF.copy(); p[p < 0] *= self.lam
        return p @ dist

    def _probs(self):
        if self.level == 0:
            return self._softmax(self.attr)
        seed = self.attr if self.level % 2 == 0 else self.p_attr
        p = self._softmax(self._ev(seed))
        for _ in range(1, self.level):
            p = self._softmax(self._ev(p))
        return p

    def _is_noisy(self):
        if not self.scores:
            return False
        time_horizon   = 5
        success_crit   = 0.5
        skewness       = 1.3
        max_win_streak = 2 if self.level == 0 else 3
        streak_max     = int(self.rng.integers(3, 5))

        recent  = np.array(self.scores[-time_horizon:])
        success = float(np.mean(recent > 0))

        win_streak = lose_streak = tie_streak = 0
        for s in reversed(self.scores):
            if s > 0:
                win_streak += 1
                if win_streak >= max_win_streak: break
            else: break
        for s in reversed(self.scores):
            if s < 0:
                lose_streak += 1
                if lose_streak >= streak_max: break
            else: break
        for s in reversed(self.scores):
            if s == 0:
                tie_streak += 1
                if tie_streak >= streak_max: break
            else: break

        if lose_streak >= streak_max or tie_streak >= streak_max:
            return True
        if success >= success_crit and win_streak >= 1:
            if win_streak >= max_win_streak:
                return True
            chance = (1 / (max_win_streak + 1 - win_streak)) ** skewness
            if self.rng.random() < chance:
                return True
        return False

    def choose(self):
        p = self._probs().copy()
        self.last_was_noisy = self._is_noisy()
        if self.last_was_noisy:
            p[np.argmax(p)] = 0; p /= p.sum()
        return int(self.rng.choice(3, p=p))

    def update(self, own: int, opp: int, score: int | None = None):
        ind = np.zeros(3); ind[own] = 1
        self.attr += self.alpha * (ind - self.attr)
        ind = np.zeros(3); ind[opp] = 1
        self.p_attr += self.alpha * (ind - self.p_attr)
        if score is not None:
            self.scores.append(score)


# ── Block order (mirrors backend/app/routers/sessions.py) ─────────────────────

def generate_block_order(seed: int) -> list[dict]:
    """
    Random 6-block order: all (condition, level) combos exactly once,
    first block not level 2, no two consecutive blocks share a level.
    """
    rng = random.Random(seed)
    all_blocks = [
        {"condition": c, "level": l}
        for c in ("friendly", "neutral")
        for l in (0, 1, 2)
    ]
    while True:
        blocks = all_blocks.copy()
        rng.shuffle(blocks)
        if blocks[0]["level"] >= 2:
            continue
        if any(blocks[i]["level"] == blocks[i + 1]["level"] for i in range(5)):
            continue
        return blocks


# ── Generative model ──────────────────────────────────────────────────────────

def generate_session(
    true_params: dict,
    block_order: list[dict],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one participant session using ground-truth CHASE parameters.

    The opponent at each block is a CHASEBot at that block's level (0/1/2).
    Participant state (attractions, beliefs) resets to uniform at each block
    boundary, matching Buergi et al.'s fitting methodology.

    Returns
    -------
    p_choices    : (240,) participant choices, 0-indexed
    opp_choices  : (240,) opponent choices, 0-indexed
    trial_in_block : (240,) 1-indexed trial-within-block counter
    """
    model = CHASEModel()
    rng   = np.random.default_rng(seed)

    alpha = true_params["alpha"]
    beta  = true_params["beta"]
    gamma = true_params["gamma"]
    lam   = true_params["lam"]
    kappa = int(round(true_params["kappa"]))
    kappa = max(0, min(kappa, model.max_kappa))

    n_belief_levels = max(kappa, 1)
    payoff_scaled   = model.PAYOFF.copy()
    payoff_scaled[payoff_scaled == -1] *= lam

    all_p   = []
    all_opp = []
    all_tib = []

    for block in block_order:
        bot = CHASEBot(level=block["level"], seed=int(rng.integers(1_000_000_000)))

        # Block reset: uniform attractions and beliefs
        own_attr = np.ones(3) / 3.0
        opp_attr = np.ones(3) / 3.0
        beliefs  = np.ones(n_belief_levels) / n_belief_levels

        for t in range(N_TRIALS_PER_BLOCK):

            # ── Participant chooses ───────────────────────────────────────────
            if kappa == 0:
                probs = model._softmax(own_attr, beta)
            else:
                integrated_opp = np.zeros(3)
                for k in range(kappa):
                    integrated_opp += beliefs[k] * model._recursive_probs(
                        own_attr, opp_attr, beta, k, lam
                    )
                probs = model._softmax(payoff_scaled @ integrated_opp, beta)

            p_choice   = int(rng.choice(3, p=probs))
            opp_choice = bot.choose()

            all_p.append(p_choice)
            all_opp.append(opp_choice)
            all_tib.append(t + 1)

            # ── Update beliefs (kappa ≥ 2 only) ──────────────────────────────
            if kappa >= 2:
                beliefs, _ = model._belief_update(
                    beliefs, own_attr, opp_attr, opp_choice, beta, gamma, lam, kappa
                )

            # ── Update attractions ────────────────────────────────────────────
            ind = np.zeros(3); ind[p_choice] = 1.0
            own_attr += alpha * (ind - own_attr)

            ind = np.zeros(3); ind[opp_choice] = 1.0
            opp_attr += alpha * (ind - opp_attr)

            # ── Update bot (own choice, participant choice, participant score) ─
            score = int(model.PAYOFF[p_choice, opp_choice])
            bot.update(opp_choice, p_choice, score)

    return np.array(all_p), np.array(all_opp), np.array(all_tib)


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_true_params(rng: np.random.Generator, kappa: int) -> dict:
    """Sample realistic ground-truth parameters for a given kappa."""
    return {
        "alpha": rng.uniform(0.05, 0.95),
        "beta":  rng.uniform(0.5,  15.0),
        "gamma": rng.uniform(0.1,  10.0),
        "lam":   rng.uniform(0.1,   3.5),
        "kappa": kappa,
    }


# ── Single recovery run ───────────────────────────────────────────────────────

def run_single(sim_id: int, kappa: int, rng: np.random.Generator) -> dict | None:
    seed        = int(rng.integers(0, 2**31))
    true_params = sample_true_params(rng, kappa)
    block_order = generate_block_order(seed)

    p_choices, opp_choices, tib = generate_session(true_params, block_order, seed)

    model  = CHASEModel()
    result = model.fit(
        p_choices   + 1,   # convert to 1-indexed for model.fit()
        opp_choices + 1,
        n_restarts     = N_RESTARTS,
        seed           = seed,
        trial_in_block = tib,
    )

    if not result.converged:
        return None

    row = {"sim_id": sim_id, "true_kappa": kappa}
    for p in ALL_PARAMS:
        row[f"true_{p}"]      = true_params[p]
        row[f"recovered_{p}"] = getattr(result, p if p != "lam" else "lam")
    return row


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(n_sims: int, seed: int, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rng       = np.random.default_rng(seed)
    rows      = []
    kappa_vals = [0, 1, 2, 3, 4]
    sims_per_kappa = n_sims // len(kappa_vals)
    total     = sims_per_kappa * len(kappa_vals)

    print(f"Design:  {N_BLOCKS} blocks × {N_TRIALS_PER_BLOCK} trials = "
          f"{N_BLOCKS * N_TRIALS_PER_BLOCK} trials/participant")
    print(f"Sims:    {sims_per_kappa} per kappa level × {len(kappa_vals)} levels = {total}")
    print(f"Fitting: {N_RESTARTS} random restarts per sim\n")

    t0   = time.time()
    done = 0

    for kappa in kappa_vals:
        converged = 0
        for i in range(sims_per_kappa):
            sim_id = kappa * sims_per_kappa + i
            row    = run_single(sim_id, kappa, rng)
            if row is not None:
                rows.append(row)
                converged += 1
            done += 1

            if done % 10 == 0 or done == total:
                elapsed   = time.time() - t0
                rate      = elapsed / done
                remaining = rate * (total - done)
                print(f"  {done:3d}/{total}  kappa={kappa}  "
                      f"converged={converged}/{i+1}  "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"remaining≈{remaining/60:.1f}min",
                      flush=True)

        print(f"  kappa={kappa}: {converged}/{sims_per_kappa} converged\n")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "recovery_new_design.csv")
    df.to_csv(out_csv, index=False)
    print(f"Raw results → {out_csv}")
    return df


# ── Analysis & plotting ───────────────────────────────────────────────────────

def analyse_and_plot(df: pd.DataFrame, output_dir: str):
    rng = np.random.default_rng(0)

    # ── Apply NaN masking (same logic as Buergi's results_and_figures.m) ──────
    df = df.copy()
    df.loc[df["recovered_kappa"] < 2,  "recovered_gamma"] = np.nan
    df.loc[df["recovered_kappa"] == 0, "recovered_lam"]   = np.nan

    def pearson_masked(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return np.nan, mask.sum()
        return pearsonr(x[mask], y[mask])[0], int(mask.sum())

    lims = {
        "alpha": (0, 1), "beta": (0, 15), "gamma": (0, 10),
        "lam": (0, 3.5), "kappa": (-0.5, 4.5),
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    print("\n── Parameter recovery (Pearson r) ──────────────────────────────")
    print(f"{'param':<8}  {'r':>6}  {'n':>5}  {'Buergi r':>9}")
    print("-" * 36)

    for ax, param in zip(axes, ALL_PARAMS):
        x = df[f"true_{param}"].values
        y = df[f"recovered_{param}"].values
        r, n = pearson_masked(x, y)

        lim = lims[param]
        ax.plot(lim, lim, "--", color="gray", linewidth=1)

        if param == "kappa":
            jx = x + rng.normal(0, 0.15, size=len(x))
            jy = y + rng.normal(0, 0.15, size=len(y))
            ax.scatter(jx, jy, alpha=0.25, s=30, color="steelblue", edgecolors="none")
        else:
            mask = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(x[mask], y[mask], alpha=0.25, s=30,
                       color="steelblue", edgecolors="none")

        r_str = f"{r:.2f}" if not np.isnan(r) else "nan"
        ax.set_title(f"{param}\nr = {r_str}  (n={n})", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True"); ax.set_ylabel("Recovered")
        ax.set_aspect("equal")

        buergi_r = BUERGI_R.get(param, np.nan)
        print(f"  {param:<6}  {r_str:>6}  {n:>5}  {buergi_r:>9.2f}")

    fig.suptitle(
        f"Parameter recovery — new 6-block design "
        f"({N_BLOCKS} blocks × {N_TRIALS_PER_BLOCK} trials, n={len(df)})",
        fontsize=12,
    )
    plt.tight_layout()
    out_png = os.path.join(output_dir, "recovery_new_design.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nFigure → {out_png}")

    # ── Per-kappa breakdown ───────────────────────────────────────────────────
    print("\n── Per-kappa recovery (Pearson r, continuous params only) ──────")
    print(f"{'kappa':<7}", end="")
    for p in ["alpha", "beta", "gamma", "lam"]:
        print(f"  {p:>7}", end="")
    print(f"  {'n':>4}")
    print("-" * 50)

    for k in sorted(df["true_kappa"].unique()):
        sub = df[df["true_kappa"] == k].copy()
        print(f"  k={int(k):<4}", end="")
        for p in ["alpha", "beta", "gamma", "lam"]:
            r, _ = pearson_masked(sub[f"true_{p}"].values,
                                   sub[f"recovered_{p}"].values)
            print(f"  {r:>7.2f}" if not np.isnan(r) else "     nan", end="")
        print(f"  {len(sub):>4}")

    # ── kappa confusion matrix ────────────────────────────────────────────────
    print("\n── kappa confusion (true rows × recovered cols) ─────────────────")
    kappa_range = range(5)
    header = "true \\ rec" + "".join(f"  k={k}" for k in kappa_range)
    print(header)
    for kt in kappa_range:
        sub = df[df["true_kappa"] == kt]
        if len(sub) == 0:
            continue
        row_str = f"  true k={kt}  "
        for kr in kappa_range:
            count = (sub["recovered_kappa"] == kr).sum()
            row_str += f"  {count:3d}"
        print(row_str)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims",     type=int, default=N_SIMS_DEFAULT,
                        help="Total simulations (split evenly across kappa 0-4). Default: 200")
    parser.add_argument("--output_dir", default="results/recovery_new_design/")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    df = run_simulation(args.n_sims, args.seed, args.output_dir)
    analyse_and_plot(df, args.output_dir)
    print("\nDone.")
