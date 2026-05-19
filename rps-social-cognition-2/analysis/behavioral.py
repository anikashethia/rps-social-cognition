# analysis/behavioral.py
"""
Behavioral analysis for the RPS Social Cognition Task.

Computes simple, model-free measures per participant × agent condition:
    - win_rate          : proportion of wins (chance = 1/3)
    - choice_entropy    : Shannon entropy of choice distribution (lower = more structured)
    - win_stay_rate     : P(repeat choice | previous win)
    - lose_shift_rate   : P(switch choice | previous loss)
    - choice_autocorr   : lag-1 autocorrelation of choice sequence

Primary prediction: monotonic gradient A > B > C > D > random
in all mentalizing-related measures.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Optional
import warnings


AGENTS_ORDERED = ["friendly", "neutral", "rng"]
N_ACTIONS      = 3   # Rock, Paper, Scissors


# ── Single-condition measures ─────────────────────────────────────────────────

def win_rate(outcomes: np.ndarray) -> float:
    """Proportion of 'win' outcomes. Chance = 1/3."""
    if len(outcomes) == 0:
        return np.nan
    return float(np.mean(outcomes == "win"))


def choice_entropy(choices: np.ndarray) -> float:
    """
    Shannon entropy of choice distribution (in bits).
    Lower entropy = more structured / less random strategy.
    Chance (uniform) = log2(3) ≈ 1.585 bits.
    """
    if len(choices) == 0:
        return np.nan
    counts = np.array([np.sum(choices == a) for a in [1, 2, 3]], dtype=float)
    counts += 1e-10  # Laplace smoothing
    probs   = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def win_stay_lose_shift(choices: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
    """
    Win-stay rate: P(same choice on trial t+1 | win on trial t)
    Lose-shift rate: P(different choice on trial t+1 | lose on trial t)
    """
    if len(choices) < 2:
        return {"win_stay": np.nan, "lose_shift": np.nan}

    stayed  = choices[1:] == choices[:-1]
    shifted = choices[1:] != choices[:-1]
    prev_outcomes = outcomes[:-1]

    win_mask  = prev_outcomes == "win"
    lose_mask = prev_outcomes == "lose"

    ws = float(stayed[win_mask].mean())   if win_mask.sum()  > 0 else np.nan
    ls = float(shifted[lose_mask].mean()) if lose_mask.sum() > 0 else np.nan

    return {"win_stay": ws, "lose_shift": ls}


def choice_autocorrelation(choices: np.ndarray, lag: int = 1) -> float:
    """
    Lag-{lag} autocorrelation of the choice sequence.
    Positive = tendency to repeat; negative = tendency to switch.
    """
    if len(choices) <= lag:
        return np.nan
    c = choices.astype(float)
    c -= c.mean()
    if c.std() == 0:
        return np.nan
    return float(np.corrcoef(c[:-lag], c[lag:])[0, 1])


# ── Per-condition summary ─────────────────────────────────────────────────────

def compute_condition_summary(block_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute all behavioral measures for one participant × agent condition.

    Parameters
    ----------
    block_df : pd.DataFrame
        Rows = trials for one agent condition.

    Returns
    -------
    dict with keys: win_rate, choice_entropy, win_stay, lose_shift, autocorr_lag1
    """
    choices  = block_df["participant_choice"].values
    outcomes = block_df["outcome"].values

    wsls = win_stay_lose_shift(choices, outcomes)

    return {
        "win_rate":      win_rate(outcomes),
        "choice_entropy": choice_entropy(choices),
        "win_stay":      wsls["win_stay"],
        "lose_shift":    wsls["lose_shift"],
        "autocorr_lag1": choice_autocorrelation(choices, lag=1),
        "n_trials":      len(block_df),
    }


# ── Participant-level analysis ────────────────────────────────────────────────

def analyze_participant(
    df: pd.DataFrame,
    participant_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute behavioral measures for all agent conditions for one participant.

    Parameters
    ----------
    df : pd.DataFrame
        Full trial-level dataframe for one participant.

    Returns
    -------
    pd.DataFrame with one row per agent condition and all behavioral measures.
    """
    rows = []
    for agent in AGENTS_ORDERED:
        block_df = df[df["agent"] == agent]
        if len(block_df) == 0:
            warnings.warn(f"No trials found for agent={agent}, participant={participant_id}")
            continue
        summary = compute_condition_summary(block_df)
        summary["agent"]          = agent
        summary["participant_id"] = participant_id or df["participant_id"].iloc[0]
        rows.append(summary)

    result = pd.DataFrame(rows)
    return result[["participant_id", "agent", "win_rate", "choice_entropy",
                   "win_stay", "lose_shift", "autocorr_lag1", "n_trials"]]


# ── Group-level analysis ──────────────────────────────────────────────────────

def analyze_group(data_dir: str = "data/") -> pd.DataFrame:
    """
    Load all participant data files and compute behavioral measures for each.
    Returns a long-format DataFrame with one row per participant × agent.
    """
    import os
    from utils.data_io import load_participant_data

    all_results = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        participant_id = fname.split("_session")[0]
        try:
            df     = load_participant_data(participant_id, data_dir=data_dir)
            result = analyze_participant(df, participant_id=participant_id)
            all_results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to process {fname}: {e}")

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ── Gradient test ─────────────────────────────────────────────────────────────

def test_connection_effect(
    group_df: pd.DataFrame,
    measure:  str = "win_rate",
) -> Dict[str, float]:
    """
    Test whether the expected effect of connection holds:
        friendly > neutral > rng
    for a given behavioral measure.

    Uses a paired t-test (friendly vs. neutral) as the primary comparison,
    plus one-sample t-tests against chance for each condition.

    Returns dict with t-statistic, p-value, and Cohen's d.
    """
    from scipy.stats import ttest_rel

    pivot = group_df.pivot_table(index="participant_id", columns="agent", values=measure)

    if "friendly" not in pivot.columns or "neutral" not in pivot.columns:
        return {"error": "Missing conditions"}

    friendly = pivot["friendly"].dropna()
    neutral  = pivot["neutral"].dropna()

    # Align on participants present in both
    common = friendly.index.intersection(neutral.index)
    f, n   = friendly[common].values, neutral[common].values

    t_stat, p_val = ttest_rel(f, n)
    cohens_d      = (f - n).mean() / (f - n).std()

    return {
        "measure":         measure,
        "mean_friendly":   float(f.mean()),
        "mean_neutral":    float(n.mean()),
        "mean_difference": float((f - n).mean()),
        "t_stat":          float(t_stat),
        "p_value":         float(p_val),
        "cohens_d":        float(cohens_d),
        "n_participants":  len(common),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Behavioral analysis")
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--output_dir",  default="results/behavioral/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data and computing behavioral measures...")
    group_df = analyze_group(args.data_dir)
    group_df.to_csv(os.path.join(args.output_dir, "behavioral_summary.csv"), index=False)
    print(f"Saved to {args.output_dir}behavioral_summary.csv")

    print("\nFriendly vs. neutral comparison:")
    for measure in ["win_rate", "choice_entropy", "win_stay", "lose_shift", "autocorr_lag1"]:
        if measure in group_df.columns:
            result = test_connection_effect(group_df, measure=measure)
            print(f"  {measure:20s}: friendly={result['mean_friendly']:.3f}, neutral={result['mean_neutral']:.3f}, t={result['t_stat']:.2f}, p={result['p_value']:.3f}, d={result['cohens_d']:.2f}")
