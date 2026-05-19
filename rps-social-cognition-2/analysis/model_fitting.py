# analysis/model_fitting.py
"""
Model fitting pipeline — fits CHASE (and alternative models) to each
participant × agent condition using MLE and computes AIC for comparison.

Usage:
    python analysis/model_fitting.py --data_dir data/ --output_dir results/models/
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from models.chase import CHASEModel, CHASEResult
from models.alternatives import (
    RLModel, FictitiousPlayModel, EWAModel, SelfTuningEWAModel, ToMkModel
)
from utils.data_io import load_participant_data


AGENTS_ORDERED = ["friendly", "neutral", "rng"]

MODEL_REGISTRY = {
    "chase": CHASEModel,
    "rl":    RLModel,
    "fp":    FictitiousPlayModel,
    "ewa":   EWAModel,
    "ewa_s": SelfTuningEWAModel,
    "tomk":  ToMkModel,
}


# ── Single participant × agent ────────────────────────────────────────────────

def fit_participant_condition(
    model_cls,
    df:             pd.DataFrame,
    participant_id: str,
    agent_id:       str,
    n_restarts:     int = 10,
    seed:           Optional[int] = None,
) -> dict:
    """
    Fit one model to one participant × agent condition.

    Returns a flat dict with: participant_id, agent, model,
    all fitted params, log_likelihood, aic, n_trials, converged.
    """
    block_df = df[df["agent"] == agent_id].sort_values("trial")

    if len(block_df) < 5:
        warnings.warn(f"Too few trials: {participant_id} / {agent_id}")
        return {}

    choices          = block_df["participant_choice"].values.astype(int)
    opponent_choices = block_df["agent_choice"].values.astype(int)

    try:
        model  = model_cls()
        result = model.fit(choices, opponent_choices, n_restarts=n_restarts, seed=seed)
    except Exception as e:
        warnings.warn(f"Fit failed for {participant_id}/{agent_id}: {e}")
        return {}

    row = {
        "participant_id": participant_id,
        "agent":          agent_id,
        "model":          model_cls.__name__,
        "log_likelihood": result.log_likelihood,
        "aic":            result.aic,
        "n_trials":       len(block_df),
        "converged":      result.converged,
    }

    # Add model-specific parameters
    for param in ["alpha", "beta", "gamma", "lam", "kappa"]:
        if hasattr(result, param):
            row[f"param_{param}"] = getattr(result, param)

    return row


def fit_participant(
    df:             pd.DataFrame,
    participant_id: str,
    model_names:    List[str] = ("chase",),
    n_restarts:     int = 10,
    seed:           Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit all requested models to all agent conditions for one participant.
    Returns a DataFrame with one row per agent × model.
    """
    rows = []
    for agent_id in AGENTS_ORDERED:
        for model_name in model_names:
            model_cls = MODEL_REGISTRY[model_name]
            row = fit_participant_condition(
                model_cls, df, participant_id, agent_id,
                n_restarts=n_restarts, seed=seed
            )
            if row:
                rows.append(row)

    return pd.DataFrame(rows)


# ── Belief update extraction ──────────────────────────────────────────────────

def extract_trial_level_estimates(
    df:             pd.DataFrame,
    participant_id: str,
    agent_id:       str,
    fitted_params:  dict,
) -> pd.DataFrame:
    """
    Re-run CHASE forward pass with fitted parameters to extract
    trial-by-trial estimates: belief_update (BU), ape, choice_value,
    inferred_level.

    Returns DataFrame with one row per trial.
    """
    block_df = df[df["agent"] == agent_id].sort_values("trial").copy()
    choices  = block_df["participant_choice"].values.astype(int)
    opp      = block_df["agent_choice"].values.astype(int)

    model  = CHASEModel()
    params = {k.replace("param_", ""): v for k, v in fitted_params.items()
              if k.startswith("param_")}
    result = model.simulate(params, choices, opp)

    block_df["belief_update"]   = result.belief_updates
    block_df["ape"]             = result.ape
    block_df["choice_value"]    = result.choice_values
    block_df["inferred_level"]  = result.inferred_level

    # Belief distribution columns (k=0 … kappa-1)
    for k in range(result.beliefs.shape[1]):
        block_df[f"belief_k{k}"] = result.beliefs[:, k]

    return block_df


# ── Group fitting pipeline ────────────────────────────────────────────────────

def fit_group(
    data_dir:    str = "data/",
    output_dir:  str = "results/models/",
    model_names: List[str] = ("chase", "rl", "fp", "ewa", "ewa_s", "tomk"),
    n_restarts:  int = 10,
    seed:        Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit all models to all participants and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    participant_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".csv")]
    n = len(participant_files)

    for i, fname in enumerate(participant_files):
        participant_id = fname.split("_session")[0]
        print(f"[{i+1}/{n}] Fitting {participant_id}...")

        try:
            df = load_participant_data(participant_id, data_dir=data_dir)
        except Exception as e:
            warnings.warn(f"Could not load {participant_id}: {e}")
            continue

        result_df = fit_participant(
            df, participant_id,
            model_names=model_names,
            n_restarts=n_restarts,
            seed=seed,
        )
        all_results.append(result_df)

        # Save trial-level CHASE estimates for this participant
        chase_rows = result_df[result_df["model"] == "CHASEModel"]
        trial_dfs  = []
        for _, row in chase_rows.iterrows():
            try:
                tdf = extract_trial_level_estimates(
                    df, participant_id, row["agent"], row.to_dict()
                )
                trial_dfs.append(tdf)
            except Exception as e:
                warnings.warn(f"Trial-level extraction failed: {participant_id}/{row['agent']}: {e}")

        if trial_dfs:
            trial_level = pd.concat(trial_dfs)
            out_path    = os.path.join(output_dir, f"{participant_id}_trial_level.csv")
            trial_level.to_csv(out_path, index=False)

    if not all_results:
        return pd.DataFrame()

    group_df = pd.concat(all_results, ignore_index=True)
    group_df.to_csv(os.path.join(output_dir, "model_fits_all.csv"), index=False)
    print(f"\nSaved group model fits to {output_dir}model_fits_all.csv")
    return group_df


# ── AIC-based model comparison ────────────────────────────────────────────────

def aic_model_comparison(fits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean AIC and AIC weights across participants for each model.
    Lower AIC = better fit.

    Returns a DataFrame with one row per model, sorted by mean AIC.
    """
    summary = (
        fits_df.groupby("model")["aic"]
        .agg(mean_aic="mean", se_aic="sem", n="count")
        .reset_index()
        .sort_values("mean_aic")
    )

    # Delta AIC relative to best model
    summary["delta_aic"] = summary["mean_aic"] - summary["mean_aic"].min()

    # AIC weights: relative likelihood exp(-0.5 * delta)
    weights = np.exp(-0.5 * summary["delta_aic"].values)
    summary["aic_weight"] = weights / weights.sum()

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model fitting pipeline")
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--output_dir", default="results/models/")
    parser.add_argument("--model",      default="chase",
                        help="Comma-separated list of models to fit (chase,rl,fp,ewa,ewa_s,tomk)")
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.model.split(",")]

    fits = fit_group(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        model_names = model_names,
        n_restarts  = args.n_restarts,
        seed        = args.seed,
    )

    if len(model_names) > 1 and not fits.empty:
        print("\nModel comparison (AIC):")
        comparison = aic_model_comparison(fits)
        print(comparison.to_string(index=False))
        comparison.to_csv(
            os.path.join(args.output_dir, "model_comparison_aic.csv"), index=False
        )
