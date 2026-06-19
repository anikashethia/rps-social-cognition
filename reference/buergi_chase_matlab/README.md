# Buergi et al. CHASE model — reference copy

MATLAB source for the CHASE model and its fitting/simulation pipeline, copied verbatim from:

**https://github.com/ruffgroup/neural_signature_of_mentalization**

(code release for Buergi, Aydogan, Konovalov & Ruff, "A neural signature of adaptive mentalization," *Nature Neuroscience*, 2026)

## Why this is here

`models/chase.py` is a Python reimplementation of CHASE for this project. This folder is kept as a reference to validate that reimplementation against the original — e.g. confirming the two-attraction-tracker recursion (`f_mat_own` / `f_mat_other`), the parameter bounds in `BAKR_2024_CHASE_config.m`, and the bot-level rotation logic in `mn_RPS_config.m`.

## What's included

Only the CHASE-model-relevant files — not their fMRI/neuroimaging analysis code, data, or results (those live in `data/`, `masks/`, `pattern/`, `results/` in the original repo and aren't needed here):

- `BAKR_2024_run_model_fitting.m` — top-level script showing how model fitting, model recovery, and parameter recovery are invoked
- `source/BAKR_2024_CHASE_model.m` — the core CHASE model (fit + simulate)
- `source/BAKR_2024_CHASE_config.m` — parameter bounds/grid-search config
- `source/BAKR_2024_CHASE_LR_init.m` / `BAKR_2024_CHASE_LR_update.m` — the two-attraction-tracker (`f_mat_own`/`f_mat_other`) learning rule
- `source/BAKR_2024_simulate_data.m` — synthetic data generation for recovery
- `source/comp_paymatrix.m` — payoff matrix construction
- `source/MERLIN_toolbox/` — the generic model-fitting/simulation toolbox (`mn_fit`, `mn_sim`, etc.) needed to actually run the above, plus `mn_RPS_config.m`/`mn_RPS_task.m` (their task/opponent definition, including the rotating bot-level logic)
- `README_original.md` — the original repo's README, for context

## License note

The source repo has no LICENSE file, so it's "all rights reserved" by default rather than under an explicit open-source license. This copy is kept here for reference and cross-validation against `models/chase.py` only — not redistributed as part of this project's own claimed code.
