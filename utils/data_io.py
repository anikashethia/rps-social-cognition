# utils/data_io.py
"""
Data loading and saving utilities for the RPS Social Cognition Task.
"""

import os
import csv
import pandas as pd
from typing import Optional
from task.config import DATA_DIR, OUTPUT_COLUMNS


def get_output_path(participant_id: str, session: int, data_dir: str = DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{participant_id}_session{session}_rps.csv")


def save_trial(trial_data: dict, participant_id: str, session: int,
               data_dir: str = DATA_DIR) -> None:
    """Append one trial to the participant's CSV (write header on first trial)."""
    path       = get_output_path(participant_id, session, data_dir)
    write_hdr  = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_hdr:
            writer.writeheader()
        writer.writerow(trial_data)


def load_participant_data(
    participant_id: str,
    session:        Optional[int] = None,
    data_dir:       str = DATA_DIR,
) -> pd.DataFrame:
    """
    Load trial data for one participant. If session is None, loads all sessions
    and concatenates them.
    """
    if session is not None:
        path = get_output_path(participant_id, session, data_dir)
        return pd.read_csv(path)

    # Load all sessions
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith(participant_id) and f.endswith("_rps.csv")
    ]
    if not files:
        raise FileNotFoundError(f"No data files found for {participant_id} in {data_dir}")

    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in sorted(files)]
    return pd.concat(dfs, ignore_index=True)


def load_all_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all participant data files."""
    dfs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith("_rps.csv"):
            dfs.append(pd.read_csv(os.path.join(data_dir, fname)))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
