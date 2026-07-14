#!/usr/bin/env python3
"""
Export jsPsych trial data from SQLite to CSV for Buergi et al. MATLAB fitting.

Produces: analysis/buergi_export.csv

Column mapping → Buergi's behavioral_data.mat format:
  subjID        numeric participant index (1-based)
  choice_own    participant choice  (1=Rock, 2=Paper, 3=Scissors; NaN if missing)
  choice_other  bot choice          (1=Rock, 2=Paper, 3=Scissors)
  score_own     outcome from participant's view (+1 win, -1 lose, 0 tie; NaN if missing)
  bot_level     CHASE level for that trial (0, 1, or 2)
  missing       1 if no response (timeout), 0 otherwise
  trial         trial number within block (1–40)
  block         block number (1–6)
  condition     "friendly" or "neutral"
  dataset       study label (hardcoded)

Run from repo root:
  python3 analysis/export_for_matlab.py [--db path/to/rps.db] [--out path/to/output.csv]
"""

import argparse
import os
import sys

import pandas as pd
import sqlalchemy as sa

DEFAULT_DB  = os.path.join(os.path.dirname(__file__), '..', 'backend', 'rps.db')
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'buergi_export.csv')
DATASET     = 'yale_rps'


def export(db_path: str, out_path: str) -> None:
    engine = sa.create_engine(f'sqlite:///{os.path.abspath(db_path)}')

    with engine.connect() as conn:
        # Pull all trials joined with their session's participant_id
        trials = pd.read_sql("""
            SELECT
                s.participant_id,
                t.block,
                t.trial_in_block,
                t.participant_choice,
                t.agent_choice,
                t.outcome,
                t.rt_ms,
                t.level,
                t.condition
            FROM trials t
            JOIN sessions s ON t.session_id = s.id
            ORDER BY s.participant_id, t.block, t.trial_in_block
        """, conn)

    if trials.empty:
        print("No trials found in database — have you run any sessions yet?")
        sys.exit(1)

    # Build a stable numeric subject index (alphabetical order of participant_id)
    participants = sorted(trials['participant_id'].unique())
    pid_to_idx   = {pid: i + 1 for i, pid in enumerate(participants)}
    print(f"Found {len(participants)} participant(s): {participants}")

    # Map outcome string → numeric score (from participant's perspective)
    OUTCOME_MAP = {'win': 1, 'lose': -1, 'tie': 0}

    rows = []
    for _, t in trials.iterrows():
        missing    = 1 if pd.isna(t['rt_ms']) or t['participant_choice'] is None else 0
        choice_own   = float(t['participant_choice']) if not missing else float('nan')
        score_own    = float(OUTCOME_MAP[t['outcome']]) if not missing else float('nan')

        rows.append({
            'subjID':       pid_to_idx[t['participant_id']],
            'participant_id': t['participant_id'],   # keep for reference, not used by MATLAB
            'choice_own':   choice_own,
            'choice_other': float(t['agent_choice']),
            'score_own':    score_own,
            'bot_level':    int(t['level']),
            'missing':      missing,
            'trial':        int(t['trial_in_block']),
            'block':        int(t['block']),
            'condition':    t['condition'],
            'dataset':      DATASET,
        })

    df = pd.DataFrame(rows)

    # Sanity checks
    n_trials_per_subj = df.groupby('subjID').size()
    expected = 240  # 6 blocks × 40 trials
    for sid, n in n_trials_per_subj.items():
        if n != expected:
            print(f"  WARNING: subjID {sid} has {n} trials (expected {expected})")

    df.to_csv(out_path, index=False)
    print(f"Exported {len(df)} trials ({len(participants)} participant(s)) → {out_path}")

    # Print per-participant summary
    print("\nPer-participant summary:")
    for pid in participants:
        sub = df[df['participant_id'] == pid]
        n_missing = sub['missing'].sum()
        levels    = sorted(sub['bot_level'].unique().tolist())
        print(f"  {pid} (subjID={pid_to_idx[pid]}): {len(sub)} trials, "
              f"{n_missing} missing, levels={levels}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',  default=DEFAULT_DB,  help='Path to rps.db')
    parser.add_argument('--out', default=DEFAULT_OUT, help='Output CSV path')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Run the backend first and collect some data, or pass --db /path/to/rps.db")
        sys.exit(1)

    export(args.db, args.out)
