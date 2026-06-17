"""Load trial data from the SQLite backend database."""

import sqlite3
import pandas as pd
from typing import Optional

_QUERY_BASE = """
    SELECT
        s.participant_id,
        s.config_index,
        s.mode,
        s.id          AS session_id,
        t.block,
        t.agent,
        t.condition,
        t.trial_in_block,
        t.trial_global,
        t.participant_choice,
        t.agent_choice,
        t.outcome,
        t.points_delta,
        t.points_cumulative,
        t.rt_ms,
        t.onset_ms,
        t.iti_duration_ms,
        t.block_onset_ms
    FROM trials t
    JOIN sessions s ON t.session_id = s.id
"""


def load_participant_data(
    participant_id: str,
    db_path:        str = "backend/rps.db",
    session_id:     Optional[int] = None,
) -> pd.DataFrame:
    """
    Load all trials for one participant from the SQLite database.

    Returns a DataFrame sorted by trial_global with columns from both the
    sessions and trials tables, including condition (friendly/neutral/control).
    """
    conn   = sqlite3.connect(db_path)
    where  = "WHERE s.participant_id = ?"
    params = [participant_id]

    if session_id is not None:
        where  += " AND s.id = ?"
        params.append(session_id)

    query = f"{_QUERY_BASE} {where} ORDER BY t.trial_global"
    df    = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_all_data(db_path: str = "backend/rps.db") -> pd.DataFrame:
    """Load all trials from the database, joined with session metadata."""
    conn = sqlite3.connect(db_path)
    query = f"{_QUERY_BASE} ORDER BY s.participant_id, t.trial_global"
    df   = pd.read_sql_query(query, conn)
    conn.close()
    return df


def list_participants(db_path: str = "backend/rps.db") -> list:
    """Return sorted list of all participant IDs in the database."""
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query(
        "SELECT DISTINCT participant_id FROM sessions ORDER BY participant_id", conn
    )
    conn.close()
    return df["participant_id"].tolist()
