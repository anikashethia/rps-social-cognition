# utils/counterbalancing.py
"""
Block order counterbalancing for the RPS Social Cognition Task.

Uses a Latin square design across participants to ensure each agent
appears equally often in each block position across the sample.
The RNG control is always included. The four social agents (A-D)
vary in position; the RNG is inserted at a counterbalanced position.

Counterbalancing is deterministic given (participant_id, session),
so the same order is reproduced if the task is re-run.
"""

from itertools import permutations
import hashlib
from typing import List

ALL_AGENTS = ["friendly", "neutral", "rng"]

# All 6 orderings of the 3 conditions
BLOCK_ORDERS = list(permutations(ALL_AGENTS))


def _participant_hash(participant_id: str, session: int) -> int:
    """Deterministic integer hash for a participant × session."""
    key    = f"{participant_id}_{session}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest, 16)


def get_block_order(participant_id: str, session: int = 1) -> List[str]:
    """
    Return a counterbalanced block order for this participant and session.
    Cycles through all 6 permutations of [friendly, neutral, rng].

    Parameters
    ----------
    participant_id : str
    session : int

    Returns
    -------
    list of 3 agent identifiers in presentation order.
    """
    h = _participant_hash(participant_id, session)
    return list(BLOCK_ORDERS[h % len(BLOCK_ORDERS)])


def describe_counterbalancing(n_participants: int = 12) -> None:
    """Print block orders for a range of participants to verify balance."""
    print(f"{'Participant':<15} {'Block order'}")
    print("-" * 50)
    for i in range(1, n_participants + 1):
        pid   = f"SUB{i:03d}"
        order = get_block_order(pid, session=1)
        print(f"{pid:<15} {' > '.join(order)}")


if __name__ == "__main__":
    describe_counterbalancing(20)
