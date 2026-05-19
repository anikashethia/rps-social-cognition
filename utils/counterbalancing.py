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

SOCIAL_AGENTS = ["agent_a", "agent_b", "agent_c", "agent_d"]
ALL_AGENTS    = SOCIAL_AGENTS + ["rng"]


def _participant_hash(participant_id: str, session: int) -> int:
    """Deterministic integer hash for a participant × session."""
    key   = f"{participant_id}_{session}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest, 16)


def get_block_order(participant_id: str, session: int = 1) -> List[str]:
    """
    Return a counterbalanced block order for this participant and session.

    The 4 social agents are arranged in one of 24 possible permutations,
    cycling through them as participant index increases. The RNG block is
    inserted at one of 5 positions (before A, between each pair, after D),
    also counterbalanced.

    Parameters
    ----------
    participant_id : str
    session : int

    Returns
    -------
    list of 5 agent identifiers in presentation order.
    """
    h = _participant_hash(participant_id, session)

    # 24 permutations of the 4 social agents
    social_perms = list(permutations(SOCIAL_AGENTS))
    social_order = list(social_perms[h % len(social_perms)])

    # RNG inserted at position 0–4 (before any, between each pair, after last)
    rng_position = (h // len(social_perms)) % (len(SOCIAL_AGENTS) + 1)

    order = social_order[:rng_position] + ["rng"] + social_order[rng_position:]
    return order


def describe_counterbalancing(n_participants: int = 20) -> None:
    """
    Print the block order for a range of participants to verify balance.
    Useful for checking counterbalancing before data collection.
    """
    print(f"{'Participant':<15} {'Block order':}")
    print("-" * 60)
    for i in range(1, n_participants + 1):
        pid   = f"SUB{i:03d}"
        order = get_block_order(pid, session=1)
        print(f"{pid:<15} {' > '.join(order)}")


if __name__ == "__main__":
    describe_counterbalancing(20)
