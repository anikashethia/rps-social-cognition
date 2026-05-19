# task/agents.py
"""
AI agent strategy implementations for the RPS Social Cognition Task.

All four social agents play at the same strategic level (k ≈ 1–2 in CHASE).
The RNG control plays uniformly at random. Agents are based on a simplified
CHASE model (Buergi et al., 2026) with fixed sophistication — no adaptive
belief updating on the agent side, but calibrated noise to ensure human-like
behavior that passes a Turing test.
"""

import numpy as np
from typing import List, Optional
from task.config import AGENT_STRATEGY_LEVELS, AGENT_NOISE, ACTIONS


class BaseAgent:
    """Abstract base for all agents."""

    def __init__(self, agent_id: str, seed: Optional[int] = None):
        self.agent_id    = agent_id
        self.rng         = np.random.default_rng(seed)
        self.action_history: List[int]  = []
        self.opp_history: List[int]     = []

    def choose(self, opponent_history: List[int]) -> int:
        raise NotImplementedError

    def update(self, own_action: int, opp_action: int) -> None:
        self.action_history.append(own_action)
        self.opp_history.append(opp_action)

    def reset(self) -> None:
        self.action_history = []
        self.opp_history    = []


class RandomAgent(BaseAgent):
    """
    Level-0 / RNG control.
    Chooses uniformly at random — no strategy, no mind to model.
    Participants are told there is no opponent; a generic icon is shown.
    """

    def choose(self, opponent_history: List[int]) -> int:
        return int(self.rng.choice(ACTIONS))


class CHASEAgent(BaseAgent):
    """
    Fixed-level CHASE agent (Buergi et al., 2026).

    Implements A1 (action-frequency tracking) + A2 (recursive reasoning)
    at a fixed sophistication level k, with calibrated noise gamma so the
    agent is indistinguishable from human opponents in a Turing test.

    Parameters
    ----------
    level : int
        Fixed strategic level k (1 or 2).
    alpha : float
        Learning rate for action-frequency tracker (attractions).
    beta : float
        Softmax noise for action selection.
    noise : float
        Additional epsilon-greedy noise to humanise behavior.
    """

    def __init__(
        self,
        agent_id:  str,
        level:     int   = 1,
        alpha:     float = 0.3,
        beta:      float = 3.0,
        noise:     float = 0.15,
        seed:      Optional[int] = None,
    ):
        super().__init__(agent_id, seed)
        self.level  = level
        self.alpha  = alpha
        self.beta   = beta
        self.noise  = noise  # Epsilon-greedy noise probability

        # Action-frequency attractions (A1), initialised uniformly
        self.attractions = np.ones(len(ACTIONS)) / len(ACTIONS)

        # Payoff matrix: payoff[i][j] = 1 if action i beats action j, -1 if loses, 0 if draw
        # Actions indexed 0,1,2 = Rock, Paper, Scissors
        self._payoff = np.array([
            [ 0, -1,  1],   # Rock:     draws Rock, loses to Paper, beats Scissors
            [ 1,  0, -1],   # Paper:    beats Rock, draws Paper, loses to Scissors
            [-1,  1,  0],   # Scissors: loses to Rock, beats Paper, draws Scissors
        ], dtype=float)

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        v = self.beta * values
        v -= v.max()
        e = np.exp(v)
        return e / e.sum()

    def _level0_probs(self) -> np.ndarray:
        """P(a | k=0): softmax over current attractions."""
        return self._softmax(self.attractions)

    def _best_respond(self, opp_probs: np.ndarray) -> np.ndarray:
        """
        Given a distribution over opponent actions, compute the
        best-response distribution (one step of recursive reasoning).
        Returns softmax over expected payoffs.
        """
        ev = self._payoff @ opp_probs
        return self._softmax(ev)

    def _compute_action_probs(self) -> np.ndarray:
        """Compute P(a | k=self.level) by applying k steps of recursion."""
        probs = self._level0_probs()
        for _ in range(self.level):
            probs = self._best_respond(probs)
        return probs

    def choose(self, opponent_history: List[int]) -> int:
        """
        Select an action given the opponent's history.
        Uses the agent's own history (stored internally) to update attractions.
        """
        probs = self._compute_action_probs()

        # Epsilon-greedy noise: with probability `noise`, choose uniformly
        if self.rng.random() < self.noise:
            action_idx = int(self.rng.choice(len(ACTIONS)))
        else:
            action_idx = int(self.rng.choice(len(ACTIONS), p=probs))

        return ACTIONS[action_idx]

    def update(self, own_action: int, opp_action: int) -> None:
        """Update action-frequency attractions after each trial."""
        super().update(own_action, opp_action)

        # Delta-rule update on own past actions (A1)
        action_idx = ACTIONS.index(own_action)
        indicator  = np.zeros(len(ACTIONS))
        indicator[action_idx] = 1.0
        self.attractions += self.alpha * (indicator - self.attractions)

    def reset(self) -> None:
        super().reset()
        self.attractions = np.ones(len(ACTIONS)) / len(ACTIONS)


def build_agent(agent_id: str, seed: Optional[int] = None) -> BaseAgent:
    """
    Factory: return the appropriate agent object for a given agent_id.
    All social agents (A–D) use CHASE at the same strategic level so that
    mentalizing differences are driven by prior connection, not difficulty.
    """
    if agent_id == "rng":
        return RandomAgent(agent_id, seed=seed)

    level = AGENT_STRATEGY_LEVELS[agent_id]
    noise = AGENT_NOISE[agent_id]
    return CHASEAgent(agent_id, level=level, noise=noise, seed=seed)
