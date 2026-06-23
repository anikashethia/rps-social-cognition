# models/chase.py
"""
CHASE — Cognitive Hierarchy Assessment with Sophistication Estimation
(Buergi, Aydogan, Konovalov & Ruff, Nature Neuroscience, 2026)

Implements the full CHASE model for fitting to participant RPS behavior.
Captures *adaptive mentalization*: trial-by-trial Bayesian updating of
beliefs about an opponent's strategic sophistication level (k).

Key model outputs:
    - Fitted parameters: alpha, beta, gamma, lambda, kappa
    - Trial-by-trial belief distributions B(k)_t
    - Belief updates (BU): KL divergence between B(k)_t and B(k)_{t-1}
    - Action prediction errors (APE)
    - Choice values (CV)

Usage:
    from models.chase import CHASEModel
    model = CHASEModel()
    result = model.fit(choices, opponent_choices)
    bu_timeseries = result.belief_updates
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings


# ── Parameter bounds ──────────────────────────────────────────────────────────

PARAM_BOUNDS = {
    "alpha":  (1e-4, 1.0 - 1e-4),   # Learning rate
    "beta":   (0.01, 20.0),          # Softmax inverse temperature
    "gamma":  (0.01, 20.0),          # Sensitivity to opponent-level evidence
    "lam":    (0.1,  5.0),           # Loss sensitivity (lambda; renamed to avoid builtin)
    "kappa":  (1,    4),             # Max sophistication level (integer, fitted as continuous then rounded)
}

PARAM_INIT = {
    "alpha": 0.3,
    "beta":  3.0,
    "gamma": 2.0,
    "lam":   1.0,
    "kappa": 3,    # Fitted over {1,2,3,4}
}

# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class CHASEResult:
    """Container for CHASE fit results and trial-by-trial estimates."""
    # Fitted parameters
    alpha:  float = np.nan
    beta:   float = np.nan
    gamma:  float = np.nan
    lam:    float = np.nan
    kappa:  int   = 3

    # Fit quality
    log_likelihood: float = np.nan
    aic:            float = np.nan
    converged:      bool  = False

    # Trial-by-trial outputs (length = n_trials)
    beliefs:       np.ndarray = field(default_factory=lambda: np.array([]))  # shape (T, kappa)
    belief_updates: np.ndarray = field(default_factory=lambda: np.array([]))  # KL divergence, shape (T,)
    action_probs:   np.ndarray = field(default_factory=lambda: np.array([]))  # P(a|participant), shape (T, 3)
    choice_values:  np.ndarray = field(default_factory=lambda: np.array([]))  # CV, shape (T,)
    ape:            np.ndarray = field(default_factory=lambda: np.array([]))  # APE, shape (T,)
    inferred_level: np.ndarray = field(default_factory=lambda: np.array([]))  # argmax belief, shape (T,)


# ── Core model ────────────────────────────────────────────────────────────────

class CHASEModel:
    """
    Full CHASE model for fitting to participant behavior.

    The game is RPS with 3 actions (Rock=0, Paper=1, Scissors=2 internally;
    displayed as 1,2,3 externally — converted on input).

    Payoff matrix Π: Π[i,j] = +1 if action i beats j, -1 if loses, 0 if draw.
    Rock(0) beats Scissors(2): Π[0,2]=+1; Paper(1) beats Rock(0): Π[1,0]=+1; etc.
    """

    N_ACTIONS = 3
    N_PARAMS  = 5  # alpha, beta, gamma, lam, kappa

    # Internal 0-indexed payoff matrix
    PAYOFF = np.array([
        [ 0, -1,  1],  # Rock
        [ 1,  0, -1],  # Paper
        [-1,  1,  0],  # Scissors
    ], dtype=float)

    def __init__(self, max_kappa: int = 4):
        self.max_kappa = max_kappa

    # ── Core computations ─────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
        z = temp * x
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def _recursive_probs(
        self, own_attractions: np.ndarray, opp_attractions: np.ndarray,
        beta: float, level: int, lam: float,
    ) -> np.ndarray:
        """
        P(a | opponent is at reasoning level `level`), built via recursive
        best-response. Even levels chain from the opponent's own attractions
        (their non-strategic habit); odd levels chain from the participant's
        own attractions (a level-1 opponent assumes the participant is
        non-strategic and predicts them via the participant's own habit).
        Matches Buergi et al.'s two-track recursion (their f_mat_other /
        f_mat_own) — RPS's symmetric payoff matrix makes pi_own == pi_other,
        so levels >= 2 are pure alternating recursion from these two seeds.
        """
        payoff = self.PAYOFF.copy()
        payoff[payoff == -1] *= lam  # Loss sensitivity

        seed  = opp_attractions if level % 2 == 0 else own_attractions
        probs = self._softmax(seed, beta)
        for _ in range(level):
            ev    = payoff @ probs
            probs = self._softmax(ev, beta)
        return probs

    def _belief_update(
        self, prior: np.ndarray, own_attractions: np.ndarray, opp_attractions: np.ndarray,
        opp_action: int, beta: float, gamma: float, lam: float, kappa: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Bayesian update of beliefs about opponent's level.

        Returns:
            posterior: updated belief distribution (length kappa)
            bu: KL divergence between posterior and prior
        """
        # Likelihood L(k | opp_action) for k in 0..kappa-1
        likelihood = np.array([
            self._recursive_probs(own_attractions, opp_attractions, beta, k, lam)[opp_action]
            for k in range(kappa)
        ])

        # Distort likelihood with gamma (participant's sensitivity to level evidence)
        distorted = self._softmax(likelihood, gamma)

        # Bayes update
        unnorm    = distorted * prior
        posterior = unnorm / unnorm.sum()

        # KL divergence: BU = sum_k posterior[k] * log(posterior[k] / prior[k])
        # Add small epsilon for numerical stability
        eps = 1e-12
        bu  = float(np.sum(posterior * np.log((posterior + eps) / (prior + eps))))

        return posterior, bu

    # ── Trial-by-trial simulation ─────────────────────────────────────────────

    def simulate(
        self,
        params:           dict,
        choices:          np.ndarray,       # Participant choices, 0-indexed, shape (T,)
        opponent_choices: np.ndarray,       # Opponent choices, 0-indexed, shape (T,)
        trial_in_block:   Optional[np.ndarray] = None,  # 1-indexed trial-within-block, shape (T,)
    ) -> CHASEResult:
        """
        Run the CHASE model forward given parameters and observed choices.
        Returns a CHASEResult with trial-by-trial estimates.

        If `trial_in_block` is given, attractions and beliefs are reset to
        uniform whenever trial_in_block[t] == 1 (t > 0) -- i.e. at the start
        of each new block within a multi-block sequence, matching Buergi et
        al.'s "reset all relevant prior belief variables... to a uniform
        distribution at the beginning of each block" (fit jointly as one set
        of parameters per participant, across blocks).
        """
        alpha = params["alpha"]
        beta  = params["beta"]
        gamma = params["gamma"]
        lam   = params["lam"]
        kappa = int(round(params["kappa"]))
        kappa = max(1, min(kappa, self.max_kappa))

        T = len(choices)

        def uniform_state():
            return (
                np.ones(self.N_ACTIONS) / self.N_ACTIONS,
                np.ones(self.N_ACTIONS) / self.N_ACTIONS,
                np.ones(kappa) / kappa,
            )

        own_attractions, opp_attractions, beliefs = uniform_state()

        all_beliefs  = np.zeros((T, kappa))
        all_bu       = np.zeros(T)
        all_probs    = np.zeros((T, self.N_ACTIONS))
        all_cv       = np.zeros(T)
        all_ape      = np.zeros(T)

        for t in range(T):
            if trial_in_block is not None and t > 0 and trial_in_block[t] == 1:
                own_attractions, opp_attractions, beliefs = uniform_state()

            p_choice   = choices[t]
            opp_choice = opponent_choices[t]

            # ── Choice phase ─────────────────────────────────────────────────
            # Integrated prediction over opponent levels
            integrated_opp_probs = np.zeros(self.N_ACTIONS)
            for k in range(kappa):
                opp_probs_k          = self._recursive_probs(own_attractions, opp_attractions, beta, k, lam)
                integrated_opp_probs += beliefs[k] * opp_probs_k

            # Participant best-responds to integrated prediction
            payoff_scaled = self.PAYOFF.copy()
            payoff_scaled[payoff_scaled == -1] *= lam
            ev_participant = payoff_scaled @ integrated_opp_probs

            # Choice probabilities P(a | kappa)
            action_probs = self._softmax(ev_participant, beta)
            all_probs[t] = action_probs

            # Choice value: expected payoff of chosen action
            all_cv[t] = float(ev_participant[p_choice])

            # ── Feedback phase ────────────────────────────────────────────────
            # Action prediction error: deviation of opponent action from prediction
            all_ape[t] = float(1.0 - integrated_opp_probs[opp_choice])

            # Belief update (A3): Bayes update on opponent level
            beliefs, bu = self._belief_update(
                beliefs, own_attractions, opp_attractions, opp_choice, beta, gamma, lam, kappa
            )
            all_beliefs[t] = beliefs
            all_bu[t]      = bu

            # Update attractions (A1): delta rule, tracked separately for each player
            own_indicator = np.zeros(self.N_ACTIONS)
            own_indicator[p_choice] = 1.0
            own_attractions += alpha * (own_indicator - own_attractions)

            opp_indicator = np.zeros(self.N_ACTIONS)
            opp_indicator[opp_choice] = 1.0
            opp_attractions += alpha * (opp_indicator - opp_attractions)

        result = CHASEResult(
            alpha=alpha, beta=beta, gamma=gamma, lam=lam, kappa=kappa,
            beliefs       = all_beliefs,
            belief_updates = all_bu,
            action_probs   = all_probs,
            choice_values  = all_cv,
            ape            = all_ape,
            inferred_level = np.argmax(all_beliefs, axis=1),
        )
        return result

    # ── Log-likelihood ────────────────────────────────────────────────────────

    def log_likelihood(
        self,
        params:           dict,
        choices:          np.ndarray,
        opponent_choices: np.ndarray,
        trial_in_block:   Optional[np.ndarray] = None,
    ) -> float:
        """Compute negative log-likelihood of participant choices under CHASE."""
        result = self.simulate(params, choices, opponent_choices, trial_in_block)
        eps    = 1e-12
        ll     = np.sum(np.log(result.action_probs[np.arange(len(choices)), choices] + eps))
        return float(ll)

    # ── MLE fitting ──────────────────────────────────────────────────────────

    def fit(
        self,
        choices:          np.ndarray,
        opponent_choices: np.ndarray,
        n_restarts:       int = 10,
        seed:             Optional[int] = None,
        trial_in_block:   Optional[np.ndarray] = None,
    ) -> CHASEResult:
        """
        Fit CHASE parameters via maximum likelihood estimation.
        Uses random restarts to avoid local minima.

        Parameters
        ----------
        choices : np.ndarray, shape (T,)
            Participant's choices, 0-indexed (0=Rock, 1=Paper, 2=Scissors).
        opponent_choices : np.ndarray, shape (T,)
            Opponent's choices, 0-indexed.
        n_restarts : int
            Number of random restarts.
        trial_in_block : np.ndarray, shape (T,), optional
            1-indexed trial-within-block counter. If given, attractions and
            beliefs reset to uniform at the start of each new block (matches
            Buergi et al.'s joint per-participant, per-block-reset fitting).

        Returns
        -------
        CHASEResult with fitted parameters and trial-by-trial estimates.
        """
        rng = np.random.default_rng(seed)

        # Convert external 1-indexed to 0-indexed if needed
        if choices.min() == 1:
            choices          = choices - 1
            opponent_choices = opponent_choices - 1

        best_ll     = -np.inf
        best_params = None
        best_res    = None

        for restart in range(n_restarts):
            # Random initial parameters within bounds
            if restart == 0:
                p0 = {k: v for k, v in PARAM_INIT.items()}
            else:
                p0 = {
                    "alpha": rng.uniform(*PARAM_BOUNDS["alpha"]),
                    "beta":  rng.uniform(*PARAM_BOUNDS["beta"]),
                    "gamma": rng.uniform(*PARAM_BOUNDS["gamma"]),
                    "lam":   rng.uniform(*PARAM_BOUNDS["lam"]),
                    "kappa": rng.integers(1, self.max_kappa + 1),
                }

            # Optimise over continuous params (alpha, beta, gamma, lam)
            # kappa is treated as a discrete grid search
            for kappa_try in range(1, self.max_kappa + 1):
                p0["kappa"] = kappa_try

                def neg_ll(x):
                    params = {
                        "alpha": x[0], "beta": x[1],
                        "gamma": x[2], "lam":  x[3],
                        "kappa": kappa_try,
                    }
                    return -self.log_likelihood(params, choices, opponent_choices, trial_in_block)

                x0     = [p0["alpha"], p0["beta"], p0["gamma"], p0["lam"]]
                bounds = [
                    PARAM_BOUNDS["alpha"], PARAM_BOUNDS["beta"],
                    PARAM_BOUNDS["gamma"], PARAM_BOUNDS["lam"],
                ]

                try:
                    opt = minimize(
                        neg_ll, x0,
                        method  = "L-BFGS-B",
                        bounds  = bounds,
                        options = {"maxiter": 500, "ftol": 1e-9},
                    )
                    if not opt.success:
                        continue

                    fitted = {
                        "alpha": opt.x[0], "beta": opt.x[1],
                        "gamma": opt.x[2], "lam":  opt.x[3],
                        "kappa": kappa_try,
                    }
                    ll = self.log_likelihood(fitted, choices, opponent_choices, trial_in_block)

                    if ll > best_ll:
                        best_ll     = ll
                        best_params = fitted
                        best_res    = opt

                except Exception as e:
                    warnings.warn(f"Optimisation failed (restart {restart}, kappa={kappa_try}): {e}")
                    continue

        if best_params is None:
            warnings.warn("All optimisation restarts failed.")
            return CHASEResult()

        # Run forward pass with best parameters to get trial-by-trial outputs
        result = self.simulate(best_params, choices, opponent_choices, trial_in_block)
        result.log_likelihood = best_ll
        result.aic            = -2 * best_ll + 2 * (self.N_PARAMS - 1)  # kappa is discrete
        result.converged      = True
        return result
