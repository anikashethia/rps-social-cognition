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

import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings

from config import N_ACTIONS, PARAM_BOUNDS, PARAM_INIT

# ── Grid search values (BAKR_2024_CHASE_config.m, n_samples = 5) ─────────────
# Stage 1 of fitting: evaluate LL at every Cartesian combination (5^4 = 625
# points per kappa), then optimise from the top-N.

_N_GRID = 5
GRID_VALUES = {
    "beta":  np.linspace(0.1,  5.0,  _N_GRID),   # [0.10, 1.33, 2.55, 3.78, 5.00]
    "lam":   np.linspace(1e-4, 2.0,  _N_GRID),   # [0.00, 0.50, 1.00, 1.50, 2.00]
    "gamma": np.linspace(1e-4, 2.0,  _N_GRID),   # same
    "alpha": np.linspace(1e-3, 0.99, _N_GRID),   # [0.00, 0.25, 0.50, 0.74, 0.99]
}

# ── Parameter-space transforms for unconstrained optimisation ─────────────────
# Matches Buergi's estimation-space transforms in mn_fitModel.m:
#   alpha ∈ (0,1)   → logit  (inverse: sigmoid)
#   beta, gamma, lam ∈ (0,∞) → log (inverse: exp)
# Optimising in this space lets the solver move freely without hitting bounds.

def _to_est(alpha: float, beta: float, gamma: float, lam: float) -> np.ndarray:
    return np.array([
        np.log(alpha / (1.0 - alpha)),   # logit
        np.log(beta),
        np.log(gamma),
        np.log(lam),
    ])

def _from_est(x: np.ndarray) -> dict:
    return {
        "alpha": float(1.0 / (1.0 + np.exp(-x[0]))),          # sigmoid
        "beta":  float(np.clip(np.exp(x[1]), 0, PARAM_BOUNDS["beta"][1])),
        "gamma": float(np.clip(np.exp(x[2]), 0, PARAM_BOUNDS["gamma"][1])),
        "lam":   float(np.clip(np.exp(x[3]), 0, PARAM_BOUNDS["lam"][1])),
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
    beliefs:            np.ndarray = field(default_factory=lambda: np.array([]))  # shape (T, kappa)
    belief_updates:     np.ndarray = field(default_factory=lambda: np.array([]))  # KL divergence, shape (T,)
    action_probs:       np.ndarray = field(default_factory=lambda: np.array([]))  # P(a|participant), shape (T, 3)
    choice_values:      np.ndarray = field(default_factory=lambda: np.array([]))  # CV, shape (T,)
    ape:                np.ndarray = field(default_factory=lambda: np.array([]))  # APE, shape (T,)
    inferred_level:     np.ndarray = field(default_factory=lambda: np.array([]))  # argmax belief, shape (T,)
    # Full attraction histories — shape (T, 3), row t = attractions used for trial t's choice.
    # Matches Buergi's f_mat_own / f_mat_other (BAKR_2024_CHASE_LR_init.m RW-freq convention).
    own_attractions:    np.ndarray = field(default_factory=lambda: np.array([]))  # participant habit tracker
    opp_attractions:    np.ndarray = field(default_factory=lambda: np.array([]))  # opponent habit tracker


# ── Core model ────────────────────────────────────────────────────────────────

class CHASEModel:
    """
    Full CHASE model for fitting to participant behavior.

    The game is RPS with 3 actions (Rock=0, Paper=1, Scissors=2 internally;
    displayed as 1,2,3 externally — converted on input).

    Payoff matrix Π: Π[i,j] = +1 if action i beats j, -1 if loses, 0 if draw.
    Rock(0) beats Scissors(2): Π[0,2]=+1; Paper(1) beats Rock(0): Π[1,0]=+1; etc.
    """

    N_ACTIONS = N_ACTIONS  # from config
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
        kappa = max(0, min(kappa, self.max_kappa))

        T = len(choices)
        n_belief_levels = max(kappa, 1)  # kappa=0 has no belief levels; keep array shape sane

        def uniform_state():
            return (
                np.ones(self.N_ACTIONS) / self.N_ACTIONS,
                np.ones(self.N_ACTIONS) / self.N_ACTIONS,
                np.ones(n_belief_levels) / n_belief_levels,
            )

        own_attractions, opp_attractions, beliefs = uniform_state()

        all_beliefs       = np.full((T, n_belief_levels), np.nan)
        all_bu            = np.full(T, np.nan)
        all_probs         = np.zeros((T, self.N_ACTIONS))
        all_cv            = np.zeros(T)
        all_ape           = np.zeros(T)
        all_own_attr      = np.full((T, self.N_ACTIONS), np.nan)  # pre-update, used for choice
        all_opp_attr      = np.full((T, self.N_ACTIONS), np.nan)

        payoff_scaled = self.PAYOFF.copy()
        payoff_scaled[payoff_scaled == -1] *= lam

        for t in range(T):
            if trial_in_block is not None and t > 0 and trial_in_block[t] == 1:
                own_attractions, opp_attractions, beliefs = uniform_state()

            p_choice   = choices[t]
            opp_choice = opponent_choices[t]

            # Snapshot attractions before update — these are what's used for
            # the choice at trial t (matches Buergi's f_mat_bot(iTrial-1,:)).
            all_own_attr[t] = own_attractions
            all_opp_attr[t] = opp_attractions

            # ── Choice phase ─────────────────────────────────────────────────
            if kappa == 0:
                # Egocentric / non-strategic: ignore the opponent entirely.
                # No payoff matrix, no opponent prediction -- matches Buergi
                # et al.'s static branch k=0 case (subj_resp = softmax(f_mat_own)).
                # subj_pred is conventionally uniform here, used only for the
                # APE/SV bookkeeping below, never for the actual choice.
                integrated_opp_probs = np.ones(self.N_ACTIONS) / self.N_ACTIONS
                action_probs = self._softmax(own_attractions, beta)
                ev_participant = payoff_scaled @ integrated_opp_probs
            else:
                # Integrated prediction over opponent levels
                integrated_opp_probs = np.zeros(self.N_ACTIONS)
                for k in range(kappa):
                    opp_probs_k          = self._recursive_probs(own_attractions, opp_attractions, beta, k, lam)
                    integrated_opp_probs += beliefs[k] * opp_probs_k

                # Participant best-responds to integrated prediction
                ev_participant = payoff_scaled @ integrated_opp_probs
                action_probs   = self._softmax(ev_participant, beta)

            all_probs[t] = action_probs

            # Choice value: expected payoff of chosen action
            all_cv[t] = float(ev_participant[p_choice])

            # ── Feedback phase ────────────────────────────────────────────────
            # Action prediction error: deviation of opponent action from prediction
            all_ape[t] = float(1.0 - integrated_opp_probs[opp_choice])

            # Belief update (A3): Bayes update on opponent level.
            # Only defined when kappa >= 2 (matches Buergi et al.: belief
            # tracking doesn't exist at all for kappa < 2 -- output stays NaN,
            # not a degenerate placeholder).
            if kappa >= 2:
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

        inferred_level = (
            np.full(T, np.nan) if kappa < 2 else np.argmax(all_beliefs, axis=1).astype(float)
        )

        result = CHASEResult(
            alpha=alpha, beta=beta, gamma=gamma, lam=lam, kappa=kappa,
            beliefs          = all_beliefs,
            belief_updates   = all_bu,
            action_probs     = all_probs,
            choice_values    = all_cv,
            ape              = all_ape,
            inferred_level   = inferred_level,
            own_attractions  = all_own_attr,
            opp_attractions  = all_opp_attr,
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
        optim_n_it:       int = 4,
        seed:             Optional[int] = None,
        trial_in_block:   Optional[np.ndarray] = None,
    ) -> CHASEResult:
        """
        Two-stage fitting matching Buergi et al. BAKR_2024_CHASE_config.m /
        mn_fitModel.m:

        Stage 1 — Cartesian grid (LL evaluation only, no optimisation):
          5 values per continuous param → 5^4 = 625 combinations × each kappa.
          Cheap: just a single forward pass per combination.

        Stage 2 — Optimise from top-`optim_n_it` grid points per kappa:
          Parameters are transformed to unconstrained estimation space
          (logit for alpha, log for beta/gamma/lam) so the solver moves
          freely without boundary artefacts — matches Buergi's fminunc call.

        Parameters
        ----------
        choices : np.ndarray, shape (T,)
            Participant's choices, 0-indexed (0=Rock, 1=Paper, 2=Scissors).
        opponent_choices : np.ndarray, shape (T,)
            Opponent's choices, 0-indexed.
        optim_n_it : int
            Top-N grid points per kappa to use as optimisation starting points
            (default 4, matching Buergi's config.optim_n_it).
        trial_in_block : np.ndarray, shape (T,), optional
            1-indexed trial-within-block counter. If given, attractions and
            beliefs reset to uniform at the start of each new block (matches
            Buergi et al.'s joint per-participant, per-block-reset fitting).

        Returns
        -------
        CHASEResult with fitted parameters and trial-by-trial estimates.
        """
        # Normalise to 0-indexed (accept either 0- or 1-indexed input)
        choices          = np.asarray(choices, dtype=int)
        opponent_choices = np.asarray(opponent_choices, dtype=int)
        if choices.min() >= 1:
            choices          = choices - 1
            opponent_choices = opponent_choices - 1

        best_ll     = -np.inf
        best_params = None

        # ── Stage 1: Cartesian grid LL evaluation ─────────────────────────────
        # All 5^4 = 625 (alpha, beta, gamma, lam) combinations per kappa value.

        grid_combos = list(itertools.product(
            GRID_VALUES["alpha"], GRID_VALUES["beta"],
            GRID_VALUES["gamma"], GRID_VALUES["lam"],
        ))

        for kappa_try in range(self.max_kappa + 1):

            grid_lls: list[tuple[float, float, float, float, float]] = []
            for (a, b, g, l) in grid_combos:
                params = {"alpha": a, "beta": b, "gamma": g, "lam": l, "kappa": kappa_try}
                ll = self.log_likelihood(params, choices, opponent_choices, trial_in_block)
                grid_lls.append((ll, a, b, g, l))

            # Sort descending by LL; take top-N as optimisation starting points
            grid_lls.sort(key=lambda t: -t[0])
            top_starts = grid_lls[:optim_n_it]

            # ── Stage 2: optimise in unconstrained estimation space ────────────

            for ll_init, a0, b0, g0, l0 in top_starts:
                x0 = _to_est(a0, b0, g0, l0)

                def neg_ll_est(x, _k=kappa_try):
                    return -self.log_likelihood(
                        {**_from_est(x), "kappa": _k},
                        choices, opponent_choices, trial_in_block,
                    )

                try:
                    opt = minimize(
                        neg_ll_est, x0,
                        method  = "BFGS",          # matches fminunc in MATLAB
                        options = {"maxiter": 500, "gtol": 1e-6},
                    )
                    fitted = {**_from_est(opt.x), "kappa": kappa_try}
                    ll = self.log_likelihood(fitted, choices, opponent_choices, trial_in_block)
                    if ll > best_ll:
                        best_ll     = ll
                        best_params = fitted

                except Exception as e:
                    warnings.warn(f"Optimisation failed (kappa={kappa_try}, start={a0:.2f},{b0:.2f}): {e}")

        if best_params is None:
            warnings.warn("All optimisation restarts failed.")
            return CHASEResult()

        # Run forward pass with best parameters to get trial-by-trial outputs
        result = self.simulate(best_params, choices, opponent_choices, trial_in_block)
        result.log_likelihood = best_ll
        result.aic            = -2 * best_ll + 2 * (self.N_PARAMS - 1)  # kappa is discrete
        result.converged      = True
        return result
