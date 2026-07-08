#!/usr/bin/env python3
"""
Verify that the CHASEAgent (frontend TypeScript bot) behaves as expected before
running participants.

This script is a Python mirror of agents.ts and runs three checks:

  1. LEVEL DIFFERENTIATION
     Against a Rock-biased participant (70% Rock), does each level do the right thing?
       Level 0: plays own habit — stays near uniform, ignores participant
       Level 1: best-responds to participant's tracker — rapidly shifts to Paper
       Level 2: two-step reasoning from own habits — distinct from both 0 and 1

  2. WSLS NOISE
     Does the noise mechanism fire at the right times and rates?
       Win streak ≥ 2 (k=0) or ≥ 3 (k>0) → deterministically noisy
       Lose/tie streak ≥ 3-4 → deterministically noisy
       Noise = plays non-optimal action (not max-probability choice)

  3. BLOCK RESET
     After reset(), do both trackers return to uniform?

Run from repo root:
  python3 analysis/verify_agent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (BOT_ALPHA, BOT_BETA, BOT_LAMBDA, N_ACTIONS,
                    NOISE_TIME_HORIZON, NOISE_SUCCESS_CRIT, NOISE_SKEWNESS)


# ── Python mirror of TypeScript CHASEAgent (agents.ts) ───────────────────────
# Keep this in sync with any changes to agents.ts.

class CHASEBot:
    """Python replica of agents.ts CHASEAgent for offline verification."""

    PAYOFF = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

    def __init__(self, level: int, alpha: float = BOT_ALPHA, beta: float = BOT_BETA,
                 lam: float = BOT_LAMBDA, seed=None):
        self.level = level
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.attr   = np.ones(3) / 3   # bot's own action frequencies
        self.p_attr = np.ones(3) / 3   # participant's action frequencies (tracked by bot)
        self.scores: list[int] = []    # +1 participant win, -1 lose, 0 tie
        self.last_was_noisy = False    # set by choose(); avoids double-calling _is_noisy()

    def _softmax(self, v: np.ndarray) -> np.ndarray:
        z = self.beta * v; z -= z.max()
        e = np.exp(z); return e / e.sum()

    def _ev(self, dist: np.ndarray) -> np.ndarray:
        """Expected value for each bot action against opponent distribution `dist`."""
        payoff = self.PAYOFF.copy(); payoff[payoff < 0] *= self.lam
        return payoff @ dist

    def _probs(self) -> np.ndarray:
        """
        Action probabilities matching mn_RPS_task.m:
          k=0: softmax(attr, β)
          k=1: softmax(π @ p_attr, β)          raw p_attr, no initial softmax
          k=2: softmax(π @ softmax(π @ attr, β), β)
        """
        if self.level == 0:
            return self._softmax(self.attr)
        seed = self.attr if self.level % 2 == 0 else self.p_attr
        p = self._softmax(self._ev(seed))
        for _ in range(1, self.level):
            p = self._softmax(self._ev(p))
        return p

    def _is_noisy(self) -> bool:
        """
        Adaptive WSLS noise matching get_noise_level() in mn_RPS_task.m.
        Returns True → play non-optimal action this trial.
        """
        if not self.scores:
            return False

        time_horizon   = NOISE_TIME_HORIZON
        success_crit   = NOISE_SUCCESS_CRIT
        skewness       = NOISE_SKEWNESS
        max_win_streak = 2 if self.level == 0 else 3   # streak_bounds[2]
        streak_max     = int(self.rng.integers(3, 5))  # randi([3 4])

        recent  = np.array(self.scores[-time_horizon:])
        success = float(np.mean(recent > 0))

        # Consecutive wins at end, capped at max_win_streak (mirrors TypeScript loop)
        win_streak = 0
        for s in reversed(self.scores):
            if s > 0:
                win_streak += 1
                if win_streak >= max_win_streak:
                    break
            else:
                break

        # Consecutive losses at end, capped at streak_max
        lose_streak = 0
        for s in reversed(self.scores):
            if s < 0:
                lose_streak += 1
                if lose_streak >= streak_max:
                    break
            else:
                break

        # Consecutive ties at end, capped at streak_max
        tie_streak = 0
        for s in reversed(self.scores):
            if s == 0:
                tie_streak += 1
                if tie_streak >= streak_max:
                    break
            else:
                break

        if lose_streak >= streak_max or tie_streak >= streak_max:
            return True

        if success >= success_crit and win_streak >= 1:
            # Check deterministic case first (avoids division by zero)
            if win_streak >= max_win_streak:
                return True
            chance = (1 / (max_win_streak + 1 - win_streak)) ** skewness
            if self.rng.random() < chance:
                return True

        return False

    def choose(self) -> int:
        p = self._probs().copy()
        self.last_was_noisy = self._is_noisy()   # store so callers don't need a second call
        if self.last_was_noisy:
            p[np.argmax(p)] = 0
            p /= p.sum()
        return int(self.rng.choice(3, p=p))

    def update(self, own: int, opp: int, score: int | None = None):
        """own = bot choice, opp = participant choice, score = participant outcome."""
        ind_own = np.zeros(3); ind_own[own] = 1
        self.attr += self.alpha * (ind_own - self.attr)
        ind_opp = np.zeros(3); ind_opp[opp] = 1
        self.p_attr += self.alpha * (ind_opp - self.p_attr)
        if score is not None:
            self.scores.append(score)

    def reset(self):
        self.attr[:]   = 1 / 3
        self.p_attr[:] = 1 / 3
        self.scores.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

PAYOFF = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

def outcome(p_choice: int, a_choice: int) -> int:
    """Return participant's score: +1 win, -1 lose, 0 tie."""
    return int(PAYOFF[p_choice, a_choice])

ACTION_NAMES = ["Rock", "Paper", "Scissors"]


# ── Check 1: Level differentiation ───────────────────────────────────────────

def check_level_differentiation(n_sims: int = 2000, n_trials: int = 40, seed: int = 0):
    """
    Simulate n_sims blocks at each level against a participant who plays Rock 70%
    of the time. Compare the bot's action distributions across levels.

    Expected:
      Level 0 → stays near uniform (ignores participant)
      Level 1 → shifts heavily to Paper (exploits Rock bias)
      Level 2 → distinct from both, driven by own-habit recursion
    """
    print("\n" + "="*60)
    print("CHECK 1: Level differentiation")
    print("Participant strategy: Rock 70%, Paper 15%, Scissors 15%")
    print(f"Settings: {n_sims} sims × {n_trials} trials/block")
    print("="*60)

    p_probs = np.array([0.70, 0.15, 0.15])  # participant's fixed bias
    rng = np.random.default_rng(seed)

    # For the figure: track Paper probability over trials (level 1 should rise fastest)
    paper_over_time = {}   # level → (n_trials,) array of P(Paper) averaged across sims
    final_dists     = {}   # level → avg action distribution at end of block

    checks_passed = []

    for level in [0, 1, 2]:
        paper_traj = np.zeros((n_sims, n_trials))
        final_actions = np.zeros(3)

        for sim in range(n_sims):
            bot = CHASEBot(level=level, seed=int(rng.integers(1e9)))
            for t in range(n_trials):
                p_choice = int(rng.choice(3, p=p_probs))
                paper_traj[sim, t] = bot._probs()[1]
                a_choice = bot.choose()
                score = outcome(p_choice, a_choice)
                bot.update(a_choice, p_choice, score)
            final_actions += bot._probs()

        paper_over_time[level] = paper_traj.mean(axis=0)
        final_dists[level] = final_actions / n_sims

        dist = final_dists[level]
        print(f"\n  Level {level} — final mean action probs:")
        for i, name in enumerate(ACTION_NAMES):
            print(f"    P({name:8s}) = {dist[i]:.3f}")

    # Checks
    p0 = final_dists[0][1]   # P(Paper) for level 0
    p1 = final_dists[1][1]   # P(Paper) for level 1
    p2 = final_dists[2][1]

    print("\n  --- Checks ---")

    c1 = p1 > 0.65
    checks_passed.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Level 1 plays Paper >65%  (got {p1:.3f})")

    c2 = p1 > p0 + 0.30
    checks_passed.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Level 1 exploits Rock bias more than level 0"
          f"  (level1={p1:.3f}, level0={p0:.3f}, diff={p1-p0:.3f})")

    c3 = abs(p2 - p1) > 0.10
    checks_passed.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Level 2 is distinct from level 1"
          f"  (level1={p1:.3f}, level2={p2:.3f}, diff={abs(p1-p2):.3f})")

    # Check convergence speed for level 1
    trial5_paper = paper_over_time[1][4]   # P(Paper) at trial 5 (0-indexed: t=4)
    c4 = trial5_paper > 0.60
    checks_passed.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] Level 1 reaches P(Paper)>60% by trial 5"
          f"  (got {trial5_paper:.3f})")

    return paper_over_time, final_dists, checks_passed


# ── Check 2: WSLS noise ───────────────────────────────────────────────────────

def check_wsls_noise(n_trials: int = 500, n_sims: int = 200, seed: int = 1):
    """
    Verify the noise mechanism fires at the right times.

    Three participant strategies:
      'win_always':  participant always plays the winning action → bot should go noisy often
      'lose_always': participant always plays the losing action  → bot should go noisy after 3-4 losses
      'tie_always':  participant always draws                    → bot should go noisy after 3-4 ties
      'random':      participant plays uniformly                 → moderate background noise
    """
    print("\n" + "="*60)
    print("CHECK 2: WSLS noise mechanism")
    print("="*60)

    rng = np.random.default_rng(seed)
    checks_passed = []

    # --- Simulate and count noise ---
    results = {}
    for strategy_name, strategy_fn, description in [
        ("random",      lambda a, rng: int(rng.integers(3)),       "Random opponent"),
        ("win_always",  lambda a, rng: int((a + 1) % 3),           "Participant always wins"),
        ("lose_always", lambda a, rng: int((a + 2) % 3),           "Participant always loses"),
        ("tie_always",  lambda a, rng: int(a),                      "Participant always ties"),
    ]:
        noise_rates = []
        for sim in range(n_sims):
            bot = CHASEBot(level=1, seed=int(rng.integers(1e9)))
            noisy_count = 0
            for _ in range(n_trials):
                is_noisy = bot._is_noisy()
                a_choice = bot.choose()
                p_choice = strategy_fn(a_choice, rng)
                score = outcome(p_choice, a_choice)
                bot.update(a_choice, p_choice, score)
                if is_noisy:
                    noisy_count += 1
            noise_rates.append(noisy_count / n_trials)
        results[strategy_name] = np.array(noise_rates)
        mean_rate = np.mean(noise_rates) * 100
        print(f"\n  {description}")
        print(f"    Noise rate: {mean_rate:.1f}%  (±{np.std(noise_rates)*100:.1f}%)")

    print("\n  --- Checks ---")

    # When participant always wins, noise rate should be high
    c1 = np.mean(results["win_always"]) > 0.25
    checks_passed.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Noise rate >25% when participant always wins"
          f"  (got {np.mean(results['win_always'])*100:.1f}%)")

    # When participant always loses, noise rate should also be substantial
    c2 = np.mean(results["lose_always"]) > 0.20
    checks_passed.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Noise rate >20% when participant always loses"
          f"  (got {np.mean(results['lose_always'])*100:.1f}%)")

    # When participant always ties, noise rate should be substantial
    c3 = np.mean(results["tie_always"]) > 0.20
    checks_passed.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Noise rate >20% when participant always ties"
          f"  (got {np.mean(results['tie_always'])*100:.1f}%)")

    # Random opponent should have moderate/low noise
    c4 = np.mean(results["random"]) < 0.25
    checks_passed.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] Noise rate <25% against random opponent"
          f"  (got {np.mean(results['random'])*100:.1f}%)")

    # Win-always should have higher noise than random
    c5 = np.mean(results["win_always"]) > np.mean(results["random"]) + 0.10
    checks_passed.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] Win-always triggers more noise than random"
          f"  (win={np.mean(results['win_always'])*100:.1f}%"
          f", random={np.mean(results['random'])*100:.1f}%)")

    # Verify noise excludes the max-probability action.
    # Use bot.last_was_noisy (set inside choose()) to avoid calling _is_noisy() twice
    # with different random draws for streak_max.
    bot = CHASEBot(level=1, seed=42)
    noisy_violations = 0
    noisy_trials = 0
    rng2 = np.random.default_rng(42)
    for _ in range(5000):
        p_ideal = bot._probs()
        best_action = int(np.argmax(p_ideal))
        a_choice = bot.choose()                  # sets bot.last_was_noisy
        p_choice = int(rng2.integers(3))
        score = outcome(p_choice, a_choice)
        bot.update(a_choice, p_choice, score)
        if bot.last_was_noisy:
            noisy_trials += 1
            if a_choice == best_action:
                noisy_violations += 1

    c6 = noisy_violations == 0
    checks_passed.append(c6)
    print(f"  [{'PASS' if c6 else 'FAIL'}] Noisy trials never play the max-probability action"
          f"  ({noisy_violations} violations in {noisy_trials} noisy trials)")

    return results, checks_passed


# ── Check 3: Block reset ──────────────────────────────────────────────────────

def check_block_reset():
    """After a block of play, reset() restores both trackers to uniform."""
    print("\n" + "="*60)
    print("CHECK 3: Block reset")
    print("="*60)
    checks_passed = []
    rng = np.random.default_rng(0)

    for level in [0, 1, 2]:
        bot = CHASEBot(level=level, seed=0)

        # Verify initial state is uniform
        c_init = np.allclose(bot.attr, 1/3) and np.allclose(bot.p_attr, 1/3)
        checks_passed.append(c_init)
        print(f"\n  Level {level}:")
        print(f"  [{'PASS' if c_init else 'FAIL'}] Initial trackers are uniform")

        # Play 40 biased trials
        for _ in range(40):
            p_choice = int(rng.choice(3, p=[0.7, 0.15, 0.15]))  # Rock-biased
            a_choice = bot.choose()
            score = outcome(p_choice, a_choice)
            bot.update(a_choice, p_choice, score)

        # Trackers should have drifted
        drifted = not (np.allclose(bot.attr, 1/3) and np.allclose(bot.p_attr, 1/3))
        print(f"  [{'OK ' if drifted else 'WARN'}] Trackers drifted during play"
              f"  (attr={bot.attr.round(3)}, p_attr={bot.p_attr.round(3)})")

        # Reset
        bot.reset()
        c_reset = np.allclose(bot.attr, 1/3) and np.allclose(bot.p_attr, 1/3)
        c_scores = len(bot.scores) == 0
        checks_passed.append(c_reset)
        checks_passed.append(c_scores)
        print(f"  [{'PASS' if c_reset else 'FAIL'}] Attraction trackers reset to uniform after reset()")
        print(f"  [{'PASS' if c_scores else 'FAIL'}] Score history cleared after reset()")

    return checks_passed


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(paper_over_time: dict, noise_results: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: P(Paper) over trials at each level vs Rock-biased participant
    ax = axes[0]
    colors = {0: "steelblue", 1: "tomato", 2: "seagreen"}
    for level in [0, 1, 2]:
        traj = paper_over_time[level]
        ax.plot(range(1, len(traj)+1), traj, color=colors[level],
                linewidth=2, label=f"Level {level}")
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=0.8, label="Uniform (1/3)")
    ax.set_xlabel("Trial within block")
    ax.set_ylabel("P(Paper)")
    ax.set_title("Bot's P(Paper) vs 70%-Rock participant\n"
                 "(Level 1 should exploit bias fastest)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(paper_over_time[0]))

    # Panel 2: Noise rate distribution by participant strategy
    ax = axes[1]
    strategy_labels = {
        "random":     "Random",
        "win_always": "Always wins",
        "lose_always":"Always loses",
        "tie_always": "Always ties",
    }
    positions = list(range(len(strategy_labels)))
    data = [noise_results[k] * 100 for k in strategy_labels]
    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, color in zip(bp["boxes"], ["lightblue", "tomato", "gold", "lightgreen"]):
        patch.set_facecolor(color)
    ax.set_xticks(positions)
    ax.set_xticklabels(strategy_labels.values(), fontsize=9)
    ax.set_ylabel("Noise rate (%)")
    ax.set_title("WSLS noise rate by participant strategy\n(Level 1 bot, 500 trials)")
    ax.axhline(20, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CHASEAgent verification")
    print(f"Parameters: alpha={BOT_ALPHA}, beta={BOT_BETA} (matching Buergi et al. mn_RPS_config.m)")

    paper_over_time, final_dists, checks1 = check_level_differentiation()
    noise_results,              checks2   = check_wsls_noise()
    checks3                               = check_block_reset()

    all_checks = checks1 + checks2 + checks3
    n_pass = sum(all_checks)
    n_total = len(all_checks)

    print("\n" + "="*60)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("All checks passed — agent matches expected Buergi et al. behavior.")
    else:
        print(f"WARNING: {n_total - n_pass} check(s) failed — review output above.")
    print("="*60)

    make_figure(paper_over_time, noise_results,
                "analysis/verify_agent_output.png")
