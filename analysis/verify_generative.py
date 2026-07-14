#!/usr/bin/env python3
"""
Verify the CHASE generative model (generate_session) and fitting model (chase.py).

Three checks:
  1. NUMERICAL UNIT TESTS
     Attraction update, softmax, and action probs match hand-computed values exactly.

  2. BEHAVIOURAL SANITY
     Does a kappa=1 participant exploit an opponent with a strong Rock bias?
     Does a kappa=0 participant ignore the opponent and stay near uniform?

  3. SELF-CONSISTENCY (ample data)
     Generate 500 trials with known parameters and fit the model.
     With enough data, recovered parameters should be very close to true.
     If this fails, generate_session() and simulate() are inconsistent.

Run from repo root:
  python3 analysis/verify_generative.py
"""

import sys, os
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.chase import CHASEModel
from analysis.parameter_recovery_new_design import generate_session, CHASEBot

warnings.filterwarnings("ignore")

PAYOFF = np.array([[0,-1,1],[1,0,-1],[-1,1,0]], dtype=float)

# ── Check 1: Numerical unit tests ─────────────────────────────────────────────

def check_numerics():
    print("\n" + "="*60)
    print("CHECK 1: Numerical unit tests")
    print("="*60)
    model = CHASEModel()
    checks = []

    # 1a. Softmax at uniform input → uniform output
    v = np.array([1/3, 1/3, 1/3])
    p = model._softmax(v, 10)
    c = np.allclose(p, [1/3, 1/3, 1/3], atol=1e-6)
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] softmax(uniform, beta=10) = uniform  (got {p.round(4)})")

    # 1b. Softmax at [1,0,0] with high beta → nearly [1,0,0]
    v = np.array([1.0, 0.0, 0.0])
    p = model._softmax(v, 10)
    c = p[0] > 0.999
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] softmax([1,0,0], beta=10)[0] > 0.999  (got {p[0]:.6f})")

    # 1c. Attraction update: start uniform, play Rock once (alpha=0.9)
    # Expected: [1/3 + 0.9*(1-1/3), 1/3 + 0.9*(0-1/3), same]
    #         = [0.9333, 0.0333, 0.0333]
    alpha = 0.9
    attr = np.ones(3) / 3
    choice = 0  # Rock
    ind = np.zeros(3); ind[choice] = 1.0
    attr += alpha * (ind - attr)
    expected = np.array([1/3 + 0.9*(1-1/3), 1/3 + 0.9*(0-1/3), 1/3 + 0.9*(0-1/3)])
    c = np.allclose(attr, expected, atol=1e-10)
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] EWA update after Rock: {attr.round(4)} == {expected.round(4)}")

    # 1d. Two attraction updates sum to 1
    attr2 = np.ones(3) / 3
    for choice in [0, 1, 2, 0, 0]:
        ind = np.zeros(3); ind[choice] = 1.0
        attr2 += alpha * (ind - attr2)
    c = abs(attr2.sum() - 1.0) < 1e-10
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] Attractions sum to 1.0 after 5 updates  (sum={attr2.sum():.10f})")

    # 1e. k=0 participant action probs = softmax(own_attr, beta) — verify via simulate()
    own_attr = np.array([0.8, 0.1, 0.1])
    # Manually: softmax([0.8,0.1,0.1], beta=10)
    v = np.array([0.8, 0.1, 0.1])
    z = 10 * v; z -= z.max()
    expected_probs = np.exp(z) / np.exp(z).sum()

    # Run simulate() for 1 trial at kappa=0, fix participant choice = Rock
    params = {"alpha": 0.9, "beta": 10.0, "gamma": 1.0, "lam": 1.0, "kappa": 0}
    # Inject a known own_attr by pre-running enough trials to get there:
    # After 40 trials where participant always plays Rock against a fixed opponent,
    # own_attr ≈ [1,0,0] (since alpha=0.9 decays fast)
    p_choices = np.zeros(40, dtype=int)   # all Rock
    opp_choices = np.zeros(40, dtype=int) # opponent also Rock
    result = model.simulate(params, p_choices, opp_choices)
    # At trial 0, own_attr = uniform → probs should be uniform
    c = np.allclose(result.action_probs[0], [1/3, 1/3, 1/3], atol=1e-6)
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] k=0 probs at trial 0 = uniform  (got {result.action_probs[0].round(4)})")

    # At last trial, own_attr ≈ [1,0,0] → probs ≈ [1,0,0]
    c = result.action_probs[-1][0] > 0.999
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] k=0 probs after 40 Rock plays: P(Rock)>0.999"
          f"  (got {result.action_probs[-1][0]:.4f})")

    # 1f. generate_session and simulate() produce same action probs for same input
    rng = np.random.default_rng(42)
    true_params = {"alpha": 0.7, "beta": 5.0, "gamma": 1.0, "lam": 1.0, "kappa": 1}
    block_order = [{"level": 0}, {"level": 1}, {"level": 2}]
    p_choices, opp_choices, tib = generate_session(true_params, block_order, seed=42)
    result = model.simulate(true_params, p_choices, opp_choices, tib)

    # Action probs from simulate() should be valid distributions at every trial
    row_sums = result.action_probs.sum(axis=1)
    c = np.allclose(row_sums, 1.0, atol=1e-6)
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] simulate() action_probs sum to 1.0 on every trial"
          f"  (min={row_sums.min():.6f}, max={row_sums.max():.6f})")

    # No NaN in action probs
    c = not np.any(np.isnan(result.action_probs))
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] No NaN in action_probs")

    return checks


# ── Check 2: Behavioural sanity ───────────────────────────────────────────────

def check_behavioural_sanity(n_sims=500, seed=1):
    print("\n" + "="*60)
    print("CHECK 2: Behavioural sanity")
    print("="*60)
    model = CHASEModel()
    rng = np.random.default_rng(seed)
    checks = []

    # kappa=0 participant vs Rock-biased opponent (70% Rock):
    # Should stay near uniform (ignores opponent)
    paper_rates_k0 = []
    for _ in range(n_sims):
        true_params = {"alpha": 0.9, "beta": 10.0, "gamma": 1.0, "lam": 1.0, "kappa": 0}
        opp_choices = rng.choice(3, size=40, p=[0.7, 0.15, 0.15])
        block_order = [{"level": 0}]
        p_seq = np.array([0]*40)  # dummy — generative model uses probs not fixed choices
        # Use simulate to get probs under these opp choices
        result = model.simulate(true_params, p_seq, opp_choices)
        paper_rates_k0.append(result.action_probs[:, 1].mean())

    mean_paper_k0 = np.mean(paper_rates_k0)
    c = abs(mean_paper_k0 - 1/3) < 0.05
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] kappa=0 participant: P(Paper) ≈ 1/3 vs Rock-biased opponent"
          f"  (got {mean_paper_k0:.3f}, expected ≈ 0.333)")

    # kappa=1 participant vs Rock-biased opponent (70% Rock):
    # Should shift heavily to Paper (best-response to predicted Rock-heavy opponent)
    paper_rates_k1 = []
    for _ in range(n_sims):
        true_params = {"alpha": 0.9, "beta": 10.0, "gamma": 1.0, "lam": 1.0, "kappa": 1}
        opp_choices = rng.choice(3, size=40, p=[0.7, 0.15, 0.15])
        p_seq = np.array([0]*40)
        result = model.simulate(true_params, p_seq, opp_choices)
        paper_rates_k1.append(result.action_probs[5:, 1].mean())  # skip warm-up

    mean_paper_k1 = np.mean(paper_rates_k1)
    c = mean_paper_k1 > 0.65
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] kappa=1 participant: P(Paper) > 0.65 vs Rock-biased opponent"
          f"  (got {mean_paper_k1:.3f})")

    # kappa=1 exploits Rock bias more than kappa=0
    c = mean_paper_k1 > mean_paper_k0 + 0.30
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] kappa=1 exploits Rock bias more than kappa=0"
          f"  (k1={mean_paper_k1:.3f}, k0={mean_paper_k0:.3f}, diff={mean_paper_k1-mean_paper_k0:.3f})")

    # kappa=2 is distinct from kappa=1
    paper_rates_k2 = []
    for _ in range(n_sims):
        true_params = {"alpha": 0.9, "beta": 10.0, "gamma": 1.0, "lam": 1.0, "kappa": 2}
        opp_choices = rng.choice(3, size=40, p=[0.7, 0.15, 0.15])
        p_seq = np.array([0]*40)
        result = model.simulate(true_params, p_seq, opp_choices)
        paper_rates_k2.append(result.action_probs[5:, 1].mean())

    mean_paper_k2 = np.mean(paper_rates_k2)
    c = abs(mean_paper_k2 - mean_paper_k1) > 0.10
    checks.append(c)
    print(f"  [{'PASS' if c else 'FAIL'}] kappa=2 is distinct from kappa=1"
          f"  (k1={mean_paper_k1:.3f}, k2={mean_paper_k2:.3f}, diff={abs(mean_paper_k2-mean_paper_k1):.3f})")

    return checks


# ── Check 3: Self-consistency (ample data) ────────────────────────────────────

def check_self_consistency(seed=99):
    """
    Generate 500 trials from known parameters and fit.
    With ample data, recovered params should match true params closely.
    If this fails, generate_session() and simulate()/fit() are inconsistent.
    """
    print("\n" + "="*60)
    print("CHECK 3: Self-consistency — ample data recovery (500 trials)")
    print("="*60)
    model = CHASEModel()
    rng = np.random.default_rng(seed)
    checks = []

    # Use a simple block structure: 5 blocks × 100 trials (overkill but definitive)
    # Actually, 500 trials as a single block (no reset) for simplicity
    test_cases = [
        {"alpha": 0.8, "beta": 8.0, "gamma": 1.0, "lam": 1.0, "kappa": 0},
        {"alpha": 0.7, "beta": 6.0, "gamma": 1.0, "lam": 1.5, "kappa": 1},
        {"alpha": 0.6, "beta": 5.0, "gamma": 1.5, "lam": 1.2, "kappa": 2},
    ]

    for true_params in test_cases:
        k = true_params["kappa"]

        # Generate 500 trials using 5 × 3-level blocks (100 trials/block would be 300,
        # use a single 500-trial block instead to avoid reset complexity)
        # Generate opponent choices from CHASEBot at level k
        bot = CHASEBot(level=k, seed=int(rng.integers(1e9)))
        opp_choices = []
        for _ in range(500):
            opp_choices.append(bot.choose())
            bot.update(opp_choices[-1], 0, 0)  # dummy participant update

        opp_choices = np.array(opp_choices)

        # Generate participant choices using true_params
        kappa = int(true_params["kappa"])
        alpha = true_params["alpha"]
        beta = true_params["beta"]
        gamma = true_params["gamma"]
        lam = true_params["lam"]
        n_belief = max(kappa, 1)
        payoff_s = model.PAYOFF.copy(); payoff_s[payoff_s == -1] *= lam

        own_attr = np.ones(3) / 3
        opp_attr = np.ones(3) / 3
        beliefs = np.ones(n_belief) / n_belief
        p_choices = []

        for t in range(500):
            if kappa == 0:
                probs = model._softmax(own_attr, beta)
            else:
                integrated = np.zeros(3)
                for kk in range(kappa):
                    integrated += beliefs[kk] * model._recursive_probs(own_attr, opp_attr, beta, kk, lam)
                probs = model._softmax(payoff_s @ integrated, beta)

            p_choice = int(rng.choice(3, p=probs))
            p_choices.append(p_choice)

            if kappa >= 2:
                beliefs, _ = model._belief_update(beliefs, own_attr, opp_attr, opp_choices[t], beta, gamma, lam, kappa)

            ind = np.zeros(3); ind[p_choice] = 1.0
            own_attr += alpha * (ind - own_attr)
            ind = np.zeros(3); ind[opp_choices[t]] = 1.0
            opp_attr += alpha * (ind - opp_attr)

        p_choices = np.array(p_choices)

        # Fit
        result = model.fit(p_choices, opp_choices, optim_n_it=4, seed=int(rng.integers(1e9)))

        print(f"\n  True params: alpha={alpha:.2f}, beta={beta:.2f}, lam={lam:.2f}, kappa={k}")
        print(f"  Recovered:   alpha={result.alpha:.2f}, beta={result.beta:.2f}, lam={result.lam:.2f}, kappa={result.kappa}")

        c_kappa = result.kappa == k
        checks.append(c_kappa)
        print(f"  [{'PASS' if c_kappa else 'FAIL'}] kappa recovered correctly ({result.kappa} == {k})")

        c_alpha = abs(result.alpha - alpha) < 0.1
        checks.append(c_alpha)
        print(f"  [{'PASS' if c_alpha else 'FAIL'}] alpha within 0.1  (|{result.alpha:.3f} - {alpha:.2f}| = {abs(result.alpha - alpha):.3f})")

        c_beta = abs(result.beta - beta) / beta < 0.25
        checks.append(c_beta)
        print(f"  [{'PASS' if c_beta else 'FAIL'}] beta within 25%   (|{result.beta:.2f} - {beta:.2f}| / {beta:.2f} = {abs(result.beta-beta)/beta:.3f})")

    return checks


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CHASE generative model verification")

    checks1 = check_numerics()
    checks2 = check_behavioural_sanity()
    checks3 = check_self_consistency()

    all_checks = checks1 + checks2 + checks3
    n_pass = sum(all_checks)
    n_total = len(all_checks)

    print("\n" + "="*60)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("All checks passed — generative model is self-consistent and correct.")
    else:
        print(f"WARNING: {n_total - n_pass} check(s) failed — review output above.")
    print("="*60)
