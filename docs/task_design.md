# Task Design Specification

## Social Cognition Task — RPS Component

**Version:** 2026-05-19  
**Session context:** Administered in-scanner, after conversation phase and face-viewing task  
**Duration:** ~20–25 minutes

---

## Purpose

This task measures whether felt connection with AI agents changes how participants cognitively represent each agent — specifically, whether they attribute a mind to it. The core measure is implicit, online mentalizing captured through Rock-Paper-Scissors (RPS): each round requires predicting the opponent's choice and selecting a counter, a core theory-of-mind operation.

---

## Agents

Four AI agents (A, B, C, D) that participants have previously interacted with during the conversation phase, ordered by decreasing social connection (A = highest, D = lowest). A fifth condition uses a random number generator (RNG) with no opponent.

Critically, **all four social agents play at the same strategic level** (k ≈ 1–2 in the CHASE framework): sophisticated enough to engage mentalizing, not so difficult as to produce ceiling/floor effects. Any mentalizing gradient across agents is therefore attributable to prior connection, not difficulty.

---

## Trial Structure

```
┌─────────────────────────────────────────────────────┐
│  1. OPPONENT DISPLAY    (~1 s)                       │
│     Avatar (agent) or generic icon (RNG) shown       │
│                                                       │
│  2. CHOICE              (self-paced, max 5 s)        │
│     1 = Rock / 2 = Paper / 3 = Scissors              │
│                                                       │
│  3. OUTCOME FEEDBACK    (~1.5 s)                     │
│     Win / Lose / Draw + points delta                 │
│                                                       │
│  4. ITI (fixation)      (0.5–2.0 s, jittered)       │
└─────────────────────────────────────────────────────┘
```

Average trial duration: ~7 s.

---

## Block Structure

| Block | Agent | Trials | Duration |
|-------|-------|--------|----------|
| 1–5 (counterbalanced) | A, B, C, D, RNG | 35 per block | ~4 min |

Total: 5 × 35 = 175 trials, ~20–25 min.

Block order is counterbalanced across participants using a Latin square on the 4 social agents, with RNG inserted at one of 5 positions (also counterbalanced).

---

## Payoff Structure

| Outcome | Points |
|---------|--------|
| Win     | +3     |
| Lose    | −3     |
| Draw    | 0      |

Starting balance: 100 points. Final score reported at end of session.

---

## RNG Control Condition

- No opponent — a random draw
- Choice is Rock, Paper, or Scissors selected uniformly at random
- Participants are told there is no opponent; a generic non-face icon is shown
- Provides a non-social baseline: no mind to model, so any mentalizing-region activation during agent blocks relative to RNG reflects agent-directed mentalizing
- Strategic level = k = 0

---

## Stimuli

### Avatars
- Four distinct face images, one per social agent (A–D)
- Matched on low-level visual features (luminance, spatial frequency) where possible
- Same avatar used throughout the entire block

### RNG Icon
- Generic abstract symbol (e.g., dice or shuffle icon)
- No face-like features

---

## Behavioral Outcome Measures

| Measure | Operationalisation | Prediction |
|---------|--------------------|------------|
| Win rate | Proportion wins (chance = 1/3) | A > B > C > D > RNG |
| Choice entropy | Shannon entropy of choice distribution | A < B < C < D < RNG |
| Win-stay rate | P(repeat \| win) | A > B > C > D > RNG |
| Lose-shift rate | P(switch \| lose) | A > B > C > D > RNG |
| Choice autocorrelation | Lag-1 sequential dependency | A > B > C > D > RNG |

---

## Computational Model: CHASE

The **Cognitive Hierarchy Assessment with Sophistication Estimation** (Buergi et al., 2026) is applied to each participant × agent condition separately.

### Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Learning rate | α | Speed of action-frequency updating |
| Softmax temp | β | Noisiness of action selection |
| Level sensitivity | γ | Sensitivity to evidence about opponent's sophistication (Bayesian learning rate analog) |
| Loss sensitivity | λ | Asymmetric weighting of wins vs. losses |
| Max sophistication | κ | Participant's maximum reasoning depth |

### Trial-level outputs

| Output | Description |
|--------|-------------|
| k | Inferred strategic level used by participant |
| B(k)_t | Belief distribution over opponent levels at trial t |
| BU_t | Belief update magnitude = KL(B(k)_t \|\| B(k)_{t-1}) |
| APE_t | Action prediction error = 1 − P(a_opp \| current beliefs) |
| CV_t | Choice value = expected payoff of chosen action |

### Prediction

Monotonic gradient in CHASE parameters: γ (level sensitivity) and mean BU should be higher for higher-connection agents, reflecting more active mentalizing. κ is predicted to be similar across agents (it is a participant-level trait, not connection-dependent).

---

## Task Placement Rationale

Administered after the conversation phase and face-viewing task (not interleaved with conversations) because:

1. **Avoid priming mentalizing** during naturalistic conversation
2. **Test persistence** of connection-induced mentalizing beyond the interaction
3. **Clean analytic separation**: conversation → social dynamics data; face-viewing → RSA data; RPS → ToM data

---

## Scanner Administration Notes

- Response device: MRI-compatible button box, keys 1/2/3
- Scanner TTL trigger received before each block
- ITIs jittered to maximise design efficiency and decorrelate phases
- Motion parameters and physiological noise regressors included in fMRI GLM

---

## References

Buergi, N., Aydogan, G., Konovalov, A. & Ruff, C.C. (2026). A neural signature of adaptive mentalization. *Nature Neuroscience*, 29, 934–944. https://doi.org/10.1038/s41593-026-02219-x
