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

Two AI agents that participants have previously interacted with during the conversation phase:

- **Friendly agent** — the high-connection agent (warm, engaging prior conversation)
- **Neutral agent** — the low-connection agent (minimal, task-like prior conversation)

Plus a **random draw (RNG)** control with no opponent.

Both social agents play at the **same strategic level** (k ≈ 1–2 in CHASE). Any mentalizing difference between conditions is therefore attributable to prior connection, not difficulty.

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
| 1–3 (counterbalanced) | friendly, neutral, rng | 35 per block | ~4 min |

Total: 3 × 35 = 105 trials, ~12–15 min.

Block order counterbalanced across participants using all 6 permutations of the 3 conditions.

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
| Win rate | Proportion wins (chance = 1/3) | friendly > neutral > RNG |
| Choice entropy | Shannon entropy of choice distribution | friendly < neutral < RNG |
| Win-stay rate | P(repeat \| win) | friendly > neutral > RNG |
| Lose-shift rate | P(switch \| lose) | friendly > neutral > RNG |
| Choice autocorrelation | Lag-1 sequential dependency | friendly > neutral > RNG |

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

γ (level sensitivity) and mean BU should be higher for the friendly agent than the neutral agent, reflecting more active mentalizing toward a socially connected partner. κ is a participant-level trait and is not predicted to differ across conditions.

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
