# ADR-0001: RL framework — self-built PyTorch PPO

- Status: **Accepted**
- Date: 2026-06-08

## Context

The long-term ceiling of this project depends on a deeply customized model:
entity-list + attention encoder, recurrent memory (GRU/LSTM), hybrid action space
(discrete movement/aim + button bitmask + duration + optional macro), and
per-component action masking. We also need local inference / remote training with
explicit `policy_version` staleness control.

Candidates: self-built PyTorch PPO; Ray RLlib; SB3-contrib RecurrentPPO.

## Decision

Build PPO / RecurrentPPO / APPO **in-house on PyTorch** (`hkrl/training/`), with a
clean `ActorCritic` interface (`hkrl/models/base.py`) and a learner interface that
a future RLlib/Ray backend could implement.

## Rationale

- **Maximum control** over the model: entity-attention + recurrence + hybrid
  heads + masking are awkward to express in SB3 and verbose in RLlib.
- **Hybrid action + mask** need custom log-prob/entropy per head — trivial in-house,
  painful in wrappers.
- **Recurrent sequence training** (truncated BPTT + burn-in + mask) is easier to
  get correct when we own the buffer.
- We keep the **learner/worker boundary** abstract, so RLlib can be slotted in for
  distributed scale later without rewriting the model.

## Consequences

- We own correctness of GAE, clipping, KL, sequence masking (covered by tests).
- More upfront code than SB3, but no framework-fighting at the ceiling.
- Distributed scale (Phase 8) may still adopt Ray for the *fabric*, not the algo.

## Alternatives rejected

- **SB3-contrib RecurrentPPO** — fastest start, but entity-list + attention +
  custom masks become a long-term bottleneck.
- **Ray RLlib** — great distributed primitives, but deep model customization is
  cumbersome; revisit only for the worker/coordinator fabric.
