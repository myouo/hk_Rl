# ADR-0007: Factorized primitive actions with a semantic combination catalog

- Status: **Accepted**
- Date: 2026-07-26

## Context

The primitive policy space contains `3 × 3 × 2^9 × 4 = 18,432` joint choices
before the optional macro branch. Enumerating meaningful variants such as
jump + down-slash, jump + up-slash, airborne spell directions, and charged
nail-art releases as another categorical action would duplicate primitive
controls, grow the policy head, and create two probability paths for the same
game input. Multi-tick combinations also depend on changing cooldown, resource,
grounded, and FSM state, so a static combo id cannot make an entire sequence
valid in advance.

The existing policy already has small action branches, a live per-branch action
mask, and recurrent memory. This is consistent with the action-dimension
scaling motivation in the
[Branching Dueling Q-Network paper](https://arxiv.org/abs/1711.08946), while
[invalid-action masking](https://arxiv.org/abs/2006.14171) gives a policy-gradient
compatible way to prevent unavailable primitives. PPO itself is on-policy
([Schulman et al.](https://arxiv.org/abs/1707.06347)); injecting externally
selected combo actions into ordinary rollouts without their behavior
probability would invalidate that contract.

## Decision

- Keep the policy action as the existing factorization:
  `movement_x`, `aim_y`, nine button Bernoullis, `duration`, and an optional
  macro. Do not add a combo id to FlatBuffers, Gymnasium, or the model head.
- Publish a process-local, immutable, versioned semantic catalog in
  `hkrl.action_combinations`. Version 1 contains 18 useful simultaneous or
  temporal motifs. IDs are contiguous and append-only.
- Expose the static catalog once through
  `HKRLEnv.action_combination_catalog`. At each reset/step, derive only a compact
  integer availability bitset from the existing action mask plus grounded/soul
  state. Bit `combo_id` maps to that catalog entry.
- Treat the bitset as conservative discovery/diagnostic metadata. Every phase
  of a temporal combination must still pass the live action mask when sampled.
  The recurrent policy learns timing and continuation from observation, prior
  action/reward, and GRU state.
- Use entropy regularization for ordinary recurrent-PPO exploration. A small
  UCB1 selector may prioritize unseen/under-tested catalog entries only in
  separately labelled scripted smoke tests or curriculum data collection. It
  must not override actions inside PPO/APPO rollouts.
- Record combination success as coverage evidence, not as reward or capability.
  Capability remains hitless win rate, damage taken, and time to kill.

## Consequences

- The model retains the full primitive-composition ceiling and can discover
  combinations not yet named in the catalog.
- Catalog lookup adds no network bytes and no model parameters. Dynamic
  availability is `O(K)` over 18 entries and is represented by one Python
  integer instead of per-step strings/lists.
- Existing checkpoints and schema version 6 remain compatible.
- Independent action branches do not explicitly model conditional covariance
  between sampled components. The shared recurrent representation can still
  drive coordinated high-probability choices. If this becomes the measured
  bottleneck, an autoregressive branch head is the preferred future experiment;
  it preserves primitives without introducing combo ids.

## Alternatives rejected

- **One categorical head over all 18,432 joint actions.** Excessive logits,
  sparse exploration, and expensive joint masking.
- **Add 18 combo ids beside the primitive heads.** Duplicated semantics and
  ambiguous action probabilities; named combos would cap unseen compositions.
- **Convert every useful combination into a Mod macro.** Cheap to bootstrap but
  fixes timing and reduces the high-ceiling primitive policy the arena needs.
- **Use UCB1-selected combos directly in PPO rollouts.** This changes the
  behavior policy without recording the correct joint log-probability.
