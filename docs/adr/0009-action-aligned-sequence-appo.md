# ADR-0009: Action-aligned options and sequence-preserving APPO

- Status: Accepted
- Date: 2026-07-26

## Context

The v0.8.0 Mod action vector already expresses directional movement, independent
combat buttons, variable holds, and optional primitive-only macros. Adding a
combination id would duplicate those controls and break schema/checkpoint
compatibility (ADR-0007). The post-release audit instead found two probability
and timing mismatches:

1. When `macro > 0`, the Mod ignores the sampled primitive branches, but PPO
   included those ignored branches in behavior/current log-probabilities.
2. A duration or macro plan could continue after the synchronous STEP response.
   The worker then recorded a newly sampled action while the Mod was still
   executing the previous option.

Uploaded recurrent APPO rollouts also retained every behavior hidden state but
flattened transitions into one-step evaluations, so the GRU received no
truncated backpropagation through time. Finally, one optimizer update per small
worker upload underused the remote GPU.

## Decision

Keep the FlatBuffers and Gym action dimensions unchanged, and align their
semantics:

1. Treat the macro branch as a hierarchy. `macro=0` delegates to the primitive
   distribution, so its log-probability is
   `log p(macro=0) + log p(primitives)`. `macro>0` uses only
   `log p(macro)` because the Mod ignores primitive samples. Entropy is the
   exact mixture `H(macro) + p(macro=0) H(primitives)`. Stored ignored
   primitive fields are canonical neutral values, not random recurrent context.
   Episode-reset recurrent context uses the same neutral discrete ids
   (`movement=neutral`, `aim=neutral`) instead of an all-zero vector that would
   incorrectly mean left/down.
2. Choose STEP `action_repeat` per policy decision as the maximum of the task
   minimum, selected primitive duration, and selected macro-plan length.
   Terminal events may still end the response early. A new policy action is not
   sampled until the selected option has finished.
3. Mark this behavior-probability change with RolloutBatch NPZ/envelope format
   v3. Old v2 rollouts are rejected instead of being mixed with new log-prob
   semantics.
4. Record elapsed time as `discount_exponents` in base task-decision units.
   GAE uses `gamma^exponent` and `lambda^exponent`, so a long option receives
   the correct semi-Markov discount. A time-limit truncation evaluates its
   terminal observation for bootstrap, then stops the GAE trace so the reset
   episode cannot leak backward.
5. Reconstruct uploaded APPO rollouts into fixed-length contiguous sequences.
   Split on terminal/truncation, `episode_id`, and `task_id`; seed each chunk
   with the recorded behavior hidden state; and mask padding in every loss and
   metric. The final minibatch is padded and masked to retain a fixed compiled
   shape.
6. Combine `learner.batches_per_update` worker rollouts into one GPU update.
   The default remote profile uses four 2,048-step rollouts, 1,024-transition
   sequence minibatches, and two PPO epochs.
7. Keep inference local. Consolidate the policy action/log-prob/value CPU copy
   into one synchronization, and reuse one compressed rollout payload when
   both spooling and uploading.
8. Add one typed experiment manifest plus remote-training/local-inference
   launchers. TrainConfig remains the source of hyperparameters; the manifest
   binds roles, tasks, paths, and endpoints.

## Consequences

- Existing Mod v0.8.0 and schema-v6 checkpoints remain loadable; no game action
  or observation field changes.
- PPO credit now applies only to inputs the game actually executed.
- GRU APPO receives real truncated BPTT and no sequence crosses an episode or
  task boundary.
- Larger, fixed-shape learner minibatches improve CUDA/compile utilization, and
  finite runs can force-flush a partial worker group.
- APPO still uses bounded-staleness clipped PPO, not V-trace. V-trace is the
  next distributed-algorithm experiment after live learner-lag metrics show it
  is needed.
- Flat upload format v3 still carries one tensor/GRU state. LSTM `(h, c)` upload
  support needs an explicit future format extension.

## Alternatives rejected

- **Add combination ids or enumerate the joint action space.** This duplicates
  primitives, increases sparsity, and caps unseen compositions.
- **Let a new action interrupt an active duration/macro.** The current Mod does
  not expose interruption semantics, so recorded PPO actions would remain
  misaligned.
- **Flatten recurrent APPO and rely on hidden-state snapshots.** Hidden state
  supplies context but does not train temporal credit assignment.
- **Adopt V-trace in the same change.** Correct discounts/bootstrap and
  truncation semantics should be validated separately from sequence recovery.
