# ADR-0010: Boundary-safe, versioned live tuning

- Status: Accepted
- Date: 2026-07-26

## Context

Long Godhome runs need reward and optimizer adjustments based on observed
behavior. Restarting both hosts for every small change wastes game rollout time
and makes operational experiments difficult. Applying a scalar reward or policy
loss coefficient during a rollout/update is worse: one batch can then contain
transitions from two objectives, while a remote control request in the local
action loop can stall input timing.

Some parameters are shape-neutral and safe between batches; others determine
buffer geometry, compiled graphs, action/observation layout, or model tensors.
The control surface must distinguish them explicitly and preserve the
local-inference invariant from ADR-0004.

## Decision

1. Add a narrow `LiveTuning` model containing only reward weights,
   `learning_rate`, `entropy_coef`, `value_coef`, `clip_range`,
   `max_grad_norm`, `target_kl`, and worker `time_scale`.
2. Treat every request as a complete snapshot with an exactly monotonic version.
   Store request and acknowledgement JSON atomically beside the checkpoint
   registry. Accept writes only through the registry's authenticated
   loopback-only `POST /live-tuning` endpoint.
3. The learner checks requests after uploads and idle intake timeouts. If an
   old-version partial queue exists, update it first under the old settings.
   Apply the new snapshot only when the queue is empty and immediately publish
   a checkpoint containing the snapshot and version.
4. Add `tuning_version` to RolloutBatch format v4. The learner accepts only its
   current version. Older v3 batches are rejected because their scalar rewards
   have no objective provenance.
5. Poll checkpoint metadata in a GameWorker background thread. Ordinary policy
   weights still change at a full rollout boundary. A newer tuning version may
   interrupt at the next local action boundary: discard the unfinished prefix,
   apply the verified checkpoint, clear recurrent context, and reset the arena.
   No network request occurs in the action loop.
6. Resolve every snapshot from immutable startup YAML values. `--unset` returns
   one field to startup while retaining the others; `--reset` returns all live
   fields. Embed the full snapshot in all subsequent checkpoints so restarts
   preserve it.
7. Keep geometry/layout settings restart-only, including rollout/minibatch/
   sequence sizes, epochs, aggregation, model dimensions, task enrollment,
   transport, observation/action layout, and `action_repeat`.

## Consequences

- Operators can change engagement rewards and safe PPO/APPO dynamics without
  restarting the game or learner.
- A request normally reaches the learner within the intake timeout and the
  worker within its background polling interval. At most one unfinished
  rollout fragment is discarded for an objective change.
- Checkpoints, heartbeats, rollout files, and acknowledgements expose the same
  tuning version for audit and recovery.
- Structural experiments still require a controlled restart and a composed
  configuration change.
- Material reward changes remain separate evaluation segments even though the
  process itself continues; shaping-free per-Boss evaluation remains the source
  of capability truth.

## Alternatives rejected

- **Rewrite YAML and send a signal.** This has no typed allowlist, version
  contract, atomic acknowledgement, or cross-host recovery.
- **Apply values on every environment step.** This mixes objectives inside one
  rollout and can leak recurrent state across definitions.
- **Poll the remote registry synchronously from the action loop.** Network
  latency would reintroduce the timing problem forbidden by ADR-0004.
- **Hot-reload all TrainConfig fields.** Shape and sampling-geometry changes
  invalidate buffers, compiled graphs, and potentially checkpoints.
