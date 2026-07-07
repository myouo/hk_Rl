# Architecture

> Implements PRD §4. Read alongside [`protocol.md`](./protocol.md) and the
> [ADRs](./adr/).

## 1. System view

```text
+------------------------------------------------------------------+
|                        Remote GPU Server                          |
|   Learner (PPO/APPO/IMPALA)  <-->  Checkpoint / Policy Registry    |
|        ^                                  ^                        |
|        | rollout batches                  | weights               |
|        v                                  v                        |
|   Coordinator  <-->  Metrics / Evaluator / Logger                 |
+----------------------------------^-------------------------------+
                                   | TCP / gRPC / ZeroMQ (batches+weights, async)
+----------------------------------v-------------------------------+
|                            Game PC                                |
|   GameWorker (local inference, Gym Env)  <-->  HKRLEnvMod (C#)     |
|        obs -> local policy -> action -> game   (FlatBuffers/TCP)    |
+------------------------------------------------------------------+
```

The **real-time loop is entirely on the Game PC**. The remote GPU only does
large-batch training. This is the single most important structural decision —
see [ADR-0004](./adr/0004-local-inference-remote-training.md).

## 2. Six invariants

These hold across the whole codebase. Violating one is a design regression.

1. **Local action loop never crosses the remote network.**
   `obs → local_policy(obs, hidden) → action → game` runs at a stable tick on
   the Game PC. Latency/jitter from a remote hop would corrupt action timing in
   an action game.
2. **`schema/hkrl.fbs` is the single source of truth.** C# and Python both
   consume generated bindings. Change the schema, never the bindings. See
   [`../schema/README.md`](../schema/README.md), [ADR-0002](./adr/0002-serialization-flatbuffers.md).
3. **Transport is pluggable.** Everything talks through the `Transport`
   interface; TCP is the live HKRLEnvMod transport today. Shared-memory remains
   an in-process Python prototype behind an explicit opt-in until the mod ships
   a real OS shared-memory server. Env/model code still depends only on the
   transport interface, and Python entry points construct transports through
   `hkrl.transport.factory` from YAML config rather than instantiating TCP
   directly.
4. **Config-driven + component registry.** Models, transports, reward functions,
   tasks register by name (`hkrl.utils.registry`) and are selected from YAML
   (`hkrl.utils.config`). Adding a boss or a model variant requires no core edits.
5. **Clean episode lifecycle.** A state machine + reset-ack + `episode_id` +
   event-buffer-clear guarantees no cross-episode reward contamination. See
   [`episode_lifecycle.md`](./episode_lifecycle.md).
6. **Model is decomposed and mask-aware.** `encoders → attention → memory →
   heads` with `entity_mask` threaded throughout; training path reserves
   `torch.compile` + AMP + truncated BPTT. See [`model_architecture.md`](./model_architecture.md).

## 3. Component responsibilities

| Component | Process | Responsibility |
|---|---|---|
| **HKRLEnvMod** | inside Hollow Knight (C#) | observation collect, action apply on `FixedUpdate`, reward-event hooks, reset lifecycle, time-scale control, action mask, never block the Unity main thread |
| **Transport** | both ends | framed FlatBuffers messages; heartbeat, reset ack, timeout/reconnect, version + policy_version negotiation |
| **GameWorker** | Game PC | local inference, Gym wrapper, rollout buffer, upload batches, pull checkpoints, crash/reconnect handling, local metrics |
| **Learner** | Remote GPU | collect rollout batches, filter by policy_version, PPO/APPO update, publish checkpoints, training metrics |
| **Coordinator** | Remote GPU | manage workers, assign tasks, curriculum sampling, checkpoint registry, train/eval isolation, metric aggregation |
| **Evaluator** | Remote GPU | fixed-seed per-boss evaluation, replay capture, catastrophic-forgetting detection |

## 4. Data flow (one training tick)

```text
GameWorker                         HKRLEnvMod (main thread)
  build StepRequest(action)  ──TCP──────▶  enqueue (network thread)
                                           FixedUpdate: dequeue latest action,
                                             apply, collect obs+reward events,
                                             write StepResponse to out-queue
  decode StepResponse        ◀──TCP──────  send (network thread)
  reward = reward_fn(events)
  buffer.add(obs, action, reward, value, logprob, hidden, mask)
  if buffer full: upload RolloutBatch ──▶ Learner
  if new checkpoint: load weights ◀────── Checkpoint Registry
```

The reward **scalar** is computed Python-side from mod-reported **events**
(decoupled — [`reward_design.md`](./reward_design.md)). The mod never hard-codes
final reward.

## 5. Performance posture

- Metric is **SPS (samples/sec)**, not FPS. See [`metrics.md`](./metrics.md) and
  PRD §9.6. Levers: `Time.timeScale` / `fixedDeltaTime`, `action_repeat`, reduced
  render quality, parallel instances, fast reset.
- Hot-path decode is zero-copy (FlatBuffers); `info` JSON is debug-only, never on
  the hot path.
- TCP is the supported live transport. Local shared-memory remains the planned
  high-SPS path for single-machine runs after the mod-side SHM server lands.
- Training: large-batch on GPU, `torch.compile` + mixed precision, sequence
  (truncated-BPTT) batching for recurrent policies.

## 6. Extensibility seams

| Want to add… | Touch only… |
|---|---|
| a new boss/task | `configs/tasks/*.yaml` (+ mod scene mapping) |
| a new model | `hkrl/models/*` + `@register_model` + train config |
| a new transport | `hkrl/transport/*` implementing `Transport` + `@register_transport` |
| a new reward term | `hkrl/reward.py` (event → scalar), config weights |
| a new RL algo | `hkrl/training/*` implementing the learner interface |

## 7. Roadmap → architecture mapping

See PRD §10 for the phase-by-phase plan and [`AGENTS.md`](../AGENTS.md#roadmap)
for current status. P0 surface = step/reset protocol, Gym env, clean lifecycle,
action mask, reward events, single-boss baseline.
