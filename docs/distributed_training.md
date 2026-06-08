# Distributed Training

> Implements PRD §8 + §9.5. Code: `hkrl/worker/`, `hkrl/learner/`,
> `hkrl/coordinator/`. Decision: [ADR-0004](./adr/0004-local-inference-remote-training.md).

## 1. Local inference, remote training

```text
GameWorker (Game PC):                 Learner (Remote GPU):
  load latest policy                    receive rollout batches
  loop:                                 filter by policy_version
    action = local_policy(obs, hidden)  PPO/APPO/IMPALA update
    obs, r, done, info = env.step()     publish checkpoint
    buffer.add(...)
    if full: upload RolloutBatch  ───▶
    if new checkpoint: load weights ◀───
```

The action loop is **local and synchronous**. The remote link carries only
rollout batches (up) and checkpoints (down), both asynchronous and batched.

## 2. Why not remote real-time inference (PRD §8.2)

`Game PC obs → Remote GPU → action → Game PC` is rejected: network latency and
jitter corrupt action timing; action games need a stable tick; single-sample GPU
inference isn't necessarily faster than local. Remote inference only pays off for
batched multi-env inference, which we don't need here.

## 3. RolloutBatch (PRD §8.3)

```text
obs_global, obs_player, obs_entities, entity_mask,
actions, log_probs, values, advantages, returns,
rewards, dones, truncateds, action_masks, prev_actions, rnn_states,
episode_ids, task_ids, policy_version
```

Defined in `hkrl/training/rollout_buffer.py` (+ recurrent variant). Sequences for
recurrent training preserve `rnn_states` at sequence boundaries and mask padded
timesteps.

## 4. On-policy staleness (PRD §9.5)

PPO is sensitive to stale data. Mitigations:

- Every batch carries `policy_version`.
- **Synchronous PPO:** collect at a fixed version, update together.
- **Async (APPO/IMPALA):** allow bounded staleness; down-weight or drop batches
  older than a threshold (V-trace / importance correction).

## 5. Components

- **`worker/game_worker.py`** — env loop, local policy, rollout buffer, uploader,
  checkpoint client, crash/reconnect handling, local metrics.
- **`worker/checkpoint_client.py`** — poll/pull + hash-verify checkpoints.
- **`learner/learner_server.py`** — batch intake, version filtering, optimizer,
  checkpoint publish, training metrics.
- **`learner/checkpoint_registry.py`** — versioned, hash-signed checkpoint store.
- **`coordinator/coordinator.py`** — worker registry, task assignment, train/eval
  isolation, metric aggregation, failure recovery.
- **`coordinator/task_sampler.py`** + **`curriculum.py`** — balanced/curriculum
  task sampling (PRD §7, §9.7).

## 6. Transport for batches/weights

Distinct from the env transport. TCP/gRPC/ZeroMQ across machines; Ray actors are
a future option for the coordinator/worker fabric. Security per PRD §9.10: LAN/
localhost only, token auth, hash-verified checkpoints, command whitelist.

## 7. Worker recovery

`GameWorker.run()` treats transient env/transport failures as recoverable up to a
bounded consecutive-failure limit. On failure it clears partial rollout state,
resets RNN hidden state, emits a heartbeat with `status = recovering` and
`worker_crash_count`, calls the env transport's `reconnect()` when available, and
then forces a clean `reset()` before collecting the next batch. Persistent
failures surface as errors instead of spinning forever.

## 8. PyTorch + CUDA note

`torch` is intentionally unpinned to a specific CUDA build in `pyproject.toml`.
Install the matching wheel per machine (GPU learner vs CPU-only worker). Workers
can run inference CPU-only (PRD Phase 6 milestone).
