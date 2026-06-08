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
`task_ids` are the numeric task `wire_id` values from task YAML, not the
human-readable task names used in evaluator output.

`hkrl/training/batch_io.py` serializes the same bundle as a compressed,
pickle-free NPZ file for local spooling, crash recovery, and worker/learner
integration tests. Network transports for batches should preserve this field
contract even if they use a different envelope.

For filesystem-based smoke runs, `scripts/run_worker.py --batch-dir DIR` writes
each completed rollout batch to NPZ, and `scripts/run_learner.py --batch-dir DIR`
loads all `*.npz` batches through `LearnerServer.submit()` before serving one
update cycle.

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

Runtime settings under `learner:`, `coordinator:`, and `security:` in
`configs/train/remote_learner.yaml` are typed in `hkrl.utils.config.TrainConfig`.
CLI flags on `scripts/run_learner.py` override those YAML values only when
explicitly provided.
Unknown config keys are rejected instead of ignored, so typos in distributed
settings fail during startup.

When `security.require_token` is true, Python TCP clients read the token from
`security.auth_token_env` (default `HKRL_AUTH_TOKEN`) and send it as the initial
auth frame. The mod reads the same `HKRL_AUTH_TOKEN` environment variable to
enable server-side token verification.

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

For curricula, `GameWorker` can accept a `task_provider` callback. Before each
rollout it asks for the assigned `TaskConfig`; if the task wire id changed, it
calls the env's `set_task()` through any Gym wrapper chain and starts the rollout
from the returned clean-reset observation.

## 8. Monitoring snapshot

Coordinator ingests raw worker heartbeat payloads, splits numeric metrics from
status fields, and exposes a `metrics_snapshot()` aggregate for dashboards and
JSONL logging. Fleet SPS is the sum across active workers; lost workers still
count toward `worker_crash_count` so crash/restart churn remains visible.

## 9. PyTorch + CUDA note

`torch` is intentionally unpinned to a specific CUDA build in `pyproject.toml`.
Install the matching wheel per machine (GPU learner vs CPU-only worker). Workers
can run inference CPU-only (PRD Phase 6 milestone).
