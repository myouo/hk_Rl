# Transport Protocol

> Implements PRD §5.4. Message bodies are defined in
> [`../schema/hkrl.fbs`](../schema/hkrl.fbs) (the single source of truth).

## 1. Framing

Each message is a **length-prefixed FlatBuffers buffer**:

```text
+----------------+--------------------------+
| uint32 LE len  |  FlatBuffers payload     |
+----------------+--------------------------+
```

`len` is the byte length of the payload. FlatBuffers buffers carry the
`file_identifier "HKRL"`; receivers verify it before reading. Shared-memory
transport uses the same payload inside a ring-buffer slot (length in the slot
header) — see [ADR-0002](./adr/0002-serialization-flatbuffers.md).

## 2. Messages

`StepRequest` (worker → mod) and `StepResponse` (mod → worker). Full field lists
in the schema. Highlights:

- Every message carries `schema_version`. On mismatch, the mod replies with
  `StatusCode.SchemaMismatch` and the worker aborts (no silent drift).
- Every step carries `tick_id`; the response echoes it so the worker can match
  request/response and measure latency via `client_time`.
- `policy_version` rides on each request so the learner can filter stale
  rollouts (PRD §9.5).

## 3. Commands

| Command | Meaning |
|---|---|
| `STEP` | apply `action` (`action_repeat` times) only while `RUNNING`, return next obs + reward events |
| `RESET` | begin clean episode lifecycle; response stream reports `lifecycle_state` until `RUNNING` |
| `PAUSE` / `RESUME` | freeze / unfreeze the sim |
| `SET_TASK` | switch boss/arena (`task_id`); triggers a reset |
| `SET_TIMESCALE` | set `time_scale` for SPS tuning |
| `PING` | heartbeat; response with `server_tick` only |

`PAUSE`, `RESUME`, and `SET_TIMESCALE` are applied by mod `SimControl` on the
Unity main thread. Invalid command parameters return `StatusCode.InternalError`
in the response instead of throwing through `FixedUpdate`.
Python exposes these through `HKRLEnv.pause()`, `HKRLEnv.resume()`,
`HKRLEnv.ping()`, and `HKRLEnv.set_timescale(scale)`.
For `STEP`, the mod delays the `StepResponse` until all repeated FixedUpdate
ticks have been applied, or until a terminal reward event ends the episode early.
The Python env computes reward time deltas from consecutive `server_tick`
values, so early terminal responses do not overcharge the configured repeat
count. Before `RUNNING`, the mod accepts only the canonical no-op `STEP` used as
a reset poll (`movement=neutral`, `aim=neutral`, no buttons, `duration=0`,
`macro_id=-1`, `action_repeat=1`); any other `STEP` returns
`StatusCode.NotRunning`.

## 4. Reset handshake (ack)

Task configs use a human-readable string `task_id` for logs/evaluation and a
numeric `wire_id` for `StepRequest.task_id`. The mod maps the numeric id to
Godhome scenes; rollout buffers store the same numeric id in `task_ids`.
Python exposes task switching via `HKRLEnv.set_task(task)`, which sends
`SET_TASK`, rebuilds task-driven spaces/reward defaults, and waits for the clean
reset lifecycle to reach `RUNNING`. Unknown numeric task ids are not mapped to a
fallback arena; reset readiness fails instead of silently training on the wrong
boss.

`RESET` is **not** a single round-trip. The mod walks the lifecycle state
machine ([`episode_lifecycle.md`](./episode_lifecycle.md)) and reports progress
via `StepResponse.lifecycle_state`. The worker may only send `STEP` once it sees
`LifecycleState.Running`. A failed reset returns a non-`Ok` `error_code` instead
of silently continuing (PRD §9.3).

```text
worker: RESET ─────────────────▶ mod
worker: STEP(noop) ────────────▶ mod  (poll) ──▶ lifecycle_state = WaitBossReady
worker: STEP(noop) ────────────▶ mod  (poll) ──▶ lifecycle_state = Countdown
worker: STEP(noop) ────────────▶ mod  (poll) ──▶ lifecycle_state = Running  ✅ env ready
```

The poll `STEP` above is the only pre-`RUNNING` step allowed. It advances the
lifecycle but does not apply input.

## 5. Liveness & recovery

- **Heartbeat:** `PING` at a fixed interval; missed N → worker marks the env
  unhealthy and attempts reconnect.
- **Timeout:** every request has a deadline; on timeout the worker reconnects and
  forces a `RESET`.
- **Reconnect:** the mod's network thread accepts a new connection; the Python
  worker/env then forces a `RESET` on the main-thread lifecycle path. `env_id`
  + `episode_id` disambiguate stale buffers.
  The TCP server detects half-closed clients and clears per-connection request /
  response queues before accepting the next client, so stale responses from a
  dead worker cannot be delivered to a reconnecting worker or evaluator.

## 6. Threading contract (mod side)

The network thread MUST NOT touch Unity objects. It only enqueues requests and
dequeues responses. All game access happens on the main thread in `FixedUpdate`.
See PRD §5.3 and [`mod_dev.md`](./mod_dev.md).

```text
NetworkThread:  recv StepRequest -> enqueue
MainThread (FixedUpdate): dequeue latest -> apply action -> collect obs/events
                          -> write StepResponse to out-queue
NetworkThread:  dequeue StepResponse -> send
```

## 7. Versioning

`SCHEMA_VERSION` is mirrored in `python/hkrl/protocol.py` and
`mod/HKRLEnvMod/Transport/Protocol.cs`. Bump on any schema change; evolve the
schema append-only (see [`../schema/README.md`](../schema/README.md)).

## 8. Security (PRD §9.10)

Bind to `localhost` or LAN only; never expose a public port. Optional token auth
uses an initial length-prefixed `HKRL_AUTH\0<token>` frame before any
FlatBuffers `StepRequest`; the mod consumes that frame on the network thread and
never forwards it to the StepController. Workers execute only whitelisted
commands. Checkpoints are hash-verified before load.
