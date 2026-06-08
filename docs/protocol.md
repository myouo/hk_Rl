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
| `STEP` | apply `action` (`action_repeat` times), return next obs + reward events |
| `RESET` | begin clean episode lifecycle; response stream reports `lifecycle_state` until `RUNNING` |
| `PAUSE` / `RESUME` | freeze / unfreeze the sim |
| `SET_TASK` | switch boss/arena (`task_id`); triggers a reset |
| `SET_TIMESCALE` | set `time_scale` for SPS tuning |
| `PING` | heartbeat; response with `server_tick` only |

## 4. Reset handshake (ack)

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

## 5. Liveness & recovery

- **Heartbeat:** `PING` at a fixed interval; missed N → worker marks the env
  unhealthy and attempts reconnect.
- **Timeout:** every request has a deadline; on timeout the worker reconnects and
  forces a `RESET`.
- **Reconnect:** the mod's network thread accepts a new connection and resets the
  env; `env_id` + `episode_id` disambiguate stale buffers.

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
