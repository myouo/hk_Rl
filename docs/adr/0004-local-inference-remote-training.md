# ADR-0004: Local inference + remote training (decoupled)

- Status: **Accepted**
- Date: 2026-06-08

## Context

Training benefits from a remote GPU, but Hollow Knight is a real-time action game
where action timing is critical. Where should inference run relative to the game?

## Decision

Run the **real-time action loop entirely locally** on the Game PC
(`obs → local_policy(obs, hidden) → action → game`). The remote GPU does only
**large-batch training**. The link carries rollout batches up and checkpoints
down — asynchronous and batched, never in the action loop.

## Rationale

- Network latency/jitter on the action path would destabilize a fixed-tick action
  game; the same policy would perform inconsistently.
- Single-sample remote GPU inference isn't reliably faster than local; remote
  inference only pays off for batched multi-env inference, which we don't need.
- Decoupling lets a no-GPU Game PC still run inference (PRD Phase 6 milestone),
  while the GPU stays saturated with batch updates.

## Consequences

- Two transports: the per-tick env transport (FlatBuffers, [ADR-0002](./0002-serialization-flatbuffers.md))
  and the batch/weight transport (TCP/gRPC/ZeroMQ, async).
- PPO staleness handled via `policy_version` (sync PPO, or APPO/IMPALA for async).
- Checkpoints are hash-verified before a worker loads them (security, PRD §9.10).

## Alternatives rejected

- **Remote real-time inference** (`Game PC → GPU → action → Game PC`) — rejected:
  latency/jitter corrupt action timing; no throughput benefit at single-env scale.
