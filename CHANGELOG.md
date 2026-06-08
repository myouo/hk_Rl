# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
the project version tracks the **schema_version** + roadmap phase.

## [Unreleased]

### Added
- Initial architecture scaffold: full directory tree, interface-level
  placeholders, specification docs, ADRs, and `AGENTS.md`.
- FlatBuffers schema `schema/hkrl.fbs` as the single source of truth for
  observation / action / protocol (schema_version 1).
- Python package `hkrl` skeleton: transport, env, spaces, reward, models
  (entity-attention + recurrent), training (PPO / RecurrentPPO / APPO),
  worker / learner / coordinator / eval / utils.
- C# mod skeleton `HKRLEnvMod` (HK Modding API) covering Transport / Env /
  Observation / Action / Rewards / Debug.
- Phase 0 mod bootstrap: Modding API-backed logger, persistent FixedUpdate
  driver, and periodic scene/player-position snapshot logging.
- Conda development environment (`environment.yml`) for Python runtime, dev,
  logging, and distributed extras.
- Local/CI quality gates generate FlatBuffers bindings before running checks,
  with generated code ignored and excluded from lint/typecheck.
- Python training foundations: config `defaults` composition, default reward
  composition, deterministic seeding, running metrics, and GAE.
- Python TCP transport client with uint32-LE length-prefixed framing,
  timeout/disconnect handling, and localhost framing tests.
- Python FlatBuffers StepRequest/StepResponse encode/decode helpers with schema
  version checks, action payload conversion, reward events, masks, and decoded
  observation views.
- Gymnasium action/observation space construction for hybrid actions,
  entity-list observations, ablation tiers, and action-mask layout tests.
- Mask-aware random policy for local smoke tests and scripted baseline plumbing.
- HKRLEnv construction now wires task-driven Gymnasium spaces and idempotent
  transport close behavior.
- HKRLEnv reset/step now drive the Transport protocol, poll the clean lifecycle
  until `RUNNING`, compose reward events, enforce tick echoes, and adapt decoded
  observations to Gymnasium spaces.
- NormalizeObservation wrapper now scales player/entity positions, velocities,
  hp/soul, hitboxes, and timers using the shared space constants.
- JSONL metric sink for scalar and per-episode training/evaluation records.
- Shared `hkrl`/`scripts/train.py` smoke CLI for config-driven TCP env wiring,
  normalized observations, random policy actions, and JSONL metrics.
- Mod-side protocol foundations: typed reward event buffering, heartbeat
  liveness tracking, StepRequest decode DTOs, and length-prefixed StepResponse
  encoding via generated FlatBuffers bindings.
- Mod TCP server now accepts one client at a time on a background thread,
  transfers uint32-LE length-prefixed frames through thread-safe queues, and
  keeps Unity access out of the network loop.
- Mod action mask layout now mirrors Python ordering and applies basic
  player-readiness rules for jump, dash, attack, cast, focus, and nail-art
  buttons.
- Configs (`tasks/`, `train/`) and scripts (`gen_schema`, `train`,
  `run_worker`, `run_learner`, `run_eval`).

### Decisions
- ADR-0001 RL framework: self-built PyTorch PPO.
- ADR-0002 serialization: FlatBuffers single source of truth.
- ADR-0003 mod framework: HK Modding API.
- ADR-0004 local inference + remote training decoupled.
