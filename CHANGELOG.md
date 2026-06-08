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
- Flat RolloutBuffer now stores fixed-capacity multi-env PPO transitions,
  computes GAE returns, exports RolloutBatch copies, and supports clear/reuse.
- Mask-aware PyTorch hybrid policy/value heads now sample and evaluate packed
  training action tensors for the MLP/PPO baseline path.
- MLP actor-critic baseline now flattens global/player/entity observations with
  entity-mask padding suppression and exposes act/evaluate_actions for PPO.
- Synchronous PPO now runs clipped policy/value updates over flat rollout
  batches, reports core training metrics, and exports advantages/returns in
  RolloutBatch.
- GameWorker can now collect local single-env rollouts with model inference,
  action-mask handling, tensor-to-env action conversion, and GAE bootstrap.
- Mod-side protocol foundations: typed reward event buffering, heartbeat
  liveness tracking, StepRequest decode DTOs, and length-prefixed StepResponse
  encoding via generated FlatBuffers bindings.
- Mod TCP server now accepts one client at a time on a background thread,
  transfers uint32-LE length-prefixed frames through thread-safe queues, and
  keeps Unity access out of the network loop.
- Mod action mask layout now mirrors Python ordering and applies basic
  player-readiness rules for jump, dash, attack, cast, focus, and nail-art
  buttons.
- Mod primitive action path now maps decoded wire actions into movement/aim/button
  input state with button-bit clamping and duration hold bookkeeping.
- Mod episode lifecycle now advances through reset, running, termination, report,
  cleanup, tracks episode ids, and surfaces lifecycle error codes.
- Mod StepController now drains inbound requests, dispatches reset/step/task/ping
  commands, advances lifecycle, applies running actions, drains reward events,
  computes action masks, and enqueues StepResponse frames.
- HKRLEnvMod now starts the TCP server, wires StepController into the persistent
  FixedUpdate driver, and disposes transport resources on driver destruction.
- Mod SimControl now applies time scale, pause, and resume through Unity
  `Time.timeScale`/`fixedDeltaTime` on the main thread.
- Mod SceneController now maps known task ids to Godhome scenes and reports
  scene/player/boss readiness from Unity main-thread state.
- Mod ResetManager now starts task resets, polls scene/player/boss readiness with
  a timeout, and returns concrete reset failure status codes.
- Mod reward hook scaffolds now install a shared RewardEventBuffer and expose
  typed event recorders for damage, death, heal/soul, and scene-change events.
- Mod observation path now returns a minimal global/player snapshot and empty
  entity/mask lists without throwing, ready for later boss/entity enrichment.
- Mod debug overlay now renders a minimal toggleable HKRL status/SPS panel.
- Configs (`tasks/`, `train/`) and scripts (`gen_schema`, `train`,
  `run_worker`, `run_learner`, `run_eval`).

### Decisions
- ADR-0001 RL framework: self-built PyTorch PPO.
- ADR-0002 serialization: FlatBuffers single source of truth.
- ADR-0003 mod framework: HK Modding API.
- ADR-0004 local inference + remote training decoupled.
