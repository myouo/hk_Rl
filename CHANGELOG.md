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
- TcpTransport now supports an opt-in length-prefixed auth-token handshake frame
  before regular protocol traffic.
- SharedMemoryTransport now implements bounded ring send/recv semantics with
  connection lifecycle, timeout, reconnect, and capacity handling.
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
- HKRLEnv now rejects malformed observations with non-finite values, mismatched
  entity masks, entity hp above max hp, or missing boss entities in boss tasks.
- NormalizeObservation wrapper now scales player/entity positions, velocities,
  hp/soul, hitboxes, and timers using the shared space constants.
- ObservationTier wrapper now slices privileged observations down to reduced or
  human-visible feature sets and updates Gym observation spaces for ablations.
- JSONL metric sink for scalar and per-episode training/evaluation records.
- Shared `hkrl`/`scripts/train.py` smoke CLI for config-driven TCP env wiring,
  normalized observations, random policy actions, and JSONL metrics.
- Flat RolloutBuffer now stores fixed-capacity multi-env PPO transitions,
  computes GAE returns, exports RolloutBatch copies, and supports clear/reuse.
- RecurrentRolloutBuffer now stores hidden-state-aware rollouts, computes GAE,
  and emits padded, burn-in masked sequence batches split at episode boundaries.
- RolloutBatch now has pickle-free compressed NPZ save/load helpers for stable
  worker-to-learner spooling and integration-test boundaries.
- RecurrentPPO now runs clipped sequence updates with hidden-state inputs,
  burn-in/padding loss masks, and PPO training metrics.
- APPO now accepts bounded-staleness RolloutBatches, rejects stale/future policy
  versions, and runs queued PPO-style updates for remote learner intake.
- Mask-aware PyTorch hybrid policy/value heads now sample and evaluate packed
  training action tensors for the MLP/PPO baseline path.
- MLP actor-critic baseline now flattens global/player/entity observations with
  entity-mask padding suppression and exposes act/evaluate_actions for PPO.
- Phase 5 model encoders now embed global, player, and entity features with
  learned entity type/stable-id embeddings for attention/recurrent policies.
- Phase 5 entity attention modules now provide masked Transformer pooling and
  player cross-attention with safe all-padding handling.
- EntityAttentionRecurrentAC now assembles encoders, masked entity attention,
  GRU/LSTM memory, and hybrid policy/value heads for single-step and sequence
  forward/evaluate paths.
- Synchronous PPO now runs clipped policy/value updates over flat rollout
  batches, reports core training metrics, and exports advantages/returns in
  RolloutBatch.
- GameWorker can now collect local single-env rollouts with model inference,
  action-mask handling, tensor-to-env action conversion, and GAE bootstrap.
- GameWorker now hot-swaps verified checkpoint weights before rollouts and tags
  collected batches with the loaded learner policy version.
- GameWorker run loop now supports injectable rollout upload and heartbeat
  callbacks for learner/coordinator integration.
- GameWorker now recovers from transient env/transport failures with bounded
  retries, reconnect/reset, crash heartbeats, and `worker_crash_count` metrics.
- FrameStack wrapper now stacks dict observation feature axes and updates the
  Gym observation space for short-history MLP baselines.
- ScriptedAggroPolicy now provides a mask-aware heuristic baseline that approaches
  and attacks the nearest boss/entity from structured observations.
- Evaluator now runs fixed-seed task episodes through an injected eval env factory,
  aggregates shaping-free metrics, and reports win-rate regression deltas.
- TaskSampler now provides seeded weighted task sampling with mastered-task replay
  and win-rate based reweighting for anti-forgetting curricula.
- Curriculum now exposes active task stages and advances when all active tasks
  meet configured win-rate and episode-count gates.
- Coordinator now tracks worker registration, heartbeats, task assignment,
  lost-worker state, and heartbeat timeout expiry.
- Coordinator now ingests worker heartbeat payloads and exposes aggregate worker
  monitoring metrics for active/lost workers, SPS, assignments, and crashes.
- `scripts/run_eval.py` now runs fixed-seed evaluator jobs for scripted or MLP
  checkpoint policies and emits JSON metrics/regression output.
- `scripts/train.py`/`hkrl.cli` now run local MLP+PPO training updates with
  GameWorker rollouts, JSONL metrics, and optional checkpoint emission.
- CheckpointRegistry now persists versioned torch checkpoints with sha256 hashes
  and reloadable JSONL metadata for worker verification.
- CheckpointClient now pulls local/file registry checkpoints with sha256
  verification before loading weights for hot-swaps.
- LearnerServer now provides in-process RolloutBatch intake, APPO updates,
  policy-version accounting, and checkpoint publishing through the registry.
- `scripts/run_learner.py` now builds a config-driven learner model/server and
  emits a JSON startup summary for Phase 6 smoke checks.
- `scripts/run_worker.py` now builds config/task-driven worker wiring with a
  dry-run mode and optional checkpoint registry probing.
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
- Mod macro action scheduler now expands the PRD macro ids into deterministic
  primitive input plans consumed by ActionApplier.
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
- Mod observation path now returns global/player snapshots plus best-effort
  boss/projectile/hazard entity records with stable ids and threat scores.
- Mod debug overlay now renders a minimal toggleable HKRL status/SPS panel.
- Mod observation snapshots now carry structured EntityObservation records,
  stable entity id registry plumbing, snapshot-to-FlatBuffers encoding, and
  StepController observation collection for the Phase 4 entity-list path.
- Mod SnapshotRecorder now appends enabled snapshot JSONL lines to disk for
  replay/regression capture plumbing.
- Configs (`tasks/`, `train/`) and scripts (`gen_schema`, `train`,
  `run_worker`, `run_learner`, `run_eval`).

### Decisions
- ADR-0001 RL framework: self-built PyTorch PPO.
- ADR-0002 serialization: FlatBuffers single source of truth.
- ADR-0003 mod framework: HK Modding API.
- ADR-0004 local inference + remote training decoupled.
