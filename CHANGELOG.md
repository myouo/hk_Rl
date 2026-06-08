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
- Python TCP entry points now honor `security.require_token` by reading
  `security.auth_token_env`, and the mod TCP server consumes/verifies the auth
  frame before forwarding protocol requests.
- `scripts/run_eval.py` now uses the same security token config for evaluator TCP
  env connections.
- SharedMemoryTransport now implements bounded ring send/recv semantics with
  connection lifecycle, timeout, reconnect, and capacity handling.
- Python train/worker/evaluator entry points now construct TCP or shared-memory
  env transports through a config-driven transport factory.
- Python FlatBuffers StepRequest/StepResponse encode/decode helpers with schema
  version checks, action payload conversion, reward events, masks, and decoded
  observation views.
- Gymnasium action/observation space construction for hybrid actions,
  entity-list observations, ablation tiers, and action-mask layout tests.
- Task action configs now expose `n_macro_actions` so Python action spaces,
  policy heads, and scripts share the same macro/action-mask width.
- Mask-aware random policy for local smoke tests and scripted baseline plumbing.
- HKRLEnv construction now wires task-driven Gymnasium spaces and idempotent
  transport close behavior.
- HKRLEnv reset/step now drive the Transport protocol, poll the clean lifecycle
  until `RUNNING`, compose reward events, enforce tick echoes, and adapt decoded
  observations to Gymnasium spaces.
- HKRLEnv now exposes `pause()`, `resume()`, `ping()`, and
  `set_timescale(scale)` helpers for protocol-level control commands.
- HKRLEnv now exposes `set_task(task)` to send `SET_TASK`, rebuild task-driven
  spaces/default reward weights, and wait for the clean lifecycle to reach
  `RUNNING`.
- Task configs now separate human-readable `task_id` from numeric `wire_id`;
  HKRLEnv sends `wire_id` in every StepRequest and exposes it in Gym info for
  rollout task_ids.
- HKRLEnv now rejects malformed observations with non-finite values, mismatched
  entity masks, entity hp above max hp, or missing boss entities in boss tasks.
- NormalizeObservation wrapper now scales player/entity positions, velocities,
  hp/soul, hitboxes, and timers using the shared space constants.
- ObservationTier wrapper now slices privileged observations down to reduced or
  human-visible feature sets and updates Gym observation spaces for ablations.
- JSONL metric sink for scalar and per-episode training/evaluation records.
- CSV metric sink with a stable scalar/episode envelope and optional custom
  fieldnames for fixed wide exports.
- Stdout metric sink for JSON-line scalar and episode records.
- Shared `hkrl`/`scripts/train.py` smoke CLI for config-driven TCP env wiring,
  normalized observations, random policy actions, and JSONL metrics.
- `hkrl`/`scripts/train.py` now expose `--metrics-kind {jsonl,csv}` so smoke and
  local PPO runs can write either sink backend.
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
- APPO now consumes recurrent rollout `rnn_states` from uploaded flat batches so
  GRU policies train from the same hidden-state context used during worker
  collection.
- RolloutBatch now supports in-memory NPZ serialization and an authenticated TCP
  batch intake path from `run_worker --learner` to `run_learner --intake-count`.
- Checkpoint registries now publish relative checkpoint paths, and worker
  CheckpointClient can pull hash-verified checkpoints from local/file or HTTP(S)
  registry endpoints.
- `run_worker.py` can now append coordinator-compatible heartbeat JSONL for
  offline/fallback monitoring snapshots.
- Evaluator and `run_eval.py` can now emit per-step replay JSONL for fixed-seed
  evaluation debugging.
- `run_coordinator.py` can now ingest evaluator metrics JSON and apply per-task
  win rates to TaskSampler weights/mastered-task tracking.
- `run_learner.py` now supports `--serve-forever` for long-running authenticated
  TCP rollout intake with update/publish after each accepted batch.
- Test coverage now checks Python action-mask/button layout against the C# mod
  `ActionMasker`/`InputInjector` constants to catch cross-language drift.
- Recurrent rollout sequence chunking now splits at both terminated and truncated
  episode boundaries so hidden state does not leak across resets.
- Recurrent flat RolloutBatch export now rejects LSTM tuple states explicitly
  instead of silently dropping them on the APPO upload path.
- Canonical metric definitions now include per-boss evaluator metrics, and
  EpisodeStats records heal amount alongside heal count.
- Evaluator task metrics now emit `per_boss_win_rate` and
  `per_boss_damage_ratio` directly for dashboard consumers.
- `run_coordinator.py --eval-metrics` now accepts `per_boss_win_rate` as a
  sampler-weight fallback when `win_rate` is absent.
- Evaluator regression reports now accept `per_boss_win_rate` as a fallback for
  catastrophic-forgetting win-rate deltas.
- Mod StepRequest decode failures now report schema-version drift as
  `StatusCode.SchemaMismatch` instead of a generic internal error.
- Mod STEP handling now honors `action_repeat` across FixedUpdate ticks and
  returns early if a terminal reward event occurs during the repeat window.
- HKRLEnv now computes step reward `dt` from `server_tick` deltas so early
  terminal responses do not overcharge time penalties.
- HKRLEnv now rejects action masks whose length does not match the current task's
  canonical action-mask layout.
- Mod STEP dispatch now emits `InvalidAction` reward events for wire-level
  out-of-range action components before applying safe clamped input.
- Mod StepResponse action masks now use the observed player readiness fields
  instead of always returning the all-valid fallback mask.
- Mod PlayerObserver now reads hp/soul from PlayerData via reflection with safe
  fallbacks for minor game/API field-name drift.
- Mod PlayerObserver caches PlayerData type lookup so per-tick observation reads
  do not repeatedly scan loaded assemblies.
- Local PPO/RecurrentPPO training now advances the GameWorker policy version
  after each update so subsequent rollouts are tagged with the active weights.
- RolloutBatch NPZ deserialization now rejects fields with inconsistent
  `(time, env)` prefixes before they reach learner updates.
- RolloutBatch NPZ deserialization now enforces 4D recurrent-state payloads
  shaped `(time, layers, envs, hidden)`.
- RolloutBatch NPZ format v2 now carries explicit `prev_rewards`; recurrent
  policy training/evaluation receives both `prev_action` and `prev_reward`
  memory context.
- Checkpoint registry/client parsing now rejects empty checkpoint paths before
  filesystem or HTTP reads.
- GameWorker heartbeats and run summaries now report rollout duration and SPS
  directly for coordinator monitoring snapshots.
- run_worker summaries now split learner batch uploads into submitted,
  accepted, and rejected counts.
- BatchIntakeClient now rejects malformed success ACKs that omit the explicit
  `accepted` boolean instead of treating them as stale rejections.
- TCP batch intake now uses envelope type `hkrl.rollout_batch.v2`, matching the
  RolloutBatch NPZ v2 payload contract.
- Train/task config loading now rejects invalid numeric ranges such as
  zero rollout steps, out-of-range action repeats, and invalid service ports.
- Train/task config loading now rejects empty required strings such as task ids,
  scene names, model names, bind addresses, and checkpoint directories.
- Mod reward hooks now wrap event recording in try/catch and log failures through
  `Debug.Logger`, protecting the Unity main thread from hook exceptions.
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
- GameWorker now supports an injectable task provider and switches tasks through
  env `set_task()` before rollouts for curriculum/coordinator assignments.
- GameWorker recovery now finds reconnect hooks through Gym wrapper chains, so
  normalized/wrapped HKRLEnv instances still reconnect their underlying transport.
- GameWorker now forces a clean reset before the next rollout when an episode
  terminates exactly on the final collected step.
- FrameStack wrapper now stacks dict observation feature axes and updates the
  Gym observation space for short-history MLP baselines.
- ScriptedAggroPolicy now provides a mask-aware heuristic baseline that approaches
  and attacks the nearest boss/entity from structured observations.
- Evaluator now runs fixed-seed task episodes through an injected eval env factory,
  aggregates shaping-free metrics, and reports win-rate regression deltas.
- Evaluator now aggregates heal count/amount, death rate, and death reason from
  reward events alongside win/damage/invalid-action metrics.
- Canonical metric keys now include heal amount, death rate/reason, and
  time-to-kill for evaluator/dashboard consumers.
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
- `scripts/run_eval.py` can now load MLP policies from a CheckpointRegistry
  directory by selecting the latest indexed checkpoint.
- `scripts/run_eval.py` can now write evaluator metrics/regression JSON to an
  output path while still printing the summary to stdout.
- `scripts/run_eval.py` output now includes reproducibility metadata for policy,
  tasks, seeds, train config, transport, and checkpoint selection.
- `scripts/run_eval.py --baseline` now accepts either raw per-task metrics or a
  previous evaluator output object containing a top-level `metrics` field.
- `scripts/run_eval.py --policy model` now evaluates registry-configured
  ActorCritic checkpoints, including recurrent policies with state carried
  across episode steps.
- `scripts/run_eval.py` now sha256-verifies CheckpointRegistry entries before
  loading evaluator policy weights.
- `scripts/run_eval.py` now rejects model-policy task sets with incompatible
  observation/action layouts before evaluation starts.
- `scripts/train.py`/`hkrl.cli` now run local MLP+PPO training updates with
  GameWorker rollouts, JSONL metrics, and optional checkpoint emission.
- `scripts/train.py`/`hkrl.cli` now run local recurrent PPO training with
  GameWorker-collected RNN states and sequence/burn-in minibatches.
- Local MLP+PPO checkpoints are now published through `CheckpointRegistry` with
  `policy_version`, step metadata, index records, and sha256 verification data.
- CheckpointRegistry now persists versioned torch checkpoints with sha256 hashes
  and reloadable JSONL metadata for worker verification.
- CheckpointRegistry now rejects index entries whose checkpoint paths escape the
  registry root.
- CheckpointClient now pulls local/file registry checkpoints with sha256
  verification before loading weights for hot-swaps.
- CheckpointClient now rejects registry entries whose checkpoint paths escape the
  registry root.
- CheckpointClient now rejects duplicate checkpoint versions in registry indexes.
- LearnerServer now provides in-process RolloutBatch intake, APPO updates,
  policy-version accounting, and checkpoint publishing through the registry.
- `scripts/run_learner.py` now builds a config-driven learner model/server and
  emits a JSON startup summary for Phase 6 smoke checks.
- `TrainConfig` now preserves typed `learner`, `coordinator`, and `security`
  runtime settings, and `scripts/run_learner.py` consumes YAML learner defaults
  unless CLI flags explicitly override them.
- Learner and coordinator entry points now validate service bind addresses
  against `security.bind_scope` before startup.
- Config models now reject unknown keys and invalid enum values to prevent
  silent YAML misconfiguration.
- `scripts/run_learner.py` can now ingest NPZ RolloutBatch directories and run
  the submitted batches through learner update/checkpoint publication.
- `scripts/run_learner.py` can now infer learner model/action layout from task
  YAMLs and rejects multi-task sets with incompatible observation/action widths.
- `scripts/run_worker.py` and `scripts/run_learner.py` now treat MLP
  `model.rnn_hidden: 0` as the default hidden width instead of constructing an
  invalid zero-width network.
- `scripts/run_worker.py` now builds config/task-driven worker wiring with a
  dry-run mode, optional checkpoint registry probing, NPZ batch spooling, and
  configurable recovery limits.
- `scripts/run_worker.py` now accepts `--tasks` and installs a round-robin task
  provider for multi-task rollout smoke/curriculum runs.
- `scripts/run_coordinator.py` now validates coordinator/task/worker wiring,
  emits one-shot task assignments, ingests heartbeat JSONL, and reports a JSON
  monitoring snapshot for Phase 8 smoke checks.
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
- Mod StepController now gates lifecycle wait states through ResetManager scene /
  player / boss readiness polling and returns reset failure StatusCodes instead
  of advancing to RUNNING unconditionally.
- Mod StepController now routes terminal reward events into lifecycle
  termination and suppresses non-running reward events to reduce reset
  contamination.
- Mod StepController now drains inbound requests, dispatches reset/step/task/ping
  commands, advances lifecycle, applies running actions, drains reward events,
  computes action masks, and enqueues StepResponse frames.
- HKRLEnvMod now starts the TCP server, wires StepController into the persistent
  FixedUpdate driver, and disposes transport resources on driver destruction.
- Mod SimControl now applies time scale, pause, and resume through Unity
  `Time.timeScale`/`fixedDeltaTime` on the main thread.
- Mod StepController now dispatches `PAUSE`, `RESUME`, and `SET_TIMESCALE`
  commands through SimControl and reports command dispatch failures via
  `StatusCode.InternalError`.
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
