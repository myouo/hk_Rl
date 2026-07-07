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
- CI now uses the Node 24-backed `actions/checkout@v6` action and the current
  setup-miniconda `auto-activate` input.
- Python training foundations: config `defaults` composition, default reward
  composition, deterministic seeding, running metrics, and GAE.
- Python TCP transport client with uint32-LE length-prefixed framing,
  timeout/disconnect handling, and localhost framing tests.
- Python TCP transport now rejects oversized length-prefixed frames before
  reading payload bytes, using the same 16 MiB cap as the mod TCP server.
- TcpTransport now supports an opt-in length-prefixed auth-token handshake frame
  before regular protocol traffic.
- Python TCP entry points now honor `security.require_token` by reading
  `security.auth_token_env`, and the mod TCP server consumes/verifies the auth
  frame before forwarding protocol requests.
- Env TCP clients now also send a non-empty configured auth token
  opportunistically when present, so local smoke/training/eval connects cleanly
  to token-enabled HKRLEnvMod instances even if the local train config does not
  require token auth.
- Mod TCP server now drains outbound response frames only after authentication,
  preventing stale responses from being exposed to unauthenticated clients.
- `scripts/run_eval.py` now uses the same security token config for evaluator TCP
  env connections.
- SharedMemoryTransport now implements bounded ring send/recv semantics with
  connection lifecycle, timeout, reconnect, and capacity handling.
- SharedMemoryTransport now rejects oversized frames using the same 16 MiB cap
  as TCP transport.
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
- Metric sinks now JSON-normalize numpy scalar/array episode values before
  writing JSONL, CSV record payloads, or stdout JSON lines.
- Metric sinks now normalize non-finite float values to JSON null and write
  strict JSON without NaN/Infinity tokens.
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
- APPO intake now rejects empty RolloutBatches before queuing so learner
  accepted/rejected counts match usable training data.
- APPO intake now rejects in-memory RolloutBatches with non-finite training
  values before queuing.
- Local PPO/RecurrentPPO rollout buffers and update paths now reject non-finite
  training values before they can poison learner weights.
- PPO, APPO, and RecurrentPPO now reject non-finite model outputs, losses, or
  gradient norms before applying optimizer updates.
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
- Mod RESET/SET_TASK handling now preempts pending repeated STEP actions so held
  inputs cannot bleed into the next episode.
- Mod scene reset now rejects unknown numeric task ids instead of silently
  falling back to Gruz Mother.
- Mod reset readiness now fails immediately for invalid task-scene targets
  instead of waiting for the scene-load timeout.
- Schema version 2 adds `StatusCode.NotRunning`, and the mod now rejects
  non-poll STEP requests before the lifecycle reaches `RUNNING`.
- HKRLEnv now computes step reward `dt` from `server_tick` deltas so early
  terminal responses do not overcharge time penalties.
- HKRLEnv now rejects action masks whose length does not match the current task's
  canonical action-mask layout.
- HKRLEnv now rejects action masks with empty categorical groups and STEP
  responses that leave the RUNNING/terminal lifecycle states.
- StepRequest encoding now rejects non-binary button mapping/sequence values
  instead of treating arbitrary truthy values as pressed buttons.
- DefaultReward now rejects non-finite reward inputs and negative amount-bearing
  events before they can pollute scalar rewards or evaluator stats.
- Python action-mask layout construction now rejects negative macro counts
  instead of silently producing a malformed mask width.
- Mod STEP dispatch now emits `InvalidAction` reward events for wire-level
  out-of-range action components before applying safe clamped input.
- Mod StepResponse action masks now use the observed player readiness fields
  instead of always returning the all-valid fallback mask.
- Mod StepResponse action masks now also mask macro actions whose primitive
  sequences require currently unavailable jump, dash, attack, cast, or focus
  inputs.
- Mod PlayerObserver now carries player cooldown, lock, focus, movement-state,
  and invulnerability fields through PlayerState encoding and action-mask
  readiness.
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
- RolloutBatch NPZ deserialization now rejects negative policy versions, flat
  action-mask payloads, and `prev_actions` shapes that differ from `actions`.
- RolloutBatch NPZ deserialization now rejects non-finite observation, reward,
  return, log-probability, previous-reward, and recurrent-state values.
- Checkpoint registry/client parsing now rejects empty checkpoint paths before
  filesystem or HTTP reads.
- Checkpoint registry/client parsing now rejects non-positive versions,
  negative policy/step metadata, and malformed sha256 hashes in index entries.
- Checkpoint registry/client parsing now rejects non-string checkpoint paths and
  bool/float/string version metadata instead of silently coercing malformed
  index entries.
- Checkpoint registry publishing now rejects negative policy/step metadata
  and non-integer policy/step values before writing checkpoint files or index
  entries.
- Checkpoint publishing and worker pulls now reject malformed payloads and
  non-finite model-state tensors before weights can be loaded.
- Evaluator checkpoint loading now applies the same model-state tensor
  validation before `load_state_dict`.
- GameWorker now rejects non-finite policy outputs and malformed packed actions
  before stepping the local environment.
- GameWorker now rejects policy actions that select components disabled by the
  current action mask before stepping the local environment.
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
- Learner TCP intake now rejects non-loopback service binds unless token auth is
  enabled, preserving the LAN/localhost + token-auth runtime boundary.
- Mod reward hooks now wrap event recording in try/catch and log failures through
  `Debug.Logger`, protecting the Unity main thread from hook exceptions.
- Mod reward events now have an observation-delta fallback tracker for damage,
  heal, soul gain, boss kill, and player death events.
- Mod observation collection now catches player/entity/global read failures,
  logs them, and returns a conservative fallback snapshot instead of unwinding
  `FixedUpdate`.
- Mod GlobalState `time_in_episode` now starts at the first `RUNNING` tick for
  each episode instead of using Unity scene-load time.
- Mod StepController now wraps `FixedTick` in a top-level guard that logs
  unexpected failures and clears pending repeated actions.
- HKRLEnv now rejects StepResponses whose `env_id` does not match the local
  request identity, catching stale/cross-env frames before lifecycle handling.
- HKRLEnv now surfaces unbound mod decode-error responses by `StatusCode` before
  tick-echo validation can mask the underlying protocol failure.
- Evaluator model policy runs now pass `prev_action` and `prev_reward` context
  across episode steps, matching the recurrent training/worker input path.
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
- TaskSampler now rejects non-finite per-task win rates before updating weights,
  preventing malformed evaluator metrics from skewing task assignment.
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
- `scripts/run_worker.py` now rejects empty config/task/registry/spool/heartbeat
  path arguments before constructing live env workers or writing local artifacts.
- `HKRLEnvMod` now reads `HKRL_HOST`/`HKRL_PORT` at startup, while
  `scripts/train.py` and `scripts/run_worker.py` expose matching env TCP
  endpoint overrides for live smoke, training, and multi-instance worker runs.
- `scripts/check_env.py` now provides a live HKRLEnvMod PING preflight that
  verifies TCP/schema/auth connectivity without resetting the game scene.
- GitHub Actions now includes a `C# Mod Build` workflow that generates
  FlatBuffers C# bindings and compiles `HKRLEnvMod` against checked-in CI stub
  assemblies for Unity/Hollow Knight/Modding API references.
- `scripts/run_eval.py --policy model` now preserves the task's configured
  macro-action count and validates model actions against the live action mask
  before stepping the env.
- `GameWorker` now stores GRU hidden states in APPO rollout batches when using
  recurrent local inference, keeping remote learner updates aligned with the
  policy state used during sampling.
- `scripts/run_worker.py` now accepts `--tasks` and installs a round-robin task
  provider for multi-task rollout smoke/curriculum runs.
- `scripts/run_coordinator.py` now validates coordinator/task/worker wiring,
  emits one-shot task assignments, ingests heartbeat JSONL, and reports a JSON
  monitoring snapshot for Phase 8 smoke checks.
- `scripts/run_coordinator.py` now rejects empty config/task/bind/heartbeat/eval
  metrics path arguments before task assignment or heartbeat ingestion.
- Coordinator monitoring snapshots now report active worker policy/checkpoint
  version lag, stale-version counts, missing-version counts, and recovering
  worker counts for Phase 8 dashboard health checks.
- `scripts/run_phase8_smoke.py` and `make phase8-smoke` now run an offline
  distributed wiring smoke that validates learner, worker registry probing, and
  coordinator monitoring without a live game.
- `scripts/run_phase8_smoke.py --work-dir` now resets generated checkpoints,
  batches, heartbeats, and eval metrics before each run so repeated Phase 8
  artifact targets do not reuse stale registry state, and holds a work-dir lock
  so concurrent artifact targets do not mix generations.
- `scripts/render_phase8_dashboard.py` and `make phase8-dashboard` now render a
  static Phase 8 fleet dashboard from coordinator or offline-smoke summary JSON.
- Phase 8 dashboard health now marks worker crash churn as degraded even when no
  worker is currently in recovery.
- Phase 8 dashboard health now surfaces workers that have not reported policy or
  checkpoint versions.
- Phase 8 dashboard/profile reports now surface active workers that have not
  been assigned tasks.
- Phase 8 dashboard/profile reports now also identify unassigned workers from
  per-worker `assigned_task = null` rows when aggregate assigned-worker counts
  are missing or stale.
- Phase 8 dashboard/profile reports now surface learner intake counters and flag
  rejected or still-queued learner batches.
- Worker heartbeats and Phase 8 dashboard/profile reports now surface worker-side
  learner upload counters and flag failed or rejected worker uploads.
- Phase 8 dashboard/profile reports now flag policy/checkpoint lag from max-lag
  metrics even when stale-worker counts are absent or zero.
- Phase 8 dashboard/profile reports now also identify lost workers from
  per-worker `alive = false` rows when aggregate lost-worker counts are missing
  or stale.
- Evaluator now supports `--eval-workers` task-level worker pools plus `--ports`
  round-robin env assignment for multi-task regression runs across multiple live
  env instances.
- `scripts/run_eval.py` now rejects invalid live-eval counts, seeds, duplicate
  or out-of-range ports, and multi-worker runs without one port per worker
  before connecting to env instances.
- `scripts/run_eval.py` now rejects empty task/config/checkpoint/baseline/output
  and replay paths before connecting to live env instances or writing eval
  artifacts, and preloads optional baseline metrics before starting a live eval
  run.
- `scripts/render_eval_report.py` and `make phase8-eval-report` now render
  fixed-seed evaluator JSON into JSON/Markdown regression reports.
- Phase 8 eval reports now use `per_boss_win_rate` as the task win-rate fallback
  when evaluator output omits `win_rate` or reports an invalid value.
- Phase 8 eval reports now flag non-object per-task metric payloads as critical
  findings instead of treating them as valid zero metrics.
- Phase 8 eval report summaries now compute win-rate aggregates over valid task
  rows and show separate valid/malformed task counts in JSON/Markdown.
- Phase 8 eval reports now emit a critical no-valid-task finding when every
  task metric payload is malformed.
- Phase 8 eval reports now flag malformed regression deltas as critical findings
  instead of coercing them to zero.
- Phase 8 release evidence manifests now include live eval JSON/report artifacts
  when those game-machine outputs exist locally.
- Phase 8 release evidence verification now rejects included eval reports that
  contain critical findings.
- Phase 8 release evidence verification now rejects included eval reports with
  missing or malformed finding rows.
- Phase 8 release evidence verification now rejects included eval reports with
  missing summaries, task-count mismatches, or no valid task rows.
- Phase 8 release evidence verification now rejects included eval reports with
  malformed task rows or valid-task count drift.
- Phase 8 release evidence verification now rejects included eval reports with
  duplicate task IDs.
- Phase 8 release evidence verification now rejects included eval reports whose
  malformed-task counts do not match task rows.
- Phase 8 release evidence verification now rejects included eval report
  Markdown missing the eval title, report sections, or JSON task rows.
- Phase 8 release evidence verification now rejects included eval report
  Markdown whose task row values drift from eval report JSON.
- Phase 8 release evidence verification now rejects manifests whose `git_sha`
  differs from the expected release commit when provided.
- Phase 8 release checklist and evidence artifacts now record `git_dirty`, and
  verification rejects malformed or drifted dirty-worktree metadata.
- `make phase8-release-checklist` and `make phase8-verify-release-evidence` now
  pass the current Git SHA and dirty-worktree flag through the release tooling.
- Phase 8 release evidence verification now rejects hash-valid Phase 8 smoke
  summaries that are malformed or do not report `ok=true`.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  missing coordinator metrics or learner/worker/task/checkpoint sections.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with malformed smoke config or internal artifact pointers for checkpoint,
  eval-metrics, heartbeat, or work-dir evidence.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose learner, worker, or coordinator path fields do not reference the listed
  smoke artifacts.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator `worker_count`, `active_worker_count`, or `sps` metrics are
  missing or malformed.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator `task_ids`, `num_workers`, or `metrics.worker_count`
  disagree with the top-level task and worker-id sections.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose learner or worker task sections disagree with the top-level task IDs.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose learner and worker algorithm, model, or macro-action layout disagree.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose learner/coordinator binds are not loopback-only or whose dry-run worker
  security/upload fields drift from the offline smoke contract.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator eval win rates, sampler weights, or mastered-task sampler
  state disagree with the listed task IDs.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with malformed learner policy versions or dry-run worker identities.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  missing coordinator worker rows for listed worker IDs.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with unlisted dry-run worker IDs or unexpected coordinator worker rows.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with missing, incomplete, or task-invalid coordinator assignments.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator worker rows disagree with assignments or have malformed
  heartbeat/status metadata.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator aggregate worker monitoring metrics disagree with worker
  rows.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with missing, incomplete, duplicate, or task-invalid task wire IDs.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose worker latest checkpoint or coordinator worker checkpoint versions do
  not match the listed smoke checkpoint versions.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  whose coordinator worker rows lack an `alive` flag or valid worker-side
  `sps`/`worker_crash_count` metrics.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with malformed checkpoint version lists.
- Phase 8 release evidence verification now rejects Phase 8 smoke summaries
  with duplicate top-level task IDs, worker IDs, or checkpoint versions.
- Phase 8 release evidence verification now rejects malformed Phase 8 dashboard
  JSON or dashboard models missing health/metrics/task/worker sections.
- Phase 8 release evidence verification now rejects Phase 8 dashboard models
  with malformed task or worker rows.
- Phase 8 release evidence verification now rejects Phase 8 dashboard models
  with duplicate task or worker rows.
- Phase 8 release evidence verification now rejects malformed Phase 8 dashboard
  HTML or dashboard HTML missing JSON task/worker rows.
- Phase 8 release evidence verification now rejects malformed Phase 8 profile
  JSON or profile reports missing source/metrics/findings/workers.
- Phase 8 release evidence verification now rejects Phase 8 profile reports with
  malformed finding or worker rows.
- Phase 8 release evidence verification now rejects Phase 8 profile reports with
  duplicate worker rows.
- Phase 8 release evidence verification now rejects malformed Phase 8 profile
  Markdown or profile Markdown missing JSON worker rows.
- Phase 8 release evidence verification now rejects Phase 8 profile Markdown
  whose worker row values drift from profile JSON.
- Phase 8 release evidence verification now rejects malformed release checklist
  JSON, missing Phase 8 gates, malformed check rows, or checklist commit/dirty
  drift.
- Phase 8 release evidence verification now rejects malformed release checklist
  Markdown, missing Phase 8 gate IDs, or checklist Markdown commit/dirty drift.
- Phase 8 release evidence verification now rejects release checklist Markdown
  whose command/evidence rows drift from checklist JSON.
- Phase 8 release evidence verification now rejects stale release evidence
  Markdown whose manifest metadata or artifact rows drift from `evidence.json`.
- Phase 8 release evidence verification now rejects release evidence Markdown
  with extra or reordered artifact rows.
- Phase 8 release evidence Markdown now records and verifies the manifest
  version from `evidence.json`.
- Phase 8 release evidence manifest generation now only includes live eval
  artifacts when the full eval JSON/report group exists locally.
- `scripts/render_profile_report.py` and `make phase8-profile` now render static
  Phase 8 profiling reports from coordinator/offline-smoke summaries.
- Phase 8 profile Markdown worker tables now show each worker's alive flag next
  to its status so heartbeat-expired workers are visible per row.
- Phase 8 profile reports now flag workers that have not reported policy or
  checkpoint versions.
- Phase 8 profile reports now flag workers marked lost by the coordinator
  heartbeat timeout.
- `docs/release.md`, `scripts/render_release_checklist.py`, and
  `make phase8-release-checklist` now define a Phase 8 release evidence checklist.
- Phase 8 release checklist Markdown rendering now tolerates malformed check rows
  so checklist artifacts can still be inspected.
- `scripts/render_release_evidence.py` and `make phase8-release-evidence` now
  produce a sha256 manifest for Phase 8 release artifacts.
- `scripts/verify_release_evidence.py` and `make phase8-verify-release-evidence`
  now verify Phase 8 release artifacts against that manifest.
- Phase 8 release evidence verification now fails if manifest `artifact_count`
  or `total_bytes` drift from the listed artifact rows.
- Phase 8 release evidence verification now rejects unsupported
  `manifest_version` values.
- Phase 8 release evidence verification now rejects duplicate artifact paths in
  manifests.
- Phase 8 release evidence verification now rejects absolute artifact paths in
  manifests while generation still normalizes absolute inputs under the repo root.
- Phase 8 release evidence verification now rejects non-normalized artifact path
  aliases in manifests.
- Phase 8 release evidence verification now rejects missing or malformed
  full-length `git_sha` values in manifests.
- Phase 8 release evidence verification now rejects manifests that omit required
  offline evidence artifacts.
- Phase 8 release evidence verification now rejects missing or unsupported
  release `version` values in manifests.
- Phase 8 release evidence verification now rejects partial live eval artifact
  groups so `eval.json`, `eval-report.md`, and `eval-report.json` travel together.
- Phase 8 release evidence verification now rejects non-object artifact entries
  in manifests.
- Phase 8 release evidence Markdown rendering now tolerates malformed artifact
  rows so the verifier can still report structured failures.
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

### Fixed
- CheckpointClient now bypasses system HTTP proxies for localhost/private
  checkpoint registries, keeping worker checkpoint pulls local and stable in
  proxied developer environments.
- Phase 8 release evidence verification now rejects smoke summaries that omit
  coordinator aggregate worker monitoring metrics instead of only checking them
  when present.
- Phase 8 release evidence verification now rejects malformed worker version and
  rollout/upload telemetry instead of treating invalid worker metric values as
  present.
- Phase 8 dashboard/profile rendering now treats malformed worker `alive`
  values as unknown instead of coercing truthy strings to online workers.
- Phase 8 dashboard/profile rendering now ignores malformed numeric strings for
  worker, learner, and eval win-rate telemetry instead of accepting them as
  trusted metrics.
- GameWorker learner upload accounting now treats malformed uploader ACKs as
  failures instead of counting any non-`False` value as an accepted batch.
- `scripts/run_worker.py` now rejects malformed worker ids, step limits,
  recovery thresholds, and learner endpoints before constructing live worker
  state.
- `scripts/run_phase8_smoke.py` now rejects malformed config/task paths, worker
  counts, seeds, work directories, and requested artifact output paths before
  clearing or writing smoke artifacts.
- Release checklist/evidence render scripts now reject malformed `git_dirty`,
  empty version/output paths, and malformed evidence artifact path lists before
  writing release metadata.
- Release evidence verification now rejects malformed expected `git_dirty`,
  manifest/root/output paths before loading or writing verification reports.
- Phase 8 dashboard/profile render scripts now reject empty input/output paths
  before reading summaries or writing release artifacts.
- Phase 8 eval report rendering now rejects empty eval input and report output
  paths before reading fixed-seed eval JSON or writing report artifacts.
- Coordinator heartbeat ingestion now rejects non-finite numeric metrics before
  they can poison worker monitoring aggregates.
- Coordinator startup now rejects malformed heartbeat timeouts, worker
  counts/ids, seed overrides, and ambiguous explicit worker-id/worker-count
  combinations before building fleet state.
- `run_coordinator.py --eval-metrics` now rejects malformed, non-finite, or
  out-of-range win-rate values instead of silently clipping bad evaluator input.
- `run_coordinator.py --eval-metrics` now rejects non-object or win-rate-missing
  per-task metric rows instead of silently omitting malformed evaluator tasks.
- TaskSampler now rejects malformed or out-of-range win-rate updates instead of
  silently clipping bad curriculum inputs.
- Phase 8 eval report rendering now marks tasks with missing or invalid
  `win_rate`/`per_boss_win_rate` evidence as malformed instead of coercing bad
  values to zero.
- Phase 8 eval report rendering now marks tasks with malformed numeric
  damage/timing/ratio metrics as invalid and rejects string-like regression
  deltas instead of accepting numeric-looking strings.
- Phase 8 eval report rendering now rejects non-finite or non-numeric
  win-rate/regression threshold arguments instead of letting NaN disable
  regression findings.
- `make phase8-eval-report` now fails after writing JSON/Markdown artifacts
  when the rendered fixed-seed eval report contains critical findings.
- `scripts/run_learner.py` now rejects malformed learner gate overrides for
  intake counts/timeouts, staleness, publish cadence, entity capacity, and macro
  count before partially constructing the learner.
- `scripts/run_learner.py` now rejects empty config/task/bind/batch/checkpoint
  path arguments and `--serve-forever`/`--intake-count` conflicts before model,
  registry, or intake-server construction.
- Phase 8 release evidence verification now validates eval-report task row
  numeric fields instead of trusting `metrics_valid` alone.
- Evaluator regression reports now reject malformed, non-finite, or out-of-range
  baseline/current win-rate metrics before computing catastrophic-forgetting
  deltas.

### Decisions
- ADR-0001 RL framework: self-built PyTorch PPO.
- ADR-0002 serialization: FlatBuffers single source of truth.
- ADR-0003 mod framework: HK Modding API.
- ADR-0004 local inference + remote training decoupled.
