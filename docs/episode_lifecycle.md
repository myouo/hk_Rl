# Episode Lifecycle

> Implements PRD §5.7 + §9.3. Mod code: `Env/EpisodeLifecycle.cs`,
> `Env/ResetManager.cs`. Wire enum: `LifecycleState` in
> [`../schema/hkrl.fbs`](../schema/hkrl.fbs).

## 1. State machine

```text
IDLE
  → RESET_REQUESTED
  → FREEZE_INPUT
  → CLEAR_EVENTS
  → LOAD_SCENE
  → WAIT_SCENE_READY
  → WAIT_PLAYER_READY
  → WAIT_BOSS_READY
  → RESTORE_PLAYER_STATE
  → CLEAR_PROJECTILES
  → COUNTDOWN
  → RUNNING            ← only here may the worker send STEP
  → TERMINATING        ← entered on death / win / scene change
  → REPORT_DONE
  → CLEANUP
  → IDLE
```

## 2. Hard requirements (PRD §5.7)

- Reset must NOT mix in reward events from the previous episode
  (`CLEAR_EVENTS` before anything else collects).
- `STEP` with real input is only valid once `lifecycle_state == RUNNING`; during
  reset, only the canonical no-op poll `STEP` is accepted (see
  [`protocol.md`](./protocol.md) §4).
- Every episode has a unique `episode_id`.
- Death, win, and scene change all route to `TERMINATING`.
- After `done`, no new reward events are collected.
- `RESET` / `SET_TASK` cancels any pending repeated `STEP`; old held/repeated
  actions must not carry into the next episode.
- `RESET` / `SET_TASK` uses the task config's `scene` value from
  `StepRequest.task_scene`; the legacy numeric `task_id` scene map is only a
  fallback for older clients.
- A reset **failure** returns a non-`Ok` `StatusCode` (e.g. `ResetTimeout`,
  `BossNotFound`) — never silently continue training on a bad episode.

## 3. Readiness checks

- `WAIT_SCENE_READY`: target scene loaded and active; invalid or unknown scene
  targets fail with `StatusCode.SceneLoadFailed`.
- `WAIT_PLAYER_READY`: `HeroController` spawned, controllable.
- `WAIT_BOSS_READY`: required boss(es) present with valid `HealthManager`.

Each wait has a timeout → `StatusCode.*Timeout/NotFound`.

Mod implementation: `StepController.FixedTick()` starts `ResetManager` on
`RESET`/`SET_TASK`, polls it while the lifecycle is in reset states, and only lets
`EpisodeLifecycle` leave the wait states after scene/player/boss readiness has
been confirmed. Reset failures call `EpisodeLifecycle.Fail(status)` and are
reported through `StepResponse.error_code`.
While running, terminal reward events (`BossKilled`, `PlayerDeath`,
`SceneChanged`) call `EpisodeLifecycle.RequestTerminate()` before the response is
encoded.

## 4. Worker-side contract

The Gym `reset()` (`hkrl/env.py`) issues `RESET`, polls until `RUNNING` or an
error code, then returns the first observation. On error it surfaces the code
(and increments `reset_failure` metric) rather than yielding a garbage obs.

## 5. Why this matters

Reset contamination is one of the most insidious RL-on-games bugs: stale events,
half-loaded scenes, and un-spawned bosses quietly poison the training data and
the agent learns from noise. The state machine + ack + event-clear + `episode_id`
make episodes clean by construction. See PRD §9.3 and
[`reward_design.md`](./reward_design.md) §5.
