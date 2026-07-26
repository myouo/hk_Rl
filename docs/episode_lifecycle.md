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
- Reset always re-enters the target scene through
  `GameManager.BeginSceneTransition`, including when it is already active, so
  the next episode recreates the boss and clears projectiles/player-death state
  without destroying the persistent Hero. If RESET arrives at `Menu_Title`, the
  Mod first loads the configured Godhome-capable `HKRL_SAVE_SLOT` (1–4,
  default 1).
- Save-slot bootstrap waits for a loaded, active, non-menu scene and an active
  persistent Hero before requesting the boss transition. A valid save may load
  with the Hero seated at a bench, where control and gravity are intentionally
  relinquished. PlayerAction input cannot be consumed in that state, so RESET
  waits for the active scene's `Bench Control` FSM to reach `Resting`, then
  sends its canonical `GET UP` event once. That FSM clears `atBench`, disables
  the bench kinematic mode, restores gravity/control/animation, and finishes
  its get-off sequence.
  RESET then requires normal control, gravity, body, constraints, and collision
  readiness before beginning the arena transition. This lifecycle-only recovery
  makes no direct Transform/Rigidbody write and is never policy-selected. The
  same strict readiness checks plus boss readiness still gate the target
  arena's final RESET ACK.
- Same-scene readiness requires a new Unity scene `handle`; the old arena cannot
  satisfy reset readiness while an asynchronous reload is still pending.
- On the first transition from another scene, the Mod mirrors the game's Hall
  of Gods non-cinematic bookkeeping: it sets `bossSceneToLoad`, clears
  semi-persistent combat state, supplies `BossSceneController.SetupEvent`, and
  uses the `GodsAndGlory` load visualization. It deliberately does not set
  `enterWithoutInput`, accept input, drive a Workshop statue FSM, or broadcast
  workshop-local cinematic events.
- A same-scene RESET deliberately uses the Godhome fast-reload branch: reinstall
  `SetupEvent`, restore Hero health, set `enterWithoutInput`, and call
  `BeginSceneTransition` without re-broadcasting the workshop-only
  `DREAM ENTER` / `GG TRANSITION OUT` events. Broadcasting those events from the
  active arena lets its old transition FSM relinquish the persistent Hero just
  before scene unload, which can strand its body without gravity or terrain
  collisions in the replacement scene.
  Direct `BeginSceneTransition` does not clear the legacy
  `hasFinishedEnteringScene` handshake, so RESET marks only that GameManager
  flag pending immediately before loading. The normal Hero entry sets it
  complete. This prevents a replacement Boss's
  `WaitForFinishedEnteringScene` action from consuming the previous episode's
  completion value.
  PlayerAction injection remains disabled until reset reaches `RUNNING`.
- A failed, timed-out, cancelled, or superseded RESET releases only the static
  `BossSceneController.SetupEvent` callback it owns. A stale callback cannot
  configure a later task.
- A reset **failure** returns a non-`Ok` `StatusCode` (e.g. `ResetTimeout`,
  `BossNotFound`) — never silently continue training on a bad episode.

## 3. Readiness checks

- `WAIT_SCENE_READY`: target scene loaded and active; invalid or unknown scene
  targets fail with `StatusCode.SceneLoadFailed`.
- `WAIT_PLAYER_READY`: `HeroController` spawned and active, with
  `acceptingInput == true`, control not relinquished, a gameplay actor state,
  `cState.transitioning == false`,
  `transitionState == WAITING_TO_TRANSITION`, positive `Rigidbody2D.gravityScale`,
  a dynamic and simulated body without frozen X/Y constraints, an enabled
  collider, and active terrain-ingress checks. This fails closed if the
  Godhome intro leaves the Hero in its gravity-off/no-collision transition
  state.
- `WAIT_BOSS_READY`: current-scene configured and active Boss objects are
  present with `HealthManager`; a Godhome `BossSceneController` must also report
  `HasTransitionedIn`. Discovery includes inactive configured objects in a
  bounded current-scene cache, while observation exposes active objects only.
  RESET never sends Hero exploration input or forces a Boss FSM state, physics
  property, position, velocity, or hp value.

All three conditions must stay true for 100 ms of unscaled wall time before
`RUNNING`; this filters the transient frame where `HasTransitionedIn` is set as
the event is emitted but the PlayMaker transition has not yet taken control.
Each wait has a timeout → `StatusCode.*Timeout/NotFound`.

Natural Boss activity is an evaluator acceptance check after `RUNNING`, because
some valid scenes require ordinary Hero traversal. The all-Boss sweep requires
activity and a positive/full combat-health baseline both before and after a
same-scene RESET; `0/0` transition placeholders remain observable but are not
combat-health entities.

See [ADR-0006](./adr/0006-godhome-entry-readiness.md).

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

`HKRLEnv.step()` also enforces the task's `time_limit_seconds` against observed
game time. A live Mod terminal remains `terminated`; reaching only the task
deadline returns `truncated=true` with `info["time_limit_reached"]=true`.

`GameWorker` treats both outcomes as episode boundaries. It stores the terminal
transition first, clears recurrent previous-action/reward context, then performs
the ordinary RESET handshake before collecting another transition. Its
heartbeat includes `arena_auto_reset_count`.

For standalone live validation,
[`BossArenaSupervisor`](../python/hkrl/arena.py) implements the same ordering:

```text
terminal observation/events
  → immutable ArenaEpisodeResult
  → RESET through the normal lifecycle
  → require next_episode_id != terminal_episode_id
  → next attempt
```

The supervisor never revives the Hero, respawns a Boss, edits health, or forces
an FSM. Scene recreation and state cleanup remain owned by RESET. This follows
the terminal-preservation principle used by
[Gymnasium's autoreset wrapper](https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/common/#Autoreset)
while keeping the project's explicit reset acknowledgement and episode-id
invariant.
The live acceptance command is:

```bash
python scripts/live_boss_arena.py \
  --policy contact \
  --episodes 1 \
  --output runs/live/boss-arena-death-autoreset.json
```

## 5. Why this matters

Reset contamination is one of the most insidious RL-on-games bugs: stale events,
half-loaded scenes, and un-spawned bosses quietly poison the training data and
the agent learns from noise. The state machine + ack + event-clear + `episode_id`
make episodes clean by construction. See PRD §9.3 and
[`reward_design.md`](./reward_design.md) §5.
