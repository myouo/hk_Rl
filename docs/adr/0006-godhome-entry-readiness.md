# ADR-0006: Hall of Gods entry and physics-safe readiness

- Status: **Accepted**
- Date: 2026-07-25
- Updated: 2026-07-26

## Context

Loading a `GG_*` scene and finding an active `BossSceneController` does not mean
the fight is ready. `BossSceneController.Start()` sends `GG TRANSITION IN` and
sets `HasTransitionedIn` immediately, while the transition prefab's PlayMaker
FSM continues asynchronously. During scene entry, `HeroController` deliberately
sets `gravityScale` to zero, disables ordinary control, and uses a transition
state until `FinishedEnteringScene` and the Godhome transition release it.

Real-game testing exposed three independent transition errors:

1. A hybrid first-entry path combined a direct scene transition with
   `EnterWithoutInput(true)`. The omitted Workshop cinematic sequence never
   released that ownership, so the persistent Hero could remain gravity-off or
   control-locked in an otherwise loaded arena.
2. Workshop Boss Challenge events are local to their cinematic objects.
   Broadcasting those names globally can reach unrelated persistent FSMs and
   leave Hero physics or Boss entry state owned by an unloaded source object.
3. The game's legacy `TransitionScene(TransitionPoint)` clears the private
   `hasFinishedEnteringScene` handshake, while the supported direct
   `BeginSceneTransition(SceneLoadInfo)` path does not. A same-scene reload
   could therefore carry `true` from the previous episode. The replacement
   Boss's `WaitForFinishedEnteringScene` action consumed that stale value,
   advanced its control FSM into `Wake` before the persistent Hero actually
   entered, and then remained motionless.

The 44 live Hall of Gods scenes also do not share one post-load lifecycle.
Most Bosses activate immediately, Winged Nosk needs ordinary Hero traversal,
Troupe Master Grimm replaces an initial `0/0`-HP transition object with the
combat object, and Nightmare King Grimm keeps a `0/0` transition placeholder
beside its `1250/1250` combat object. Requiring natural Boss movement inside
RESET either blocks valid scenes or forces RESET to inject a hidden policy.
That previously pushed the Hero to arena walls and made later control checks
order-dependent.

The game's community-readable `BossChallengeUI` and public mod examples show
the relevant transition/setup shape:

- [Hollow Knight `BossChallengeUI`](https://github.com/ayushpaharia/hollow-knight-code/blob/42323872025de60fe6d61c40031feea5036f354d/Assembly-CSharp/BossChallengeUI.cs)
- [HKRL `SceneHooks`](https://github.com/AdityaJain1030/HKRL/blob/f0ca584a6186aeba7771b9c5a80a296dd46c0999/Game/SceneHooks.cs)
- [HollowGym `SceneUtils`](https://github.com/CiottoloMaggico/HollowGym/blob/1499d3b4e2ca8236aa9c7d942d59cc37735940a0/src/Utils/SceneUtils.cs)
- [GodhomeQoL `FastReload`](https://github.com/NightFuryoOo/GodhomeQoL/blob/32539e89be60d2aed22c8e0a6b13dfa02630135c/Modules/FastReload/FastReload.cs)
- [BossAttacks end-to-end Boss test](https://github.com/royitaqi/HollowKnight.BossAttacks/blob/master/BossAttacks/Utils/E2eBossFightTest.cs)

These are community sources, not official Team Cherry API guarantees, so the
real-assembly build and live-game acceptance test remain required.

## Decision

- Enter boss arenas only through `GameManager.BeginSceneTransition` with
  `door_dreamEnter` and `SceneLoadVisualizations.GodsAndGlory`.
- Before every transition, set `bossSceneToLoad` and install the
  `BossSceneController.SetupEvent` callback used by the normal challenge flow.
- On first/cross-scene entry, perform only non-cinematic Hall bookkeeping:
  configure `dreamReturnScene`, clear Hero MP send events, advance time, and
  reset semi-persistent items. Do **not** set `enterWithoutInput`, call
  `AcceptInput`, drive a Workshop statue FSM, or synthesize DREAM/GG events.
- For a same-scene Boss reload, follow GodhomeQoL's fast-reload shape: reinstall
  the scene setup callback, restore Hero health, set
  `EnterWithoutInput(true)`, accept input, and directly call
  `BeginSceneTransition`. Do not replay the Workshop's dream/transition-out
  broadcasts.
- Immediately before `BeginSceneTransition`, set only GameManager's private
  `hasFinishedEnteringScene` handshake to `false`. Fail RESET if that
  version-pinned field cannot be resolved. The game's normal
  `HeroController.FinishedEnteringScene()` path sets it back to `true`; no Hero
  or Boss FSM event is synthesized.
- Do not repair a bad entry by writing `Transform`, `Rigidbody2D`, velocity,
  gravity, collider, or PlayMaker FSM state.
- Release the exact `BossSceneController.SetupEvent` callback owned by the
  pending RESET on timeout, failure, cancellation, or before the next task.
  A timed-out task must not configure the next scene.
- Discover Boss `HealthManager` components from a bounded cache of the active
  Unity scene, including inactive configured objects. Never accept persistent
  or previous-scene objects; observations include active objects only.
- Treat the Hero as ready only when it is active, accepting input, has not
  relinquished control, is in a gameplay actor state, is no longer transitioning,
  is in `WAITING_TO_TRANSITION`, has positive gravity, has a non-kinematic
  simulated body with unfrozen X/Y constraints, carries an enabled collider,
  and has active terrain-ingress checks.
- Require the complete scene/player/Boss predicate to remain true for 100 ms of
  unscaled wall time before entering `RUNNING`. The Boss part is an object gate:
  transitioned controller plus configured and active current-scene Boss
  objects. RESET disables injected input and never explores on the policy's
  behalf.
- Verify natural Boss activity after the RESET acknowledgement in the evaluator,
  not in the Mod gate. The live acceptance policy may use only bounded,
  mask-aware Hero primitives. It requires position, velocity, FSM, HP, or
  combat-entity-set activity, then repeats activation after a same-scene RESET.
- Treat only `max_hp > 0` Boss entities as combat-health baselines. A valid
  lifecycle must expose at least one positive, full-health combat entity, and
  the sorted health-capacity vector must match after RESET. `0/0` cinematic
  placeholders remain observable but cannot invalidate or satisfy this gate.
- Include the individual Hero transition/physics gates in reset timeout
  diagnostics, plus read-only Boss/FSM state descriptions.

## Consequences

- Policy input cannot race the Boss intro or observe a gravity-off transition
  state as a valid episode.
- A stale scene-entry completion value cannot prematurely release a replacement
  Boss, and a failed RESET cannot contaminate the next task's setup callback.
- Repeated episodes no longer inherit control/physics state from the old arena's
  transition-out FSM.
- Arena barriers, Boss activation, Hero gravity, and collision remain owned by
  Hollow Knight's normal FSMs.
- Distance-gated and delayed-health Bosses remain learnable because activation
  actions belong to the policy/evaluator rather than RESET. A loaded-but-dormant
  Boss still fails live acceptance if bounded ordinary input cannot activate it.
- A broken or game-version-incompatible entry fails as `PlayerNotReady` with
  actionable diagnostics instead of silently producing corrupt rollouts.
- Reset gains a small 100 ms wall-time readiness delay.

## Alternatives rejected

- **Use only `BossSceneController.HasTransitionedIn`.** The flag is raised when
  the transition event is sent, not when the asynchronous Hero transition
  finishes; it remains necessary but is not sufficient.
- **Globally replay the Workshop cinematic event names.** Vanilla targets
  local transition objects. A broadcast can reach persistent or arena FSMs and
  leave the Hero's physics disabled after source-scene unload.
- **Sleep for a fixed multi-second delay.** Public examples do this, but a state
  predicate plus a short stability window is faster and robust to time scale.
- **Repair a failed arena entry by calling `RegainControl`,
  `AffectedByGravity(true)`, or writing physics fields.** This hides an
  incomplete transition and bypasses arena/Boss FSM setup. The narrow
  pre-transition saved-bench release is separate: it sends `GET UP` to the
  source scene's real `Bench Control` FSM and waits for that lifecycle to
  restore the body before scene loading.
- **Move the Hero automatically during RESET until Boss activity appears.**
  This steals exploration from the model, can strand the Hero at a wall, and
  makes the initial policy state task-specific.
- **Force the Boss out of `Wake`/`Dormant`, send START/ENTER, enable its FSM, or
  write HP/velocity.** This fabricates training dynamics. The accepted live test
  drives only ordinary Hero inputs and observes the unmodified Boss lifecycle.
