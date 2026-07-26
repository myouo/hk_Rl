# ADR-0005: Pause-safe Hollow Knight time control

- Status: **Accepted**
- Date: 2026-07-25

## Context

The environment server consumes protocol requests in Unity `FixedUpdate`, but
Unity stops scheduling `FixedUpdate` when `Time.timeScale` is zero. Applying
`PAUSE` there and waiting for a later `RESUME` in the same loop therefore
self-deadlocks.

Writing `Time.timeScale` directly also bypasses Hollow Knight's
`GameManager.SetTimeScale(float)` / `TimeController.GenericTimeScale` path,
which the game uses for hit-stop, menus, and scene transitions. Multiplying the
environment acceleration into `Time.fixedDeltaTime` makes each physics step
coarser and can cause collision tunnelling and distorted jump/gravity behavior.

Unity documents the zero-scale `FixedUpdate` behavior in
[`Time.timeScale`](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Time-timeScale.html).
Existing Hollow Knight implementations use the game time controller, including
[HKRL's `TimeScale.cs`](https://github.com/AdityaJain1030/HKRL/blob/master/Game/TimeScale.cs)
and
[DebugMod's `TimeScale.cs`](https://github.com/TheMulhima/HollowKnight.DebugMod/blob/master/Source/MonoBehaviours/TimeScale.cs).

## Decision

- Compose the environment multiplier with game-requested scales by hooking
  `GameManager.SetTimeScale(float)` and writing
  `TimeController.GenericTimeScale`.
- Keep the captured baseline `Time.fixedDeltaTime` unchanged. Acceleration
  increases the number of fixed physics steps per wall-clock second, not the
  simulated duration of each step.
- Add a narrow main-thread `Update` recovery pump. While scaled time is stopped,
  it may only peek at a queued `RESUME`, `RESET`, `SET_TASK`, or
  `SET_TIMESCALE` request and restore the clock. It never dequeues the request,
  applies an action, advances lifecycle state, reads game state, or emits a
  response; those remain owned by `FixedUpdate`.
- Make `Resume` idempotently re-assert the active scale, including when the
  zero scale originated in a scene/menu transition rather than `PAUSE`.
- Unhook and restore the pre-mod time/fixed-step values when the controller is
  disposed.

## Consequences

- `PAUSE` can always be followed by a protocol recovery command without
  restarting the game or mod.
- Hollow Knight's hit-stop and transition callbacks retain their semantics
  while still composing with the training multiplier.
- Player and boss Rigidbody2D/FSM simulation continues at the game's native
  fixed-step resolution at accelerated training speeds.
- The main-thread-only rule is preserved. The sole non-`FixedUpdate` Unity write
  is the clock wake-up needed to make `FixedUpdate` schedulable again.

## Alternatives rejected

- **Consume and answer protocol requests from `Update`.** This would split
  lifecycle, action, and observation ownership across two Unity loops.
- **Use a tiny non-zero pause scale.** Recovery latency depends on the chosen
  scale and the game is not actually frozen.
- **Write only `Time.timeScale`.** Hollow Knight can overwrite it during
  hit-stop/transitions and its internal time-controller state diverges.
- **Scale `fixedDeltaTime` with the multiplier.** This preserves callback count,
  not physics fidelity, and is inappropriate for the RL environment.
