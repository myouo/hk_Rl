# Mod Development Guide

> Implements PRD §5.1–§5.3. Mod framework: **HK Modding API**
> ([ADR-0003](./adr/0003-mod-framework-hk-modding-api.md)). Code: `mod/HKRLEnvMod/`.

## 1. Prerequisites

- C# + Unity `MonoBehaviour` lifecycle: `Awake`, `Start`, `Update`,
  `FixedUpdate`, coroutines.
- [Hollow Knight Modding API](https://github.com/hk-modding/api) (MonoMod-based).
- Harmony patching.
- Decompiling `Assembly-CSharp` with ILSpy / dnSpy.
- Threading basics: `lock`, `ConcurrentQueue`, ring buffer.
- Unity main-thread rule: never access/modify Unity objects off the main thread.

## 2. Key game internals

`HeroController`, `PlayerData`, `HealthManager`, `BossSceneController`,
`GameManager`, `PlayMakerFSM`. Godhome / Hall of Gods scenes (e.g.
`GG_Hornet_1`) are the MVP arenas.

## 3. Environment setup (Phase 0)

```text
[ ] Install Hollow Knight (Steam).
[ ] Install the Modding API (or BepInEx — but we standardize on Modding API).
[ ] Configure a C# IDE referencing the game's managed assemblies.
[ ] Decompile Assembly-CSharp to confirm class/field/FSM names.
[ ] Build a Hello-World mod; log player position + scene name.
```

`HKRLEnvMod.csproj` references must point at the local game install
(`Managed/Assembly-CSharp.dll`, `UnityEngine.*`, the Modding API). These paths
are machine-specific — keep them in a local `.csproj.user` / props file, not in
source. Compilation is deferred until a machine with the game is configured.
The GitHub `C# Mod Build` workflow compiles the mod against minimal CI stubs
under `mod/ci-stubs/`; it catches repository-level C# compile/schema drift but
does not replace a final build against real Hollow Knight assemblies.

The distributable Mod version has a single source:
`mod/HKRLEnvMod/Version.props`. `HKRLEnvMod.GetVersion()` reads the generated
assembly version, so do not add a second literal version in C# or an install
script. Protocol compatibility remains independently pinned by the matching
schema constants in Python and C#.

At runtime the mod starts the TCP environment server from the persistent
`HKRLDriver`. Defaults are `127.0.0.1:5555`; set these environment variables
before launching Hollow Knight to line up live smoke, evaluator, or worker
processes with a specific game instance:

```bash
export HKRL_HOST=127.0.0.1
export HKRL_PORT=5555
export HKRL_SAVE_SLOT=1             # Godhome-capable save slot, 1..4
export HKRL_AUTH_TOKEN=dev-secret   # optional; enables TCP env auth
```

On Linux Steam/Proton, an already-running Steam client may not inherit variables
from the worker-launch shell. The Mod therefore also accepts a strict
`hkrl-runtime.conf` next to `HKRLEnvMod.dll`. Environment variables have higher
priority. [`scripts/linux/start_game_worker.sh`](../scripts/linux/start_game_worker.sh)
writes the file atomically with mode `0600`; token values are never logged. When
RESET is first requested from the title menu, the Mod loads `HKRL_SAVE_SLOT`
(default 1) before entering the configured Godhome arena.

Python env clients (`check_env.py`, local training, workers, evaluators) send
the same non-empty token automatically when it is present. Sending the auth
preface is harmless when mod auth is disabled, so local smoke commands do not
need a separate config edit just to match a token-enabled mod.
The TCP server treats each client connection as an isolated env session: when a
worker/evaluator disconnects or reconnects, the network thread clears queued
request/response frames and detects half-closed sockets before accepting the
next client. The Python side still issues a clean `RESET` after reconnect; the
network thread only moves frames and never touches Unity state. Request and
response queue entries carry a mod-internal transport-session id so delayed
main-thread responses from a disconnected client cannot be drained to the next
client. The main-thread controller also clears held input/repeat state when
client session state changes or a control command preempts a repeat.

`ActionApplier` stores the selected primitive during `FixedUpdate`.
`InputInjector` then commits it from InControl's public `InputManager.OnUpdate`
event, immediately after physical devices and `PlayerActionSet`s refresh. This
lets both `HeroController` and independent PlayMaker consumers such as a bench
FSM see the same injected input during that game-input tick. It preserves the
main-thread invariant while aligning injected `WasPressed`/`WasReleased` edges
with InControl's tick. Driver disposal neutralizes the action set and
unsubscribes the event so Mod reloads cannot accumulate callbacks.
`StepController` waits for `InputInjector.SuccessfulCommitCount` to advance
before collecting a STEP response. This causal boundary prevents
`action_repeat=1` from returning the observation captured before the selected
action reached Hollow Knight.

For multi-instance evaluation or worker scale-out on one game machine, launch
each Hollow Knight instance with a distinct `HKRL_PORT` and pass the matching
`--port`/`--ports` or `--env-port` value to the Python entry point. Keep
`HKRL_HOST` loopback unless the deployment is explicitly firewall-scoped to a
trusted LAN.
Use `python scripts/check_env.py --host HOST --port PORT` as the first live
diagnostic: it sends `PING` through the same FlatBuffers/TCP/auth path without
resetting the scene. Connection, timeout, and protocol failures print a JSON
summary with `ok: false` and exit non-zero so CI or launch scripts can fail fast
without scraping a Python traceback.

## 4. Module map (PRD §5.2)

```text
HKRLEnvMod/
  HKRLEnvMod.cs            Mod entry (HK Modding API `Mod` subclass)
  Transport/  TcpServer, MessageCodec (FlatBuffers), Protocol, Heartbeat
  Env/        StepController, EpisodeLifecycle, ResetManager, SimControl, SceneController
  Observation/ ObservationCollector, Global/Player/Entity/Boss/Projectile/Hazard observers, EntityRegistry
  Action/     ActionApplier, InputInjector, ActionMasker, MacroActionScheduler
  Rewards/    RewardEventBuffer, Damage/Heal/Death/Scene hooks
  Debug/      Overlay, Logger, SnapshotRecorder
  Schema/     generated FlatBuffers C# (do not edit)
```

## 5. Threading model (PRD §5.3) — the critical rule

```text
WRONG:  network thread receives action and directly calls HeroController.

RIGHT:
  NetworkThread:  recv StepRequest -> ConcurrentQueue.Enqueue
  MainThread (Update, only at timeScale=0):
      peek recovery command -> restore clock; leave request queued
  MainThread (FixedUpdate):
      dequeue latest action -> apply (InputInjector)
      collect observation, collect reward events
      write StepResponse snapshot to out-queue
  NetworkThread:  dequeue -> send StepResponse
```

The network thread must never touch Unity objects. Gameplay reads/writes happen
in `FixedUpdate` on the main thread. The only exception is the main-thread
zero-scale clock wake-up above; it cannot dequeue, dispatch, observe, or respond.
Use `ConcurrentQueue` / ring buffers across the boundary. See
[ADR-0005](./adr/0005-pause-safe-game-time-control.md).

## 6. Robustness (PRD §9.9)

- Wrap every hook in try/catch; log via `Debug/Logger`.
- `schema_version` + mod-version lock; observation health checks.
- Fallback entity fields when an FSM/field is missing on a game update.
- Debug overlay (`Debug/Overlay.cs`) to visually verify entities/hitboxes.
- Unit-test the critical hooks: enter scene, read boss, reset, death, kill.

The repository C# gate also runs `InputInjectionSmoke`: a runtime stub raises
`InputManager.OnUpdate` and verifies every movement/aim/button mapping, nail-art release
edge, commit acknowledgement, macro progression/reset, duration suspension,
bounded movement continuation across a STEP response, immediate transient-button
release, stale-continuation expiry, reward-hook lifecycle,
neutral-on-dispose behavior, hook removal, and the
environment/file runtime-config precedence used by Steam/Proton. It also verifies
that game-requested time scales compose with the environment multiplier, the
physics step stays fixed, PAUSE/RESUME recovers an externally stranded clock,
and disposal restores/unhooks time control. Hero readiness smoke cases reject
transitioning, relinquished/no-input, zero-gravity, kinematic, non-simulated,
position-constrained, or collision-less states and require a continuous
unscaled-time stability window. Transition-policy smoke cases also keep
same-scene reloads on the direct fast-reload path instead of replaying
workshop-only dream events. It does not
replace the real-game acceptance gates in
[`linux_ssh_deployment.md`](./linux_ssh_deployment.md).

## 7. Time control (PRD §9.6)

`SimControl` hooks Hollow Knight's `GameManager.SetTimeScale(float)` and composes
the environment multiplier through `TimeController.GenericTimeScale`. It keeps
the captured baseline `Time.fixedDeltaTime` unchanged: acceleration increases
fixed physics steps per wall-clock second instead of making collision/gravity
integration coarser. `StepController` applies `PAUSE`, `RESUME`, and
`SET_TIMESCALE` on the main thread. Pair with `action_repeat`.

Because Unity does not schedule `FixedUpdate` at a zero time scale, the
persistent driver has one narrow `Update` escape hatch. It may only peek at a
queued recovery command and restore the clock; request consumption, actions,
lifecycle changes, observations, and responses still happen in `FixedUpdate`.
See [ADR-0005](./adr/0005-pause-safe-game-time-control.md).

## 8. Main-thread performance

`Time.fixedDeltaTime = 0.02` schedules a nominal 50 physics updates per
game-time second; it is not a 50 FPS render cap. A rendered frame may have zero,
one, or multiple fixed updates. Do not alter that baseline to hide observation
latency, because doing so changes collision and gravity integration.

The connected-client hot path follows these constraints:

- `InputManager.OnUpdate` commits every game-input frame after physical action
  sets refresh, so physical input cannot overwrite the policy action and
  non-Hero FSMs see it. Member/method reflection is resolved only when the
  `HeroActions` instance changes. Closed delegates perform subsequent commits
  without `MethodInfo.Invoke` arrays or boxing.
- Projectile candidate discovery scans `DamageHero`/`DamageEnemies` components
  at most every 100 ms of unscaled time; it never enumerates every scene
  `Transform` for each observation.
- Likely hazard colliders are discovered once per active Unity scene handle and
  filtered for active/enabled state on each observation.
- Phase 0 diagnostic snapshots are not written while a protocol client is
  active.
- Title-menu save bootstrap may wait for the loaded scene's `Bench Control` FSM
  to reach `Resting`, then send its canonical `GET UP` event once when the save
  is marked `atBench`. The FSM restores the dynamic body/control; the Mod makes
  no direct Transform or Rigidbody write, and no policy STEP is accepted before
  `RUNNING`.

Use `scripts/live_performance_benchmark.py` before and after hot-path changes.
Compare STEP latency percentiles, requests/second, effective fixed
updates/second, and the observed `time_scale`/`fixed_delta_time`; also record
render FPS/frame pacing separately because SPS and visual smoothness answer
different questions.
