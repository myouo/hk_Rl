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

At runtime the mod starts the TCP environment server from the persistent
`HKRLDriver`. Defaults are `127.0.0.1:5555`; set these environment variables
before launching Hollow Knight to line up live smoke, evaluator, or worker
processes with a specific game instance:

```bash
export HKRL_HOST=127.0.0.1
export HKRL_PORT=5555
export HKRL_AUTH_TOKEN=dev-secret   # optional; enables TCP env auth
```

Python env clients (`check_env.py`, local training, workers, evaluators) send
the same non-empty token automatically when it is present. Sending the auth
preface is harmless when mod auth is disabled, so local smoke commands do not
need a separate config edit just to match a token-enabled mod.
The TCP server treats each client connection as an isolated env session: when a
worker/evaluator disconnects or reconnects, the network thread clears queued
request/response frames and detects half-closed sockets before accepting the
next client. The Python side still issues a clean `RESET` after reconnect; the
network thread only moves frames and never touches Unity state.

For multi-instance evaluation or worker scale-out on one game machine, launch
each Hollow Knight instance with a distinct `HKRL_PORT` and pass the matching
`--port`/`--ports` or `--env-port` value to the Python entry point. Keep
`HKRL_HOST` loopback unless the deployment is explicitly firewall-scoped to a
trusted LAN.
Use `python scripts/check_env.py --host HOST --port PORT` as the first live
diagnostic: it sends `PING` through the same FlatBuffers/TCP/auth path without
resetting the scene.

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
  MainThread (FixedUpdate):
      dequeue latest action -> apply (InputInjector)
      collect observation, collect reward events
      write StepResponse snapshot to out-queue
  NetworkThread:  dequeue -> send StepResponse
```

The network thread must never touch Unity objects. All game reads/writes happen
in `FixedUpdate` on the main thread. Use `ConcurrentQueue` / ring buffers across
the boundary.

## 6. Robustness (PRD §9.9)

- Wrap every hook in try/catch; log via `Debug/Logger`.
- `schema_version` + mod-version lock; observation health checks.
- Fallback entity fields when an FSM/field is missing on a game update.
- Debug overlay (`Debug/Overlay.cs`) to visually verify entities/hitboxes.
- Unit-test the critical hooks: enter scene, read boss, reset, death, kill.

## 7. Time control (PRD §9.6)

`SimControl` manages `Time.timeScale` and `Time.fixedDeltaTime` to raise SPS
without changing physics semantics inappropriately. `StepController` applies
`PAUSE`, `RESUME`, and `SET_TIMESCALE` commands through `SimControl` on the main
thread. Pair with `action_repeat`.
