# `mod/` — HKRLEnvMod (Hollow Knight environment server)

C# mod (HK Modding API) that turns Hollow Knight into an RL environment server.
Full guide: [`../docs/mod_dev.md`](../docs/mod_dev.md). Framework rationale:
[`../docs/adr/0003-mod-framework-hk-modding-api.md`](../docs/adr/0003-mod-framework-hk-modding-api.md).

The frozen production baseline is **HKRLEnvMod v0.8.0 / protocol schema v6**.
Edit `HKRLEnvMod/Version.props` for a future Mod release; runtime version
reporting and DLL metadata are both derived from that one value.

## The one rule

The **network thread never touches Unity objects.** It only enqueues
`StepRequest` frames and dequeues `StepResponse` frames. All game reads/writes
happen on the **main thread** in `FixedUpdate` via `StepController.FixedTick()`.
At `Time.timeScale == 0`, `HKRLDriver.Update()` may only peek at a queued
recovery command and restore the clock; it does not consume requests or touch
gameplay state.
See [`../docs/protocol.md`](../docs/protocol.md) §6.

## Module map

```text
HKRLEnvMod.cs     Mod entry + FixedUpdate driver
Transport/        TcpServer, MessageCodec (FlatBuffers), Protocol, Heartbeat
Env/              StepController, EpisodeLifecycle, ResetManager, SimControl, SceneController
Observation/      ObservationCollector + Global/Player/Entity/Boss/Projectile/Hazard observers, EntityRegistry
Action/           ActionApplier, InputInjector, ActionMasker, MacroActionScheduler
Rewards/          RewardEventBuffer + Damage/Heal/Death/Scene hooks
Debug/            Overlay, Logger, SnapshotRecorder
Schema/           generated FlatBuffers C# (gitignored; `make gen-schema-cs`)
```

## Building

Compilation needs a local Hollow Knight install + Modding API assemblies. The
`.csproj` references resolve from `$(HollowKnightManaged)` — set this in a local,
uncommitted `Directory.Build.props` or `HKRLEnvMod.csproj.user`:

```xml
<Project>
  <PropertyGroup>
    <HollowKnightManaged>C:\Path\To\Hollow Knight_Data\Managed</HollowKnightManaged>
  </PropertyGroup>
</Project>
```

Then `make gen-schema-cs` to generate `Schema/HKRL.*`, and build with `dotnet
build` / your IDE. Drop the resulting DLL into the game's `Mods/` folder.

For Windows, `scripts/windows/prepare_game_pc.ps1 -BuildAndInstallMod` performs
the pinned schema generation, builds against the installed game's real
assemblies, backs up an existing install, and copies both `HKRLEnvMod.dll` and
`Google.FlatBuffers.dll`. See
[`../docs/windows_ssh_deployment.md`](../docs/windows_ssh_deployment.md).

## Release package

After the real-assembly build is installed and the versioned 44-Boss sweep has
passed against that exact DLL, create and verify the distributable:

```bash
make mod-package
make mod-package-verify
```

The clean, tagged release is written as
`dist/mod/HKRLEnvMod-v0.8.0-schema6.zip` with a sidecar manifest and SHA-256
file. It contains `HKRLEnvMod.dll`, `Google.FlatBuffers.dll`, install
documentation, the license, and binary-bound 44-Boss plus walk-smoothness live
evidence. It deliberately excludes `hkrl-runtime.conf`: create that
permission-restricted, machine-specific file during deployment so
authentication tokens never enter a release archive.

The mod tree now contains the core environment-server components: transport,
step/reset lifecycle, action application/masking, reward-event buffering,
debugging helpers, and player/entity observation plumbing. The action path now
commits real InControl `PlayerAction` state from `InputManager.OnUpdate`; final binary
compatibility and behavioral verification still require a local Hollow Knight +
Modding API setup.
