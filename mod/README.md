# `mod/` — HKRLEnvMod (Hollow Knight environment server)

C# mod (HK Modding API) that turns Hollow Knight into an RL environment server.
Full guide: [`../docs/mod_dev.md`](../docs/mod_dev.md). Framework rationale:
[`../docs/adr/0003-mod-framework-hk-modding-api.md`](../docs/adr/0003-mod-framework-hk-modding-api.md).

## The one rule

The **network thread never touches Unity objects.** It only enqueues
`StepRequest` frames and dequeues `StepResponse` frames. All game reads/writes
happen on the **main thread** in `FixedUpdate` via `StepController.FixedTick()`.
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

The mod tree now contains the core environment-server components: transport,
step/reset lifecycle, action application/masking, reward-event buffering,
debugging helpers, and player/entity observation plumbing. Full compile and
behavioral verification still require a local Hollow Knight + Modding API setup.
