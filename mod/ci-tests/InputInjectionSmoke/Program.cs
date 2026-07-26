using System;
using System.Collections.Generic;
using System.IO;
using HKRLEnvMod;
using HKRLEnvMod.Action;
using HKRLEnvMod.Env;
using HKRLEnvMod.Rewards;
using HKRLEnvMod.Transport;
using InControl;
using Modding;

internal static class Program
{
    private static int Main()
    {
        try
        {
            Run();
            Console.WriteLine("InputInjectionSmoke: PASS");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"InputInjectionSmoke: FAIL: {exception}");
            return 1;
        }
    }

    private static void Run()
    {
        HeroActions actions = InputHandler.Instance.inputActions;
        var injector = new InputInjector();
        injector.Apply(
            new PrimitiveInput(
                movementX: -1,
                aimY: 1,
                buttons:
                    (1u << 0)
                    | (1u << 2)
                    | (1u << 3)
                    | (1u << 4)
                    | (1u << 5)
                    | (1u << 6)
                    | (1u << 7)));
        Raise();

        Expect(actions.left.IsPressed, "left");
        Expect(!actions.right.IsPressed, "right");
        Expect(actions.up.IsPressed, "up");
        Expect(!actions.down.IsPressed, "down");
        Expect(actions.jump.IsPressed, "jump");
        Expect(actions.dash.IsPressed, "dash");
        Expect(actions.attack.IsPressed, "attack/nail-art hold");
        Expect(actions.quickCast.IsPressed, "spell quickCast");
        Expect(actions.cast.IsPressed, "focus cast");
        Expect(actions.dreamNail.IsPressed, "dream nail");
        Expect(actions.moveVector.X == -1.0f, "moveVector.X");
        Expect(actions.moveVector.Y == 1.0f, "moveVector.Y");
        Expect(injector.SuccessfulCommitCount == 1, "input commit acknowledgement");

        injector.Apply(new PrimitiveInput(1, -1, 1u << 8));
        Raise();
        Expect(!actions.attack.IsPressed, "nail-art release clears attack");
        Expect(actions.attack.WasReleased, "nail-art release edge");
        Expect(actions.right.IsPressed && !actions.left.IsPressed, "right movement");
        Expect(actions.down.IsPressed && !actions.up.IsPressed, "down aim");
        Expect(actions.moveVector.X == 1.0f, "updated moveVector.X");
        Expect(actions.moveVector.Y == -1.0f, "updated moveVector.Y");

        injector.Dispose();
        ExpectNeutral(actions, "dispose");

        injector.Apply(new PrimitiveInput(-1, 1, 1u << 0));
        Raise();
        ExpectNeutral(actions, "unhook");

        TestActionExecution(actions);
        TestMacroDurations();
        TestCapabilityPolicy();
        TestActionMaskReadiness();
        TestRewardHooks();
        TestSimulationControl();
        TestEpisodeReadiness();
        TestSceneEntryLifecycle();
        TestRuntimeConfiguration();
    }

    private static void TestSceneEntryLifecycle()
    {
        var gameManager = new GameManager();
        Expect(
            gameManager.HasFinishedEnteringScene,
            "scene-entry lifecycle starts completed in regression stub");

        SceneEntryLifecycle.MarkTransitionPending(gameManager);

        Expect(
            !gameManager.HasFinishedEnteringScene,
            "direct transition marks scene entry pending");
    }

    private static void Raise()
    {
        InputManager.RaiseUpdateForTests();
    }

    private static void ExpectNeutral(HeroActions actions, string context)
    {
        Expect(!actions.left.IsPressed, $"{context}: left");
        Expect(!actions.right.IsPressed, $"{context}: right");
        Expect(!actions.up.IsPressed, $"{context}: up");
        Expect(!actions.down.IsPressed, $"{context}: down");
        Expect(!actions.jump.IsPressed, $"{context}: jump");
        Expect(!actions.dash.IsPressed, $"{context}: dash");
        Expect(!actions.attack.IsPressed, $"{context}: attack");
        Expect(!actions.quickCast.IsPressed, $"{context}: quickCast");
        Expect(!actions.cast.IsPressed, $"{context}: cast");
        Expect(!actions.dreamNail.IsPressed, $"{context}: dreamNail");
        Expect(actions.moveVector.X == 0.0f, $"{context}: moveVector.X");
        Expect(actions.moveVector.Y == 0.0f, $"{context}: moveVector.Y");
    }

    private static void Expect(bool condition, string label)
    {
        if (!condition)
        {
            throw new InvalidOperationException($"failed assertion: {label}");
        }
    }

    private static void TestActionExecution(HeroActions actions)
    {
        using var applier = new ActionApplier();
        var jumpAttack = new DecodedAction(
            movementX: 1,
            aimY: 1,
            buttons: 0,
            durationIdx: 0,
            macroId: ActionMasker.MacroJumpAttack);

        applier.Apply(jumpAttack, isNewDecision: true);
        Raise();
        Expect(actions.jump.IsPressed, "macro first primitive");

        applier.Apply(jumpAttack, isNewDecision: false);
        Raise();
        Expect(!actions.jump.IsPressed, "macro does not restart");
        Expect(actions.right.IsPressed, "macro advances movement");
        Expect(actions.attack.IsPressed, "macro advances attack");

        applier.Apply(jumpAttack, isNewDecision: false);
        Raise();
        ExpectNeutral(actions, "completed macro stays neutral within action repeat");

        var pogo = new DecodedAction(
            movementX: 1,
            aimY: 1,
            buttons: 0,
            durationIdx: 0,
            macroId: ActionMasker.MacroPogo);
        applier.Clear();
        for (var i = 0; i < 4; i++)
        {
            applier.Apply(pogo, isNewDecision: i == 0);
            Raise();
            Expect(actions.jump.IsPressed, $"pogo takeoff tick {i + 1}");
            Expect(!actions.attack.IsPressed, $"pogo delays attack tick {i + 1}");
        }
        applier.Apply(pogo, isNewDecision: false);
        Raise();
        Expect(actions.down.IsPressed, "pogo aims down after takeoff");
        Expect(actions.attack.IsPressed, "pogo attacks after takeoff");

        var heldAttack = new DecodedAction(
            movementX: 1,
            aimY: 1,
            buttons: 1u << ActionMasker.ButtonAttack,
            durationIdx: 2,
            macroId: -1);
        applier.Clear();
        applier.Apply(heldAttack);
        Raise();
        Expect(actions.attack.IsPressed, "duration starts");

        applier.SuspendInput();
        Raise();
        ExpectNeutral(actions, "suspended between responses");

        applier.Apply(new DecodedAction(1, 1, 0, 0, -1));
        Raise();
        Expect(actions.attack.IsPressed, "duration resumes on next step");

        var walkAttack = new DecodedAction(
            movementX: 2,
            aimY: 1,
            buttons: 1u << ActionMasker.ButtonAttack,
            durationIdx: 0,
            macroId: -1);
        applier.Clear();
        applier.Apply(walkAttack);
        Raise();
        Expect(actions.right.IsPressed, "walking attack starts movement");
        Expect(actions.attack.IsPressed, "walking attack starts transient button");

        applier.SuspendInput();
        Raise();
        Expect(actions.right.IsPressed, "movement bridges a STEP response");
        Expect(!actions.attack.IsPressed, "transient attack releases at STEP response");
        Expect(actions.attack.WasReleased, "transient attack keeps a release edge");

        applier.Apply(new DecodedAction(1, 1, 0, 0, -1));
        Raise();
        ExpectNeutral(actions, "next policy decision can stop bridged movement");

        applier.Apply(new DecodedAction(2, 1, 0, 0, -1));
        Raise();
        applier.SuspendInput();
        for (var i = 0; i < 9; i++)
        {
            applier.IdleTick();
            Raise();
            Expect(actions.right.IsPressed, $"bridged movement tick {i + 1}");
        }
        applier.IdleTick();
        Raise();
        ExpectNeutral(actions, "stale bridged movement expires");

        applier.Clear();
        Raise();
        ExpectNeutral(actions, "action clear");

        var nailArtHold = new DecodedAction(
            movementX: 1,
            aimY: 1,
            buttons: 1u << ActionMasker.ButtonNailArtHold,
            durationIdx: 0,
            macroId: -1);
        applier.Apply(nailArtHold);
        Raise();
        Expect(actions.attack.IsPressed, "nail-art hold starts");

        applier.SuspendInput();
        Raise();
        Expect(actions.attack.IsPressed, "nail-art hold bridges a STEP response");

        var nailArtRelease = new DecodedAction(
            movementX: 1,
            aimY: 2,
            buttons: 1u << ActionMasker.ButtonNailArtRelease,
            durationIdx: 0,
            macroId: -1);
        applier.Apply(nailArtRelease);
        Raise();
        Expect(!actions.attack.IsPressed, "nail-art release clears bridged hold");
        Expect(actions.attack.WasReleased, "nail-art release keeps physical edge");
        Expect(actions.up.IsPressed, "release direction commits with edge");

        applier.Clear();
        applier.Apply(nailArtHold);
        Raise();
        applier.SuspendInput();
        for (var i = 0; i < 10; i++)
        {
            applier.IdleTick();
            Raise();
        }
        ExpectNeutral(actions, "stale bridged hold expires");

        applier.DisableInput();
        Expect(!applier.InputEnabled, "input disabled outside RUNNING");
        Raise();
        ExpectNeutral(actions, "disabled injection stays neutral");

    }

    private static void TestCapabilityPolicy()
    {
        Expect(
            TrainingCapabilityPolicy.IsAllowed(
                new DecodedAction(2, 1, 1u << ActionMasker.ButtonAttack, 0, -1),
                enableMacroActions: true,
                macroCount: ActionMasker.DefaultMacroCount),
            "ordinary input is allowed");
        Expect(
            !TrainingCapabilityPolicy.IsAllowed(
                new DecodedAction(3, 1, 0, 0, -1),
                enableMacroActions: true,
                macroCount: ActionMasker.DefaultMacroCount),
            "out-of-range movement is rejected");
        Expect(
            !TrainingCapabilityPolicy.IsAllowed(
                new DecodedAction(1, 1, 1u << 12, 0, -1),
                enableMacroActions: true,
                macroCount: ActionMasker.DefaultMacroCount),
            "unknown capability bit is rejected");
        Expect(
            !TrainingCapabilityPolicy.IsAllowed(
                new DecodedAction(1, 1, 0, 0, 0),
                enableMacroActions: false,
                macroCount: 0),
            "disabled macro is rejected");
    }

    private static void TestMacroDurations()
    {
        var scheduler = new MacroActionScheduler();
        for (var macroId = 0; macroId < ActionMasker.DefaultMacroCount; macroId++)
        {
            scheduler.Begin(macroId);
            var emittedTicks = 0;
            while (scheduler.IsActive)
            {
                scheduler.Tick();
                emittedTicks++;
                Expect(emittedTicks <= 120, $"macro {macroId} terminates");
            }

            Expect(
                emittedTicks == MacroActionScheduler.ExpectedDurationTicks(macroId),
                $"macro {macroId} declared duration matches emitted plan");
        }
    }

    private static void TestActionMaskReadiness()
    {
        var masker = new ActionMasker();
        bool[] mask = masker.Compute(
            new PlayerActionState(
                soul: 99,
                onGround: false,
                doubleJumpAvailable: false,
                canDash: false,
                canDreamNail: false,
                canNailCharge: false,
                hasSpell: false));

        Expect(
            !mask[ActionMasker.ButtonOffset + ActionMasker.ButtonJumpTap],
            "airborne spent jump is masked");
        Expect(
            !mask[ActionMasker.ButtonOffset + ActionMasker.ButtonDash],
            "unavailable dash is masked");
        Expect(
            !mask[ActionMasker.ButtonOffset + ActionMasker.ButtonCast],
            "unlearned spell is masked");
        Expect(
            !mask[ActionMasker.ButtonOffset + ActionMasker.ButtonDreamNail],
            "unavailable dream nail is masked");
        Expect(
            !mask[ActionMasker.ButtonOffset + ActionMasker.ButtonNailArtHold],
            "unavailable nail art is masked");
        Expect(
            mask[ActionMasker.ButtonOffset + ActionMasker.ButtonAttack],
            "ordinary attack stays available independently of nail arts");
    }

    private static void TestRewardHooks()
    {
        var rewards = new RewardEventBuffer();
        DamageHooks.Install(rewards);
        DeathHooks.Install(rewards);
        HealHooks.Install(rewards);
        SceneHooks.Install(rewards);

        Expect(ModHooks.RaiseTakeHealthForTests(2) == 2, "damage hook preserves game value");
        Expect(
            ModHooks.RaiseBeforeAddHealthForTests(1) == 1,
            "heal hook preserves game value");
        ModHooks.RaiseBeforePlayerDeadForTests();
        ModHooks.RaiseSceneChangedForTests("GG_Gruz_Mother");

        var events = rewards.Drain();
        Expect(events.Count == 4, "reward hooks emit four events");
        Expect(events[0].Kind == HKRL.RewardEventKind.DamageTaken, "damage event");
        Expect(events[0].Amount == 2.0f, "damage amount");
        Expect(events[1].Kind == HKRL.RewardEventKind.Heal, "heal event");
        Expect(events[2].Kind == HKRL.RewardEventKind.PlayerDeath, "death event");
        Expect(events[3].Kind == HKRL.RewardEventKind.SceneChanged, "scene event");

        DamageHooks.Uninstall();
        DeathHooks.Uninstall();
        HealHooks.Uninstall();
        SceneHooks.Uninstall();
        ModHooks.RaiseTakeHealthForTests(3);
        ModHooks.RaiseBeforeAddHealthForTests(1);
        ModHooks.RaiseBeforePlayerDeadForTests();
        ModHooks.RaiseSceneChangedForTests("GG_Hornet_1");
        Expect(rewards.Count == 0, "reward hooks uninstall");
    }

    private static void TestSimulationControl()
    {
        UnityEngine.Time.fixedDeltaTime = 0.02f;
        TimeController.GenericTimeScale = 1.0f;
        var gameManager = new GameManager();
        var control = new SimControl();

        control.SetTimeScale(3.0f);
        Expect(TimeController.GenericTimeScale == 3.0f, "timescale applies through game");
        Expect(UnityEngine.Time.timeScale == 3.0f, "timescale reaches Unity");
        Expect(
            UnityEngine.Time.fixedDeltaTime == 0.02f,
            "timescale preserves physics step");

        On.GameManager.RaiseSetTimeScaleForTests(gameManager, 0.5f);
        Expect(
            TimeController.GenericTimeScale == 1.5f,
            "game slow motion composes with environment scale");

        control.Pause();
        Expect(control.IsPaused, "simulation marks pause");
        Expect(control.IsSimulationStopped, "pause stops FixedUpdate clock");
        Expect(TimeController.GenericTimeScale == 0.0f, "pause reaches game clock");
        On.GameManager.RaiseSetTimeScaleForTests(gameManager, 1.0f);
        Expect(
            TimeController.GenericTimeScale == 0.0f,
            "game time callbacks cannot escape environment pause");

        control.Resume();
        Expect(!control.IsPaused, "resume clears pause");
        Expect(TimeController.GenericTimeScale == 3.0f, "resume restores active scale");

        // Reproduce a scene/menu transition leaving the Unity clock at zero
        // without SimControl itself having issued PAUSE.
        TimeController.GenericTimeScale = 0.0f;
        control.Resume();
        Expect(
            TimeController.GenericTimeScale == 3.0f,
            "idempotent resume recovers an externally stranded clock");

        var rejectedNonFinite = false;
        try
        {
            control.SetTimeScale(float.NaN);
        }
        catch (ArgumentOutOfRangeException)
        {
            rejectedNonFinite = true;
        }
        Expect(rejectedNonFinite, "non-finite timescale rejected");

        control.Dispose();
        Expect(TimeController.GenericTimeScale == 1.0f, "dispose restores game scale");
        Expect(UnityEngine.Time.timeScale == 1.0f, "dispose restores Unity scale");
        Expect(UnityEngine.Time.fixedDeltaTime == 0.02f, "dispose restores physics step");

        On.GameManager.RaiseSetTimeScaleForTests(gameManager, 0.5f);
        Expect(
            TimeController.GenericTimeScale == 0.5f,
            "dispose removes game timescale hook");
    }

    private static void TestEpisodeReadiness()
    {
        Expect(
            EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "fully restored Hero is ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 0.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "zero-gravity Hero is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: true,
                transitionState: "ENTERING_SCENE",
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "transitioning Hero is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: true,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "control-relinquished Hero is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: true,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "kinematic Hero is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: false,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "non-simulated Hero body is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: false,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true),
            "position-constrained Hero is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: false),
            "Hero without terrain-ingress checks is not ready");
        Expect(
            !EpisodeReadiness.IsHeroReady(
                active: true,
                acceptingInput: true,
                controlRelinquished: false,
                gameplayState: true,
                transitioning: false,
                transitionState: EpisodeReadiness.ReadyHeroTransitionState,
                hasBody: true,
                gravityScale: 1.0f,
                bodyKinematic: false,
                bodySimulated: true,
                positionConstraintsFree: true,
                hasCollider: true,
                colliderEnabled: true,
                tilemapTestActive: true,
                groundedJumpReady: false),
            "grounded Hero whose normal jump is locked is not ready");
        Expect(
            !GodhomeTransitionPolicy.RequiresEntryBootstrap(
                "GG_Gruz_Mother",
                "GG_Gruz_Mother"),
            "same-scene reload skips workshop transition broadcasts");
        Expect(
            GodhomeTransitionPolicy.RequiresEntryBootstrap(
                "GG_Workshop",
                "GG_Gruz_Mother"),
            "first boss entry keeps full Godhome bootstrap");

        var gate = new ReadyStabilityGate(0.1f);
        Expect(!gate.Observe(true, 10.0f), "readiness window starts closed");
        Expect(!gate.Observe(true, 10.05f), "readiness window remains closed");
        Expect(gate.Observe(true, 10.1f), "readiness window opens after stability");
        Expect(!gate.Observe(false, 10.11f), "readiness loss closes window");
        Expect(!gate.Observe(true, 10.2f), "readiness must stabilize again");
    }

    private static void TestRuntimeConfiguration()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"hkrl-runtime-config-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string assemblyPath = Path.Combine(directory, "HKRLEnvMod.dll");
        string configPath = Path.Combine(
            directory,
            RuntimeConfiguration.ConfigFileName);
        var warnings = new List<string>();

        try
        {
            File.WriteAllText(
                configPath,
                "HKRL_HOST=127.0.0.2\n"
                + "HKRL_PORT=6000\n"
                + "HKRL_SAVE_SLOT=2\n"
                + "HKRL_AUTH_TOKEN=file-secret\n"
                + "UNSUPPORTED=value\n");
            var environment = new Dictionary<string, string?>
            {
                [RuntimeConfiguration.HostEnv] = "127.0.0.1",
                [RuntimeConfiguration.PortEnv] = "7000",
                [RuntimeConfiguration.AuthTokenEnv] = "environment-secret",
                [RuntimeConfiguration.SaveSlotEnv] = "3",
            };

            RuntimeConfiguration overridden = RuntimeConfiguration.Load(
                assemblyPath,
                getEnvironment: key => environment.GetValueOrDefault(key),
                warn: warnings.Add);
            Expect(overridden.Host == "127.0.0.1", "runtime env host override");
            Expect(overridden.Port == 7000, "runtime env port override");
            Expect(overridden.SaveSlot == 3, "runtime env save slot override");
            Expect(
                overridden.AuthToken == "environment-secret",
                "runtime env token override");
            Expect(overridden.FileLoaded, "runtime file detected");

            RuntimeConfiguration fromFile = RuntimeConfiguration.Load(
                assemblyPath,
                getEnvironment: _ => null,
                warn: warnings.Add);
            Expect(fromFile.Host == "127.0.0.2", "runtime file host");
            Expect(fromFile.Port == 6000, "runtime file port");
            Expect(fromFile.SaveSlot == 2, "runtime file save slot");
            Expect(fromFile.AuthToken == "file-secret", "runtime file token");
            Expect(
                warnings.TrueForAll(message => !message.Contains("secret")),
                "runtime warnings redact token values");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
