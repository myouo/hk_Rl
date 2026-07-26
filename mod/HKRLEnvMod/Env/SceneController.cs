using UnityEngine;
using UnityEngine.SceneManagement;
using HKRLEnvMod.Observation;
using System.Text;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Loads/validates Godhome boss scenes for a task (e.g. GG_Hornet_1) and reports
    /// readiness. Wraps GameManager/BossSceneController interactions. Main-thread only.
    /// </summary>
    public sealed class SceneController
    {
        private const string GodhomeEntryGateName = "door_dreamEnter";
        private const string GodhomeWorkshopSceneName = "GG_Workshop";

        private readonly int _saveSlot;
        private string _targetSceneName = "GG_Gruz_Mother";
        private bool _waitingForSaveLoad;
        private bool _targetLoadRequested;
        private bool _loadFailed;
        private bool _ownsBossSetupEvent;
        private bool _saveBootstrapReleaseAttempted;
        private PlayMakerFSM? _loadedBenchFsm;
        private bool _requiresSceneInstanceChange;
        private int _sourceSceneHandle;

        public SceneController(int saveSlot = 1)
        {
            if (saveSlot < 1 || saveSlot > 4)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(saveSlot),
                    "saveSlot must be in [1, 4]");
            }

            _saveSlot = saveSlot;
        }

        public int CurrentTaskId { get; private set; }
        public string TargetSceneName => _targetSceneName;
        public bool HasValidTarget =>
            !_loadFailed && !string.IsNullOrEmpty(_targetSceneName);

        /// <summary>Request loading the scene/arena for a task id.</summary>
        public void LoadTaskScene(int taskId, string? sceneName = null)
        {
            ReleaseOwnedBossSetupEvent();
            CurrentTaskId = taskId;
            _targetSceneName = ResolveSceneName(taskId, sceneName);
            _waitingForSaveLoad = false;
            _targetLoadRequested = false;
            _loadFailed = false;
            _ownsBossSetupEvent = false;
            _saveBootstrapReleaseAttempted = false;
            _loadedBenchFsm = null;
            _requiresSceneInstanceChange = false;
            _sourceSceneHandle = 0;
            if (!HasValidTarget)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    $"Unknown HKRL task id {taskId}; reset will fail scene readiness.");
                return;
            }

            try
            {
                GameManager? gameManager = GameManager.instance;
                if (gameManager == null)
                {
                    FailLoad("GameManager is unavailable.");
                    return;
                }

                // A Unity SceneManager jump from Menu_Title creates a boss scene
                // without the persistent Hero and breaks GameManager.SetupHeroRefs.
                // Bootstrap a configured, Godhome-capable save first, then enter
                // the arena through Hollow Knight's own transition pipeline.
                if (HeroController.SilentInstance == null && gameManager.IsMenuScene())
                {
                    _waitingForSaveLoad = true;
                    gameManager.LoadGameFromUI(_saveSlot);
                    global::HKRLEnvMod.Debug.Logger.Info(
                        $"Loading save slot {_saveSlot} before HKRL task scene "
                        + $"{_targetSceneName}.");
                    return;
                }

                BeginTargetSceneTransition(gameManager);
            }
            catch (System.Exception exception)
            {
                FailLoad($"Failed to load HKRL task scene {_targetSceneName}.", exception);
            }
        }

        public bool IsSceneReady()
        {
            if (!HasValidTarget)
            {
                return false;
            }

            ProgressPendingLoad();
            Scene scene =
                UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            return scene.isLoaded
                && scene.name == _targetSceneName
                && (!_requiresSceneInstanceChange || scene.handle != _sourceSceneHandle);
        }

        public bool IsPlayerReady()
        {
            global::HeroController? hero = global::HeroController.SilentInstance;
            GameManager? gameManager = GameManager.instance;
            return gameManager != null
                && !gameManager.IsInSceneTransition
                && !gameManager.IsLoadingSceneTransition
                && PlayerObserver.IsReadyForControl(hero);
        }

        public bool IsBossReady()
        {
            if (!IsSceneReady())
            {
                return false;
            }

            BossSceneController? controller = BossSceneController.Instance;
            var configuredBosses = BossLocator.FindConfiguredBosses();
            var bosses = BossLocator.FindActiveBosses();
            return controller != null
                && controller.HasTransitionedIn
                && configuredBosses.Count > 0
                && bosses.Count > 0;
        }

        public string DescribeReadiness()
        {
            Scene scene =
                UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            return "reset "
                + $"target={_targetSceneName}, active_scene={scene.name}, "
                + $"save_slot={_saveSlot}, waiting_for_save={_waitingForSaveLoad}, "
                + $"save_bootstrap_release_attempted={_saveBootstrapReleaseAttempted}, "
                + "save_bootstrap_fsm_state="
                + $"{_loadedBenchFsm?.ActiveStateName ?? "<unavailable>"}, "
                + $"transition_requested={_targetLoadRequested}, "
                + $"scene_handle={scene.handle}, source_scene_handle={_sourceSceneHandle}, "
                + $"scene_ready={IsSceneReady()}, "
                + $"player_ready={IsPlayerReady()}, "
                + $"{PlayerObserver.DescribeControlReadiness(HeroController.SilentInstance)}, "
                + $"boss_ready={IsBossReady()}, "
                + DescribeBossLifecycle();
        }

        /// <summary>
        /// Release only the setup callback installed by this SceneController.
        /// RESET timeouts can occur before a BossSceneController consumes the
        /// static callback; carrying it into the next task contaminates that
        /// task's episode lifecycle.
        /// </summary>
        public void CancelPendingLoad()
        {
            ReleaseOwnedBossSetupEvent();
            _waitingForSaveLoad = false;
            _targetLoadRequested = false;
        }

        private static string DescribeBossLifecycle(global::HealthManager? boss = null)
        {
            try
            {
                var activeBosses = BossLocator.FindActiveBosses();
                var configuredBosses = BossLocator.FindConfiguredBosses();
                if (boss == null)
                {
                    if (activeBosses.Count > 0)
                    {
                        boss = activeBosses[0];
                    }
                    else if (configuredBosses.Count > 0)
                    {
                        boss = configuredBosses[0];
                    }
                }

                var builder = new StringBuilder();
                bool first = true;
                builder.Append("boss_lifecycle={configured_count=")
                    .Append(configuredBosses.Count)
                    .Append(", active_count=")
                    .Append(activeBosses.Count);
                if (boss != null && boss.gameObject != null)
                {
                    var gameObject = boss.gameObject;
                    var body = gameObject.GetComponent<Rigidbody2D>();
                    builder.Append(", name=")
                        .Append(gameObject.name)
                        .Append(", active=")
                        .Append(gameObject.activeInHierarchy)
                        .Append(", body_kinematic=")
                        .Append(body?.isKinematic.ToString() ?? "<missing>")
                        .Append(", body_simulated=")
                        .Append(body?.simulated.ToString() ?? "<missing>")
                        .Append(", velocity=")
                        .Append(body == null ? "<missing>" : body.velocity.ToString());
                    builder.Append(", fsms=[");
                    AppendFsmStates(
                        builder,
                        gameObject.GetComponents<PlayMakerFSM>(),
                        ref first);
                    foreach (Transform child in
                             gameObject.GetComponentsInChildren<Transform>(true))
                    {
                        if (child == null || child.gameObject == gameObject)
                        {
                            continue;
                        }

                        AppendFsmStates(
                            builder,
                            child.gameObject.GetComponents<PlayMakerFSM>(),
                            ref first);
                    }
                }
                else
                {
                    builder.Append(", name=<unavailable>, fsms=[");
                }

                Scene scene =
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene();
                foreach (PlayMakerFSM fsm in UnityEngine.Object.FindObjectsOfType<PlayMakerFSM>())
                {
                    if (fsm == null
                        || fsm.gameObject == null
                        || fsm.gameObject.scene.handle != scene.handle
                        || (!string.Equals(
                                fsm.gameObject.name,
                                "Battle Scene",
                                System.StringComparison.Ordinal)
                            && !string.Equals(
                                fsm.gameObject.name,
                                "Battle Range",
                                System.StringComparison.Ordinal)
                            && !string.Equals(
                                fsm.gameObject.name,
                                "Knight Dream Arrival",
                                System.StringComparison.Ordinal)
                            && !string.Equals(
                                fsm.gameObject.name,
                                "Dream Entry",
                                System.StringComparison.Ordinal)
                            && !string.Equals(
                                fsm.gameObject.name,
                                "Start Range",
                                System.StringComparison.Ordinal)))
                    {
                        continue;
                    }

                    AppendFsmState(builder, fsm, ref first);
                }

                return builder.Append("]}").ToString();
            }
            catch (System.Exception exception)
            {
                return $"boss_lifecycle=<diagnostic failed: {exception.Message}>";
            }
        }

        private static void AppendFsmStates(
            StringBuilder builder,
            PlayMakerFSM[] fsms,
            ref bool first)
        {
            foreach (PlayMakerFSM fsm in fsms)
            {
                AppendFsmState(builder, fsm, ref first);
            }
        }

        private static void AppendFsmState(
            StringBuilder builder,
            PlayMakerFSM? fsm,
            ref bool first)
        {
            if (fsm == null || fsm.gameObject == null)
            {
                return;
            }

            if (!first)
            {
                builder.Append("; ");
            }

            first = false;
            builder.Append(fsm.gameObject.name)
                .Append(".")
                .Append(fsm.FsmName)
                .Append("=")
                .Append(fsm.ActiveStateName)
                .Append("(enabled=")
                .Append(fsm.enabled)
                .Append(", transitions=")
                .Append(DescribeActiveTransitions(fsm))
                .Append(")");
        }

        private static string DescribeActiveTransitions(PlayMakerFSM fsm)
        {
            try
            {
                object? runtimeFsm = fsm.GetType()
                    .GetProperty("Fsm")
                    ?.GetValue(fsm, null);
                object? activeState = runtimeFsm?.GetType()
                    .GetProperty("ActiveState")
                    ?.GetValue(runtimeFsm, null);
                object? transitions = activeState?.GetType()
                    .GetProperty("Transitions")
                    ?.GetValue(activeState, null);
                if (transitions is not System.Collections.IEnumerable enumerable)
                {
                    return "[]";
                }

                var builder = new StringBuilder("[");
                bool first = true;
                foreach (object transition in enumerable)
                {
                    if (!first)
                    {
                        builder.Append(", ");
                    }
                    first = false;
                    string eventName = transition.GetType()
                        .GetProperty("EventName")
                        ?.GetValue(transition, null)
                        ?.ToString() ?? string.Empty;
                    string toState = transition.GetType()
                        .GetProperty("ToState")
                        ?.GetValue(transition, null)
                        ?.ToString() ?? string.Empty;
                    builder.Append(eventName).Append("->").Append(toState);
                }

                return builder.Append("]").ToString();
            }
            catch (System.Exception exception)
            {
                return $"<unavailable:{exception.Message}>";
            }
        }

        private void ProgressPendingLoad()
        {
            if (!_waitingForSaveLoad || _targetLoadRequested || _loadFailed)
            {
                return;
            }

            try
            {
                GameManager? gameManager = GameManager.instance;
                HeroController? hero = HeroController.SilentInstance;
                Scene activeScene =
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene();
                if (gameManager == null
                    || hero == null
                    || hero.gameObject == null
                    || !hero.gameObject.activeInHierarchy
                    || !activeScene.isLoaded
                    || string.IsNullOrEmpty(activeScene.name)
                    || activeScene.name.StartsWith("Menu_", System.StringComparison.Ordinal)
                    || gameManager.IsInSceneTransition
                    || gameManager.IsLoadingSceneTransition)
                {
                    return;
                }

                if (!PlayerObserver.IsReadyForControl(hero))
                {
                    // A save may legitimately load with the Hero seated at a
                    // bench. Bench respawn deliberately relinquishes control and
                    // disables gravity, so PlayerAction input cannot release it.
                    // Use Hollow Knight's own control-restoration API once and
                    // wait for the strict readiness gate before transitioning.
                    TryReleaseLoadedBench();
                    return;
                }

                BeginTargetSceneTransition(gameManager);
            }
            catch (System.Exception exception)
            {
                FailLoad(
                    $"Failed to continue into HKRL task scene {_targetSceneName}.",
                    exception);
            }
        }

        private void BeginTargetSceneTransition(GameManager gameManager)
        {
            // Always transition, including same-scene resets. Re-entering through
            // the real game pipeline recreates boss/projectiles and preserves the
            // persistent Hero and GameManager references.
            _waitingForSaveLoad = false;
            _targetLoadRequested = true;
            Scene activeScene =
                UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            _sourceSceneHandle = activeScene.handle;
            bool sameSceneReload =
                activeScene.isLoaded && activeScene.name == _targetSceneName;
            _requiresSceneInstanceChange = sameSceneReload;
            PrepareBossSceneSetup();
            // Both first entry and same-scene reloads use the Godhome transition
            // prefab, which resolves the destination through this shared value.
            StaticVariableList.SetValue<string>("bossSceneToLoad", _targetSceneName);
            if (GodhomeTransitionPolicy.RequiresEntryBootstrap(
                    activeScene.name,
                    _targetSceneName))
            {
                PrepareGodhomeChallenge(gameManager);
            }
            else
            {
                PrepareSameSceneReload();
            }

            // Direct BeginSceneTransition calls do not clear the game's private
            // scene-entry completion flag. Mark this transition pending before
            // the replacement scene's WaitForFinishedEnteringScene actions can
            // observe the stale completion value from the previous episode.
            SceneEntryLifecycle.MarkTransitionPending(gameManager);
            gameManager.BeginSceneTransition(
                new GameManager.SceneLoadInfo
                {
                    SceneName = _targetSceneName,
                    EntryGateName = GodhomeEntryGateName,
                    EntryDelay = 0.0f,
                    Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
                    PreventCameraFadeOut = true,
                    WaitForSceneTransitionCameraFade = false,
                    AlwaysUnloadUnusedAssets = false,
                });
        }

        private static void PrepareSameSceneReload()
        {
            HeroController? hero = HeroController.SilentInstance;
            if (hero == null)
            {
                throw new System.InvalidOperationException(
                    "A persistent HeroController is required before reloading "
                    + "a Godhome boss scene.");
            }

            // Match GodhomeQoL's proven fast-reload Hero preparation. The
            // direct replacement scene still owns its normal transition-in
            // lifecycle; this only restores the persistent Hero before unload.
            hero.MaxHealth();
            hero.ClearMPSendEvents();
            hero.EnterWithoutInput(true);
            hero.AcceptInput();
        }

        private void PrepareGodhomeChallenge(GameManager gameManager)
        {
            HeroController? hero = HeroController.SilentInstance;
            PlayerData? playerData = PlayerData.instance;
            if (hero == null || playerData == null)
            {
                throw new System.InvalidOperationException(
                    "A persistent HeroController and PlayerData are required "
                    + "before entering a Godhome boss scene.");
            }

            // Direct transitions use only non-cinematic Hall bookkeeping.
            // Do not drive a workshop statue FSM or synthesize Boss events here;
            // the destination scene owns its normal transition-in lifecycle.
            playerData.dreamReturnScene = GodhomeWorkshopSceneName;
            hero.ClearMPSendEvents();
            gameManager.TimePasses();
            gameManager.ResetSemiPersistentItems();
        }

        private void PrepareBossSceneSetup()
        {
            if (BossSceneController.SetupEvent != null)
            {
                throw new System.InvalidOperationException(
                    "BossSceneController.SetupEvent is already owned by another challenge flow.");
            }

            BossSceneController.SetupEvent = ConfigureBossScene;
            _ownsBossSetupEvent = true;
        }

        private void ReleaseOwnedBossSetupEvent()
        {
            if (_ownsBossSetupEvent && BossSceneController.SetupEvent != null)
            {
                BossSceneController.SetupEvent = null;
            }

            _ownsBossSetupEvent = false;
        }

        private static void ConfigureBossScene(BossSceneController controller)
        {
            // Mirrors the legitimate Hall of Gods BossChallengeUI bootstrap:
            // providing SetupEvent keeps doTransition enabled, lets Awake call
            // Setup(), and causes Start() to emit the GG TRANSITION IN event.
            controller.BossLevel = 0;
            controller.DreamReturnEvent = "DREAM RETURN";
            controller.OnBossSceneComplete += controller.DoDreamReturn;
        }

        private void FailLoad(string message, System.Exception? exception = null)
        {
            if (exception == null)
            {
                global::HKRLEnvMod.Debug.Logger.Error(message);
            }
            else
            {
                global::HKRLEnvMod.Debug.Logger.Error(message, exception);
            }

            _waitingForSaveLoad = false;
            _targetLoadRequested = false;
            _loadFailed = true;
            ReleaseOwnedBossSetupEvent();
        }

        private void TryReleaseLoadedBench()
        {
            if (_saveBootstrapReleaseAttempted)
            {
                return;
            }

            PlayerData? playerData = PlayerData.instance;
            if (playerData == null || !playerData.atBench)
            {
                return;
            }

            if (_loadedBenchFsm == null)
            {
                Scene activeScene =
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene();
                PlayMakerFSM[] fsms =
                    UnityEngine.Object.FindObjectsOfType<PlayMakerFSM>();
                foreach (PlayMakerFSM fsm in fsms)
                {
                    if (fsm == null
                        || fsm.gameObject == null
                        || fsm.gameObject.scene.handle != activeScene.handle
                        || !string.Equals(
                            fsm.FsmName,
                            "Bench Control",
                            System.StringComparison.Ordinal))
                    {
                        continue;
                    }

                    _loadedBenchFsm = fsm;
                    break;
                }
            }

            if (_loadedBenchFsm == null
                || !string.Equals(
                    _loadedBenchFsm.ActiveStateName,
                    "Resting",
                    System.StringComparison.Ordinal))
            {
                return;
            }

            // GET UP is the vanilla transition out of the Resting state. Wait
            // until that state is actually active because PlayMaker discards
            // events that have no transition in the current initialization
            // state.
            _saveBootstrapReleaseAttempted = true;
            _loadedBenchFsm.SendEvent("GET UP");
            global::HKRLEnvMod.Debug.Logger.Info(
                "Requested loaded bench release through Bench Control.GET UP "
                + "before the HKRL boss-scene transition.");
        }

        private static string ResolveSceneName(int taskId, string? sceneName)
        {
            string configuredSceneName = sceneName ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(configuredSceneName))
            {
                return configuredSceneName.Trim();
            }

            return taskId switch
            {
                0 => "GG_Gruz_Mother",
                1 => "GG_Hornet_1",
                2 => "GG_Mantis_Lords",
                _ => string.Empty
            };
        }
    }
}
