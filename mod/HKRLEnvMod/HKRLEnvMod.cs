using System;
using HKRLEnvMod.Action;
using HKRLEnvMod.Env;
using HKRLEnvMod.Observation;
using HKRLEnvMod.Rewards;
using HKRLEnvMod.Transport;
using Modding;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKRLEnvMod
{
    /// <summary>
    /// Mod entry point (HK Modding API). Boots the environment server: starts the
    /// transport, wires the StepController into the Unity FixedUpdate loop, and
    /// installs reward hooks. See docs/architecture.md and docs/mod_dev.md.
    ///
    /// THREADING CONTRACT (docs/mod_dev.md §5, PRD §5.3): the network thread NEVER
    /// touches Unity objects. It only enqueues StepRequests and dequeues
    /// StepResponses. All game access happens on the main thread in FixedUpdate.
    /// </summary>
    public class HKRLEnvMod : Mod
    {
        private const string DriverObjectName = "HKRL Environment Driver";

        public static HKRLEnvMod? Instance { get; private set; }

        public override string GetVersion() => "0.1.0";

        public HKRLEnvMod() : base("HKRL Environment Server") { }

        /// <summary>Called by the Modding API on load. Initialize subsystems here.</summary>
        public override void Initialize()
        {
            Instance = this;
            global::HKRLEnvMod.Debug.Logger.Info($"Initializing HKRL Environment Server v{GetVersion()}.");
            EnsureDriver();
        }

        private static void EnsureDriver()
        {
            GameObject? driverObject = GameObject.Find(DriverObjectName);

            if (driverObject == null)
            {
                driverObject = new GameObject(DriverObjectName);
                HKRLDriver driver = driverObject.AddComponent<HKRLDriver>();
                UnityEngine.Object.DontDestroyOnLoad(driverObject);
                driver.Configure();
                global::HKRLEnvMod.Debug.Logger.Info("Phase 0 driver attached.");
                return;
            }

            HKRLDriver? existingDriver = driverObject.GetComponent<HKRLDriver>();
            if (existingDriver == null)
            {
                existingDriver = driverObject.AddComponent<HKRLDriver>();
                global::HKRLEnvMod.Debug.Logger.Warn(
                    "Phase 0 driver object existed without HKRLDriver; component attached.");
            }

            UnityEngine.Object.DontDestroyOnLoad(driverObject);
            existingDriver.Configure();
        }
    }

    /// <summary>
    /// MonoBehaviour driver that forwards Unity's FixedUpdate to the StepController.
    /// Created on a persistent GameObject by the mod during Initialize.
    /// </summary>
    public class HKRLDriver : MonoBehaviour
    {
        private const float Phase0LogIntervalSeconds = 2.0f;

        private float _nextPhase0LogTime;
        private TcpServer? _server;
        private StepController? _stepController;
        private bool _configured;

        public void Configure(string host = "127.0.0.1", int port = 5555)
        {
            if (_configured)
            {
                return;
            }

            try
            {
                _server = new TcpServer(host, port);
                RewardEventBuffer rewards = new RewardEventBuffer();
                DamageHooks.Install(rewards);
                DeathHooks.Install(rewards);
                HealHooks.Install(rewards);
                SceneHooks.Install(rewards);
                _stepController = new StepController(
                    _server,
                    new ActionApplier(),
                    rewards,
                    new EpisodeLifecycle(),
                    new ActionMasker(),
                    new Heartbeat(),
                    new ObservationCollector());
                _server.Start();
                _configured = true;
                global::HKRLEnvMod.Debug.Logger.Info(
                    $"HKRL TCP environment server listening on {host}:{port}.");
            }
            catch (Exception exception)
            {
                _server?.Dispose();
                _server = null;
                _stepController = null;
                _configured = false;
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to start HKRL TCP environment server",
                    exception);
            }
        }

        private void Awake()
        {
            UnityEngine.Object.DontDestroyOnLoad(gameObject);
            global::HKRLEnvMod.Debug.Logger.Info("HKRL FixedUpdate driver ready.");
        }

        private void FixedUpdate()
        {
            // MAIN THREAD ONLY. Dequeue latest action, apply, collect obs+events,
            // enqueue StepResponse. Never block on the network here.
            _stepController?.FixedTick();
            LogPhase0Snapshot();
        }

        private void OnDestroy()
        {
            _server?.Dispose();
            _server = null;
            _stepController = null;
            _configured = false;
        }

        private void LogPhase0Snapshot()
        {
            float now = Time.unscaledTime;
            if (now < _nextPhase0LogTime)
            {
                return;
            }

            _nextPhase0LogTime = now + Phase0LogIntervalSeconds;

            try
            {
                string sceneName = SceneManager.GetActiveScene().name;
                global::HeroController hero = global::HeroController.instance;

                if (hero == null)
                {
                    global::HKRLEnvMod.Debug.Logger.Info(
                        $"Phase 0 snapshot: scene={sceneName}, player=<unavailable>.");
                    return;
                }

                Vector3 position = hero.transform.position;
                global::HKRLEnvMod.Debug.Logger.Info(
                    "Phase 0 snapshot: "
                    + $"scene={sceneName}, "
                    + $"player=({position.x:F2}, {position.y:F2}, {position.z:F2}).");
            }
            catch (Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to collect Phase 0 snapshot",
                    exception);
            }
        }
    }
}
