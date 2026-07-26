using UnityEngine;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Executes the reset sequence and reports failure via a StatusCode rather than
    /// silently continuing (docs/episode_lifecycle.md §2, PRD §9.3). Performs ready
    /// checks (scene/player/boss) with timeouts.
    /// </summary>
    public sealed class ResetManager
    {
        private const float ReadyStabilitySeconds = 0.1f;

        private readonly SceneController _scene;
        private readonly float _timeoutSeconds;
        private readonly ReadyStabilityGate _readyGate =
            new ReadyStabilityGate(ReadyStabilitySeconds);
        private float _elapsedSeconds;
        private bool _active;

        public ResetManager()
            : this(new SceneController())
        {
        }

        public ResetManager(SceneController scene, float timeoutSeconds = 30.0f)
        {
            if (timeoutSeconds <= 0.0f)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(timeoutSeconds),
                    "timeoutSeconds must be positive");
            }

            _scene = scene;
            _timeoutSeconds = timeoutSeconds;
        }

        public bool IsActive => _active;
        public bool IsComplete { get; private set; }
        public string? LastErrorInfo { get; private set; }

        /// <summary>Begin a reset for the given task; drives EpisodeLifecycle waits.</summary>
        public void BeginReset(int taskId, string? sceneName = null)
        {
            _elapsedSeconds = 0.0f;
            _readyGate.Reset();
            _active = true;
            IsComplete = false;
            LastErrorInfo = null;
            _scene.LoadTaskScene(taskId, sceneName);
        }

        /// <summary>Poll a pending reset; returns a StatusCode (Ok while in progress
        /// transitions, or a terminal error like ResetTimeout/BossNotFound).</summary>
        public HKRL.StatusCode Poll()
        {
            if (!_active)
            {
                return HKRL.StatusCode.Ok;
            }

            if (!_scene.HasValidTarget)
            {
                _scene.CancelPendingLoad();
                _active = false;
                IsComplete = false;
                LastErrorInfo = _scene.DescribeReadiness();
                return HKRL.StatusCode.SceneLoadFailed;
            }

            _elapsedSeconds += Time.unscaledDeltaTime;
            bool isReady =
                _scene.IsSceneReady() && _scene.IsPlayerReady() && _scene.IsBossReady();
            if (_readyGate.Observe(isReady, Time.unscaledTime))
            {
                _active = false;
                IsComplete = true;
                return HKRL.StatusCode.Ok;
            }

            if (_elapsedSeconds <= _timeoutSeconds)
            {
                return HKRL.StatusCode.Ok;
            }

            _active = false;
            IsComplete = false;
            LastErrorInfo = _scene.DescribeReadiness();
            _scene.CancelPendingLoad();
            if (!_scene.IsSceneReady())
            {
                return HKRL.StatusCode.SceneLoadFailed;
            }
            if (!_scene.IsPlayerReady())
            {
                return HKRL.StatusCode.PlayerNotReady;
            }

            return HKRL.StatusCode.BossNotFound;
        }

        public void Clear()
        {
            if (_active && !IsComplete)
            {
                _scene.CancelPendingLoad();
            }
            _active = false;
            IsComplete = false;
            _elapsedSeconds = 0.0f;
            _readyGate.Reset();
        }
    }
}
