using System;
using UnityEngine;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Controls Hollow Knight's game time scale to raise SPS without changing the
    /// physics step (PRD §9.6, docs/metrics.md §3). Pair with action_repeat.
    /// All access is main-thread only.
    /// </summary>
    public sealed class SimControl : IDisposable
    {
        private const float GamePauseThreshold = 0.01f;

        private readonly float _baseFixedDelta;
        private readonly float _initialUnityTimeScale;
        private readonly float _initialGameTimeScale;
        private float _activeScale = 1.0f;
        private bool _paused;
        private bool _disposed;

        public SimControl()
        {
            _baseFixedDelta = IsPositiveFinite(Time.fixedDeltaTime)
                ? Time.fixedDeltaTime
                : 0.02f;
            _initialUnityTimeScale = IsNonNegativeFinite(Time.timeScale)
                ? Time.timeScale
                : 1.0f;
            _initialGameTimeScale = IsNonNegativeFinite(TimeController.GenericTimeScale)
                ? TimeController.GenericTimeScale
                : _initialUnityTimeScale;

            // Hollow Knight regularly calls GameManager.SetTimeScale for hit-stop,
            // menus, and scene transitions. Keep the environment multiplier in
            // that path instead of fighting it with a raw Time.timeScale write.
            On.GameManager.SetTimeScale_float += OnGameManagerSetTimeScale;
        }

        public bool IsPaused => _paused;

        /// <summary>
        /// True when Unity will not schedule FixedUpdate. The driver's unscaled
        /// control pump uses this only to wake recovery commands.
        /// </summary>
        public bool IsSimulationStopped =>
            !IsNonNegativeFinite(Time.timeScale) || Time.timeScale <= 0.0f;

        /// <summary>Set the simulation time scale (1.0 = normal).</summary>
        public void SetTimeScale(float scale)
        {
            ThrowIfDisposed();
            if (!IsPositiveFinite(scale))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(scale),
                    "time scale must be positive and finite");
            }

            _activeScale = scale;
            _paused = false;
            ApplyGameTimeScale(1.0f);
        }

        public void Pause()
        {
            ThrowIfDisposed();
            _paused = true;
            ApplyGameTimeScale(0.0f);
        }

        public void Resume()
        {
            ThrowIfDisposed();

            // Always re-assert the running scale. Reset/scene transitions can
            // leave the game's global scale at zero even when this instance did
            // not issue PAUSE, so an early-return here creates a permanent freeze.
            _paused = false;
            ApplyGameTimeScale(1.0f);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            On.GameManager.SetTimeScale_float -= OnGameManagerSetTimeScale;
            _paused = false;

            // Never strand the game in the environment's paused/accelerated
            // state when the driver is destroyed or the mod is unloaded.
            TimeController.GenericTimeScale = _initialGameTimeScale;
            Time.timeScale = _initialUnityTimeScale;
            Time.fixedDeltaTime = _baseFixedDelta;
        }

        private void OnGameManagerSetTimeScale(
            On.GameManager.orig_SetTimeScale_float orig,
            global::GameManager self,
            float requestedScale)
        {
            try
            {
                if (_disposed)
                {
                    orig(self, requestedScale);
                    return;
                }

                ApplyGameTimeScale(requestedScale);
            }
            catch (Exception exception)
            {
                // Hook bodies must not break Hollow Knight's time controller.
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to apply HKRL game time scale; falling back to the game",
                    exception);
                orig(self, requestedScale);
            }
        }

        private void ApplyGameTimeScale(float requestedScale)
        {
            float gameScale =
                IsPositiveFinite(requestedScale) && requestedScale > GamePauseThreshold
                    ? requestedScale
                    : 0.0f;
            float effectiveScale = _paused ? 0.0f : gameScale * _activeScale;
            if (!IsNonNegativeFinite(effectiveScale))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(requestedScale),
                    "effective time scale must be finite");
            }

            // TimeController is Hollow Knight's supported game-time path. Keep
            // fixedDeltaTime in simulated seconds so acceleration increases SPS
            // without making physics/collision integration coarser.
            TimeController.GenericTimeScale = effectiveScale;
            Time.fixedDeltaTime = _baseFixedDelta;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(SimControl));
            }
        }

        private static bool IsPositiveFinite(float value)
        {
            return value > 0.0f && !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static bool IsNonNegativeFinite(float value)
        {
            return value >= 0.0f && !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
