using UnityEngine;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Controls Time.timeScale and Time.fixedDeltaTime to raise SPS without breaking
    /// physics semantics (PRD §9.6, docs/metrics.md §3). Pair with action_repeat.
    /// All access is main-thread only.
    /// </summary>
    public sealed class SimControl
    {
        private readonly float _baseFixedDelta = Time.fixedDeltaTime;
        private float _activeScale = 1.0f;
        private bool _paused;

        /// <summary>Set the simulation time scale (1.0 = normal).</summary>
        public void SetTimeScale(float scale)
        {
            if (scale <= 0.0f)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(scale),
                    "time scale must be positive");
            }

            _activeScale = scale;
            _paused = false;
            Time.timeScale = scale;
            Time.fixedDeltaTime = _baseFixedDelta * scale;
        }

        public void Pause()
        {
            if (_paused)
            {
                return;
            }

            _paused = true;
            Time.timeScale = 0.0f;
        }

        public void Resume()
        {
            if (!_paused)
            {
                return;
            }

            _paused = false;
            Time.timeScale = _activeScale;
            Time.fixedDeltaTime = _baseFixedDelta * _activeScale;
        }
    }
}
