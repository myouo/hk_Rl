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
        private float _baseFixedDelta = Time.fixedDeltaTime;

        /// <summary>Set the simulation time scale (1.0 = normal).</summary>
        public void SetTimeScale(float scale)
        {
            // TODO(phase-2): set Time.timeScale and adjust fixedDeltaTime consistently.
        }

        public void Pause()
        {
            // TODO(phase-1): Time.timeScale = 0.
        }

        public void Resume()
        {
            // TODO(phase-1): restore previous time scale.
        }
    }
}
