using System;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Executes the reset sequence and reports failure via a StatusCode rather than
    /// silently continuing (docs/episode_lifecycle.md §2, PRD §9.3). Performs ready
    /// checks (scene/player/boss) with timeouts.
    /// </summary>
    public sealed class ResetManager
    {
        /// <summary>Begin a reset for the given task; drives EpisodeLifecycle waits.</summary>
        public void BeginReset(int taskId)
        {
            // TODO(phase-1): freeze input, clear events, load scene, wait scene/player/
            // boss ready, restore player state, clear projectiles, countdown.
            throw new NotImplementedException();
        }

        /// <summary>Poll a pending reset; returns a StatusCode (Ok while in progress
        /// transitions, or a terminal error like ResetTimeout/BossNotFound).</summary>
        public HKRL.StatusCode Poll()
        {
            // TODO(phase-1)
            return HKRL.StatusCode.Ok;
        }
    }
}
