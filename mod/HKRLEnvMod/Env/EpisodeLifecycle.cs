namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Clean episode lifecycle state machine (docs/episode_lifecycle.md, PRD §5.7).
    /// Guarantees no cross-episode reward contamination: events are cleared before
    /// collection, STEP is only valid in RUNNING, and every episode has a unique id.
    /// State enum values mirror HKRL.LifecycleState (schema/hkrl.fbs).
    /// </summary>
    public sealed class EpisodeLifecycle
    {
        /// <summary>Current state (reported in StepResponse.lifecycle_state).</summary>
        public HKRL.LifecycleState State { get; private set; } = HKRL.LifecycleState.Idle;

        /// <summary>Unique id for the current episode.</summary>
        public ulong EpisodeId { get; private set; }

        /// <summary>Advance the state machine one tick. Returns the new state.</summary>
        public HKRL.LifecycleState Tick()
        {
            // TODO(phase-1): drive transitions IDLE -> ... -> RUNNING -> TERMINATING
            // -> REPORT_DONE -> CLEANUP -> IDLE, honoring ready checks + timeouts.
            return State;
        }

        /// <summary>Route death/win/scene-change into TERMINATING.</summary>
        public void RequestTerminate()
        {
            // TODO(phase-1)
        }
    }
}
