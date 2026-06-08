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

        /// <summary>Last lifecycle error, reported in StepResponse.error_code.</summary>
        public HKRL.StatusCode ErrorCode { get; private set; } = HKRL.StatusCode.Ok;

        public bool IsRunning => State == HKRL.LifecycleState.Running;

        public void RequestReset()
        {
            EpisodeId = EpisodeId == ulong.MaxValue ? 1 : EpisodeId + 1;
            ErrorCode = HKRL.StatusCode.Ok;
            State = HKRL.LifecycleState.ResetRequested;
        }

        /// <summary>Advance the state machine one tick. Returns the new state.</summary>
        public HKRL.LifecycleState Tick()
        {
            State = State switch
            {
                HKRL.LifecycleState.ResetRequested => HKRL.LifecycleState.FreezeInput,
                HKRL.LifecycleState.FreezeInput => HKRL.LifecycleState.ClearEvents,
                HKRL.LifecycleState.ClearEvents => HKRL.LifecycleState.LoadScene,
                HKRL.LifecycleState.LoadScene => HKRL.LifecycleState.WaitSceneReady,
                HKRL.LifecycleState.WaitSceneReady => HKRL.LifecycleState.WaitPlayerReady,
                HKRL.LifecycleState.WaitPlayerReady => HKRL.LifecycleState.WaitBossReady,
                HKRL.LifecycleState.WaitBossReady => HKRL.LifecycleState.RestorePlayerState,
                HKRL.LifecycleState.RestorePlayerState => HKRL.LifecycleState.ClearProjectiles,
                HKRL.LifecycleState.ClearProjectiles => HKRL.LifecycleState.Countdown,
                HKRL.LifecycleState.Countdown => HKRL.LifecycleState.Running,
                HKRL.LifecycleState.Terminating => HKRL.LifecycleState.ReportDone,
                HKRL.LifecycleState.ReportDone => HKRL.LifecycleState.Cleanup,
                HKRL.LifecycleState.Cleanup => HKRL.LifecycleState.Idle,
                _ => State
            };
            return State;
        }

        public void Fail(HKRL.StatusCode errorCode)
        {
            ErrorCode = errorCode == HKRL.StatusCode.Ok
                ? HKRL.StatusCode.InternalError
                : errorCode;
            State = HKRL.LifecycleState.ReportDone;
        }

        /// <summary>Route death/win/scene-change into TERMINATING.</summary>
        public void RequestTerminate()
        {
            if (State == HKRL.LifecycleState.Idle || State == HKRL.LifecycleState.Cleanup)
            {
                return;
            }

            State = HKRL.LifecycleState.Terminating;
        }
    }
}
