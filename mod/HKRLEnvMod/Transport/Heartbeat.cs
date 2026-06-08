namespace HKRLEnvMod.Transport
{
    /// <summary>
    /// Liveness tracking (docs/protocol.md §5). Tracks last-seen PING/STEP time so
    /// the mod can detect a dead client and reset the env on reconnect.
    /// </summary>
    public sealed class Heartbeat
    {
        public float TimeoutSeconds { get; }

        public Heartbeat(float timeoutSeconds = 5.0f)
        {
            TimeoutSeconds = timeoutSeconds;
        }

        /// <summary>Called from the main thread each tick with current game time.</summary>
        public void Touch(float now)
        {
            // TODO(phase-1): record last activity.
        }

        /// <summary>True if no activity within TimeoutSeconds.</summary>
        public bool IsStale(float now)
        {
            // TODO(phase-1): compare now - lastActivity to TimeoutSeconds.
            return false;
        }
    }
}
