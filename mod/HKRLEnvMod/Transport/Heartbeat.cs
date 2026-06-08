namespace HKRLEnvMod.Transport
{
    /// <summary>
    /// Liveness tracking (docs/protocol.md §5). Tracks last-seen PING/STEP time so
    /// the mod can detect a dead client and reset the env on reconnect.
    /// </summary>
    public sealed class Heartbeat
    {
        private float _lastActivity;
        private bool _hasActivity;

        public float TimeoutSeconds { get; }

        public Heartbeat(float timeoutSeconds = 5.0f)
        {
            if (timeoutSeconds <= 0.0f)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(timeoutSeconds),
                    "timeoutSeconds must be positive");
            }

            TimeoutSeconds = timeoutSeconds;
        }

        /// <summary>Called from the main thread each tick with current game time.</summary>
        public void Touch(float now)
        {
            _lastActivity = now;
            _hasActivity = true;
        }

        /// <summary>True if no activity within TimeoutSeconds.</summary>
        public bool IsStale(float now)
        {
            return _hasActivity && now - _lastActivity > TimeoutSeconds;
        }
    }
}
