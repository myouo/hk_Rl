namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks player death and boss death -> PlayerDeath / BossKilled events, and
    /// signals the lifecycle to enter TERMINATING (docs/episode_lifecycle.md).
    /// Main-thread.
    /// </summary>
    public static class DeathHooks
    {
        private static RewardEventBuffer? _buffer;

        public static void Install(RewardEventBuffer buffer)
        {
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
        }

        public static void RecordPlayerDeath(int entityId = 0)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.PlayerDeath, entityId);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record player death", exception);
            }
        }

        public static void RecordBossKilled(int entityId)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.BossKilled, entityId);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record boss kill", exception);
            }
        }
    }
}
