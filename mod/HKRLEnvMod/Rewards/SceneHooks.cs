namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks scene transitions -> SceneChanged events; relevant for linear multi-boss
    /// flows and for routing unexpected scene changes into TERMINATING. Main-thread.
    /// </summary>
    public static class SceneHooks
    {
        private static RewardEventBuffer? _buffer;

        public static void Install(RewardEventBuffer buffer)
        {
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
        }

        public static void RecordSceneChanged(int fromSceneHash, int toSceneHash)
        {
            _buffer?.Add(
                HKRL.RewardEventKind.SceneChanged,
                entityId: 0,
                amount: 0.0f,
                auxInt: fromSceneHash,
                auxInt2: toSceneHash);
        }
    }
}
