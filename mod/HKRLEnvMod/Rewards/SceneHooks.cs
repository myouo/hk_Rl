namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks scene transitions -> SceneChanged events; relevant for linear multi-boss
    /// flows and for routing unexpected scene changes into TERMINATING. Main-thread.
    /// </summary>
    public static class SceneHooks
    {
        private static RewardEventBuffer? _buffer;
        private static string _previousScene = string.Empty;

        public static void Install(RewardEventBuffer buffer)
        {
            Uninstall();
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
            _previousScene =
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().name
                ?? string.Empty;
            Modding.ModHooks.SceneChanged += OnSceneChanged;
        }

        public static void Uninstall()
        {
            Modding.ModHooks.SceneChanged -= OnSceneChanged;
            _buffer = null;
            _previousScene = string.Empty;
        }

        private static void OnSceneChanged(string targetScene)
        {
            string nextScene = targetScene ?? string.Empty;
            RecordSceneChanged(StableHash(_previousScene), StableHash(nextScene));
            _previousScene = nextScene;
        }

        public static void RecordSceneChanged(int fromSceneHash, int toSceneHash)
        {
            try
            {
                _buffer?.Add(
                    HKRL.RewardEventKind.SceneChanged,
                    entityId: 0,
                    amount: 0.0f,
                    auxInt: fromSceneHash,
                    auxInt2: toSceneHash);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record scene change", exception);
            }
        }

        private static int StableHash(string text)
        {
            unchecked
            {
                int hash = (int)2166136261;
                for (var i = 0; i < text.Length; i++)
                {
                    hash ^= text[i];
                    hash *= 16777619;
                }

                return hash;
            }
        }
    }
}
