namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Pure routing rules for Hall of Gods entry versus an in-arena fast reload.
    /// Keeping this decision independent of Unity objects makes the dangerous
    /// same-scene transition branch directly smoke-testable.
    /// </summary>
    internal static class GodhomeTransitionPolicy
    {
        public static bool RequiresEntryBootstrap(
            string sourceSceneName,
            string targetSceneName)
        {
            return !string.Equals(
                sourceSceneName,
                targetSceneName,
                System.StringComparison.Ordinal);
        }
    }
}
