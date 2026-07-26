namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks focus/heal and soul changes -> Heal / SoulGained events. Main-thread.
    /// </summary>
    public static class HealHooks
    {
        private static RewardEventBuffer? _buffer;

        public static void Install(RewardEventBuffer buffer)
        {
            Uninstall();
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
            Modding.ModHooks.BeforeAddHealthHook += OnBeforeAddHealth;
        }

        public static void Uninstall()
        {
            Modding.ModHooks.BeforeAddHealthHook -= OnBeforeAddHealth;
            _buffer = null;
        }

        private static int OnBeforeAddHealth(int amount)
        {
            RecordHeal(entityId: 0, amount: amount);
            return amount;
        }

        public static void RecordHeal(int entityId, float amount)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.Heal, entityId, amount);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record heal", exception);
            }
        }

        public static void RecordSoulGained(int entityId, float amount)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.SoulGained, entityId, amount);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record soul gained", exception);
            }
        }
    }
}
