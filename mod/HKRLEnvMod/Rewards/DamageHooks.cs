namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Harmony hooks for damage dealt/taken (HealthManager.TakeDamage,
    /// HeroController damage). Emits DamageDealt / DamageTaken events into the
    /// RewardEventBuffer. Wrap hook bodies in try/catch (PRD §9.9). Main-thread.
    /// </summary>
    public static class DamageHooks
    {
        private static RewardEventBuffer? _buffer;

        /// <summary>Install the Harmony/MonoMod hooks.</summary>
        public static void Install(RewardEventBuffer buffer)
        {
            Uninstall();
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
            Modding.ModHooks.TakeHealthHook += OnTakeHealth;
        }

        public static void Uninstall()
        {
            Modding.ModHooks.TakeHealthHook -= OnTakeHealth;
            _buffer = null;
        }

        private static int OnTakeHealth(int damage)
        {
            RecordDamageTaken(entityId: 0, amount: damage);
            return damage;
        }

        public static void RecordDamageDealt(int entityId, float amount, int damageType = 0)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.DamageDealt, entityId, amount, damageType);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record damage dealt", exception);
            }
        }

        public static void RecordDamageTaken(int entityId, float amount, int damageType = 0)
        {
            try
            {
                _buffer?.Add(HKRL.RewardEventKind.DamageTaken, entityId, amount, damageType);
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to record damage taken", exception);
            }
        }
    }
}
