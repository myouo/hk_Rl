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
            _buffer = buffer ?? throw new System.ArgumentNullException(nameof(buffer));
        }

        public static void RecordDamageDealt(int entityId, float amount, int damageType = 0)
        {
            _buffer?.Add(HKRL.RewardEventKind.DamageDealt, entityId, amount, damageType);
        }

        public static void RecordDamageTaken(int entityId, float amount, int damageType = 0)
        {
            _buffer?.Add(HKRL.RewardEventKind.DamageTaken, entityId, amount, damageType);
        }
    }
}
