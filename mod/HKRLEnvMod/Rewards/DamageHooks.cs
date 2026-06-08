using System;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Harmony hooks for damage dealt/taken (HealthManager.TakeDamage,
    /// HeroController damage). Emits DamageDealt / DamageTaken events into the
    /// RewardEventBuffer. Wrap hook bodies in try/catch (PRD §9.9). Main-thread.
    /// </summary>
    public static class DamageHooks
    {
        /// <summary>Install the Harmony/MonoMod hooks.</summary>
        public static void Install(RewardEventBuffer buffer)
        {
            // TODO(phase-1): hook HealthManager.TakeDamage (boss) -> DamageDealt;
            // hook hero damage -> DamageTaken; resolve source/target entity ids.
            throw new NotImplementedException();
        }
    }
}
