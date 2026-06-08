using System;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks focus/heal and soul changes -> Heal / SoulGained events. Main-thread.
    /// </summary>
    public static class HealHooks
    {
        public static void Install(RewardEventBuffer buffer)
        {
            // TODO(phase-1): hook focus completion -> Heal; soul gain -> SoulGained.
            throw new NotImplementedException();
        }
    }
}
