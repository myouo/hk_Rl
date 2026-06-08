using System;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks player death and boss death -> PlayerDeath / BossKilled events, and
    /// signals the lifecycle to enter TERMINATING (docs/episode_lifecycle.md).
    /// Main-thread.
    /// </summary>
    public static class DeathHooks
    {
        public static void Install(RewardEventBuffer buffer)
        {
            // TODO(phase-1): hook HeroController death -> PlayerDeath + RequestTerminate;
            // hook boss HealthManager death -> BossKilled (+ terminate when last boss).
            throw new NotImplementedException();
        }
    }
}
