using System;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Hooks scene transitions -> SceneChanged events; relevant for linear multi-boss
    /// flows and for routing unexpected scene changes into TERMINATING. Main-thread.
    /// </summary>
    public static class SceneHooks
    {
        public static void Install(RewardEventBuffer buffer)
        {
            // TODO(phase-1): hook GameManager scene-change -> SceneChanged(from,to).
            throw new NotImplementedException();
        }
    }
}
