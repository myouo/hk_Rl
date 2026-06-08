using System;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Loads/validates Godhome boss scenes for a task (e.g. GG_Hornet_1) and reports
    /// readiness. Wraps GameManager/BossSceneController interactions. Main-thread only.
    /// </summary>
    public sealed class SceneController
    {
        /// <summary>Request loading the scene/arena for a task id.</summary>
        public void LoadTaskScene(int taskId)
        {
            // TODO(phase-1): map taskId -> scene name; trigger Godhome statue/boss entry.
            throw new NotImplementedException();
        }

        public bool IsSceneReady() => false;   // TODO(phase-1)
        public bool IsPlayerReady() => false;  // TODO(phase-1)
        public bool IsBossReady() => false;     // TODO(phase-1)
    }
}
