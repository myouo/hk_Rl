using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Loads/validates Godhome boss scenes for a task (e.g. GG_Hornet_1) and reports
    /// readiness. Wraps GameManager/BossSceneController interactions. Main-thread only.
    /// </summary>
    public sealed class SceneController
    {
        private string _targetSceneName = "GG_Gruz_Mother";

        public int CurrentTaskId { get; private set; }
        public string TargetSceneName => _targetSceneName;
        public bool HasValidTarget => !string.IsNullOrEmpty(_targetSceneName);

        /// <summary>Request loading the scene/arena for a task id.</summary>
        public void LoadTaskScene(int taskId)
        {
            CurrentTaskId = taskId;
            _targetSceneName = ResolveSceneName(taskId);
            if (!HasValidTarget)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    $"Unknown HKRL task id {taskId}; reset will fail scene readiness.");
                return;
            }

            if (SceneManager.GetActiveScene().name == _targetSceneName)
            {
                return;
            }

            SceneManager.LoadScene(_targetSceneName);
        }

        public bool IsSceneReady()
        {
            if (!HasValidTarget)
            {
                return false;
            }

            Scene scene = SceneManager.GetActiveScene();
            return scene.isLoaded && scene.name == _targetSceneName;
        }

        public bool IsPlayerReady()
        {
            global::HeroController? hero = global::HeroController.instance;
            return hero != null && hero.gameObject != null && hero.gameObject.activeInHierarchy;
        }

        public bool IsBossReady()
        {
            if (!IsSceneReady())
            {
                return false;
            }

            global::HealthManager[] healthManagers =
                UnityEngine.Object.FindObjectsOfType<global::HealthManager>();
            return healthManagers != null && healthManagers.Length > 0;
        }

        private static string ResolveSceneName(int taskId)
        {
            return taskId switch
            {
                0 => "GG_Gruz_Mother",
                1 => "GG_Hornet_1",
                2 => "GG_Mantis_Lords",
                _ => string.Empty
            };
        }
    }
}
