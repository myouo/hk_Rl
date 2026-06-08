using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKRLEnvMod.Observation
{
    public readonly struct GlobalObservation
    {
        public GlobalObservation(
            int sceneHash,
            int arenaId,
            int taskId,
            byte difficulty,
            float timeInEpisode,
            float timeScale,
            float fixedDeltaTime,
            int stageIndex,
            ulong episodeId)
        {
            SceneHash = sceneHash;
            ArenaId = arenaId;
            TaskId = taskId;
            Difficulty = difficulty;
            TimeInEpisode = timeInEpisode;
            TimeScale = timeScale;
            FixedDeltaTime = fixedDeltaTime;
            StageIndex = stageIndex;
            EpisodeId = episodeId;
        }

        public int SceneHash { get; }
        public int ArenaId { get; }
        public int TaskId { get; }
        public byte Difficulty { get; }
        public float TimeInEpisode { get; }
        public float TimeScale { get; }
        public float FixedDeltaTime { get; }
        public int StageIndex { get; }
        public ulong EpisodeId { get; }
    }

    /// <summary>
    /// Reads GlobalState (scene/task/episode/time context). Maps to HKRL.GlobalState.
    /// See docs/observation_schema.md.
    /// </summary>
    public sealed class GlobalObserver
    {
        public GlobalObservation Read(
            int taskId = 0,
            ulong episodeId = 0,
            int stageIndex = 0,
            float timeInEpisode = 0.0f)
        {
            Scene scene = SceneManager.GetActiveScene();
            int sceneHash = StableHash(scene.name);
            return new GlobalObservation(
                sceneHash,
                arenaId: sceneHash,
                taskId,
                difficulty: 0,
                timeInEpisode: timeInEpisode,
                timeScale: Time.timeScale,
                fixedDeltaTime: Time.fixedDeltaTime,
                stageIndex,
                episodeId);
        }

        private static int StableHash(string text)
        {
            unchecked
            {
                int hash = (int)2166136261;
                for (var i = 0; i < text.Length; i++)
                {
                    hash ^= text[i];
                    hash *= 16777619;
                }

                return hash;
            }
        }
    }
}
