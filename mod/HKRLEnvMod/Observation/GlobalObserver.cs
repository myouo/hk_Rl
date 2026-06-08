namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads GlobalState (scene/task/episode/time context). Maps to HKRL.GlobalState.
    /// See docs/observation_schema.md.
    /// </summary>
    public sealed class GlobalObserver
    {
        public void Read(/* out GlobalState fields */)
        {
            // TODO(phase-1): scene_hash, arena_id, task_id, time_in_episode,
            // time_scale, fixed_delta_time, stage_index, episode_id.
        }
    }
}
