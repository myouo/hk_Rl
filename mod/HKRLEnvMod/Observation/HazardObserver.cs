namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads static/dynamic hazards (spikes, danger zones, platform edges) for
    /// spatial avoidance (docs/observation_schema.md §3). Maps to HKRL.EntityState
    /// with entity_type = Hazard.
    /// </summary>
    public sealed class HazardObserver
    {
        public void ReadInto(/* entity list */)
        {
            // TODO(phase-4): enumerate hazards near the player.
        }
    }
}
