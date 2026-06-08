namespace HKRLEnvMod.Observation
{
    using System.Collections.Generic;

    /// <summary>
    /// Reads static/dynamic hazards (spikes, danger zones, platform edges) for
    /// spatial avoidance (docs/observation_schema.md §3). Maps to HKRL.EntityState
    /// with entity_type = Hazard.
    /// </summary>
    public sealed class HazardObserver
    {
        public void ReadInto(ICollection<EntityObservation> entities)
        {
            if (entities == null)
            {
                throw new System.ArgumentNullException(nameof(entities));
            }
        }
    }
}
