using System;

namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Assembles a full Observation snapshot each tick from the sub-observers
    /// (global/player/entities). Output maps to HKRL.Observation (schema/hkrl.fbs).
    /// Semantics: docs/observation_schema.md. Main-thread only.
    /// </summary>
    public sealed class ObservationCollector
    {
        private readonly GlobalObserver _global = new();
        private readonly PlayerObserver _player = new();
        private readonly EntityObserver _entities = new();

        /// <summary>Collect a snapshot; returns data for MessageCodec to encode.</summary>
        public void Collect(/* out snapshot */)
        {
            // TODO(phase-1): global + player + entity list + entity_mask.
            // Health-check the result before returning (docs/observation_schema.md §6).
            throw new NotImplementedException();
        }
    }
}
