using System;
using System.Collections.Generic;

namespace HKRLEnvMod.Observation
{
    public readonly struct EntityObservation
    {
    }

    public sealed class ObservationSnapshot
    {
        public ObservationSnapshot(
            GlobalObservation global,
            PlayerObservation player,
            IReadOnlyList<EntityObservation> entities,
            IReadOnlyList<bool> entityMask)
        {
            Global = global;
            Player = player;
            Entities = entities;
            EntityMask = entityMask;
        }

        public GlobalObservation Global { get; }
        public PlayerObservation Player { get; }
        public IReadOnlyList<EntityObservation> Entities { get; }
        public IReadOnlyList<bool> EntityMask { get; }
    }

    /// <summary>
    /// Assembles a full Observation snapshot each tick from the sub-observers
    /// (global/player/entities). Output maps to HKRL.Observation (schema/hkrl.fbs).
    /// Semantics: docs/observation_schema.md. Main-thread only.
    /// </summary>
    public sealed class ObservationCollector
    {
        private readonly GlobalObserver _global = new();
        private readonly PlayerObserver _player = new();

        /// <summary>Collect a snapshot; returns data for MessageCodec to encode.</summary>
        public ObservationSnapshot Collect(int taskId = 0, ulong episodeId = 0)
        {
            return new ObservationSnapshot(
                _global.Read(taskId, episodeId),
                _player.Read(),
                Array.Empty<EntityObservation>(),
                Array.Empty<bool>());
        }
    }
}
