using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace HKRLEnvMod.Rewards
{
    /// <summary>Mod-side typed reward event payload; mirrors schema/HKRL.RewardEvent.</summary>
    public readonly struct RewardEventRecord
    {
        public RewardEventRecord(
            HKRL.RewardEventKind kind,
            int entityId = 0,
            float amount = 0.0f,
            int auxInt = 0,
            int auxInt2 = 0)
        {
            Kind = kind;
            EntityId = entityId;
            Amount = amount;
            AuxInt = auxInt;
            AuxInt2 = auxInt2;
        }

        public HKRL.RewardEventKind Kind { get; }
        public int EntityId { get; }
        public float Amount { get; }
        public int AuxInt { get; }
        public int AuxInt2 { get; }
    }

    /// <summary>
    /// Collects typed reward EVENTS (not scalar reward) during a tick; the Python
    /// reward function composes the scalar (docs/reward_design.md, PRD §5.6/§9.4).
    /// Cleared on reset and after `done` to prevent cross-episode contamination
    /// (PRD §9.3). Written by hooks (any thread? — hooks fire on the main thread).
    /// </summary>
    public sealed class RewardEventBuffer
    {
        private readonly List<RewardEventRecord> _events = new();

        public int Count => _events.Count;

        /// <summary>Append an event (called from Damage/Heal/Death/Scene hooks).</summary>
        public void Add(
            HKRL.RewardEventKind kind,
            int entityId = 0,
            float amount = 0.0f,
            int auxInt = 0,
            int auxInt2 = 0)
        {
            _events.Add(new RewardEventRecord(kind, entityId, amount, auxInt, auxInt2));
        }

        /// <summary>Drain events for this StepResponse and clear the buffer.</summary>
        public IReadOnlyList<RewardEventRecord> Drain()
        {
            if (_events.Count == 0)
            {
                return Array.Empty<RewardEventRecord>();
            }

            var drained = _events.ToArray();
            _events.Clear();
            return new ReadOnlyCollection<RewardEventRecord>(drained);
        }

        /// <summary>Clear without emitting (used by lifecycle CLEAR_EVENTS).</summary>
        public void Clear() => _events.Clear();
    }
}
