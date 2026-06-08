using System.Collections.Generic;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Collects typed reward EVENTS (not scalar reward) during a tick; the Python
    /// reward function composes the scalar (docs/reward_design.md, PRD §5.6/§9.4).
    /// Cleared on reset and after `done` to prevent cross-episode contamination
    /// (PRD §9.3). Written by hooks (any thread? — hooks fire on the main thread).
    /// </summary>
    public sealed class RewardEventBuffer
    {
        private readonly List<object> _events = new(); // TODO: typed event struct

        /// <summary>Append an event (called from Damage/Heal/Death/Scene hooks).</summary>
        public void Add(/* RewardEventKind kind, payload */)
        {
            // TODO(phase-1)
        }

        /// <summary>Drain events for this StepResponse and clear the buffer.</summary>
        public IReadOnlyList<object> Drain()
        {
            // TODO(phase-1)
            return _events;
        }

        /// <summary>Clear without emitting (used by lifecycle CLEAR_EVENTS).</summary>
        public void Clear() => _events.Clear();
    }
}
