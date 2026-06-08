using System.Collections.Generic;

namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Assigns and maintains stable entity ids across frames so velocities and
    /// history stay coherent (docs/observation_schema.md §3). Maps Unity object
    /// identity to a small stable integer id reused across ticks; recycles ids when
    /// objects die. Main-thread only.
    /// </summary>
    public sealed class EntityRegistry
    {
        private readonly Dictionary<int, int> _objToStableId = new();
        private int _nextId = 1;

        /// <summary>Return the stable id for a Unity object (allocating if new).</summary>
        public int GetStableId(int unityInstanceId)
        {
            // TODO(phase-4): lookup/insert; track liveness for recycling.
            return 0;
        }

        /// <summary>Drop ids for objects no longer present this frame.</summary>
        public void PruneDead(HashSet<int> aliveInstanceIds)
        {
            // TODO(phase-4)
        }
    }
}
