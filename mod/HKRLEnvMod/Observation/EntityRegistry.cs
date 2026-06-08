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
            if (_objToStableId.TryGetValue(unityInstanceId, out var stableId))
            {
                return stableId;
            }

            stableId = _nextId++;
            _objToStableId[unityInstanceId] = stableId;
            return stableId;
        }

        /// <summary>Drop ids for objects no longer present this frame.</summary>
        public void PruneDead(HashSet<int> aliveInstanceIds)
        {
            if (aliveInstanceIds == null)
            {
                throw new System.ArgumentNullException(nameof(aliveInstanceIds));
            }

            var dead = new List<int>();
            foreach (var instanceId in _objToStableId.Keys)
            {
                if (!aliveInstanceIds.Contains(instanceId))
                {
                    dead.Add(instanceId);
                }
            }

            foreach (var instanceId in dead)
            {
                _objToStableId.Remove(instanceId);
            }
        }
    }
}
