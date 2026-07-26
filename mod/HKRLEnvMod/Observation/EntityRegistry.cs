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
        private readonly Dictionary<int, int> _maxHpByInstanceId = new();
        private readonly List<int> _deadInstanceIds = new();
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

        /// <summary>
        /// Return the best max-hp value seen for an object. Many Hollow Knight
        /// HealthManager variants expose only their current `hp`; caching the
        /// episode's observed maximum keeps hp normalization and damage deltas
        /// usable without hard-coded per-boss health tables.
        /// </summary>
        public int ObserveMaxHp(int unityInstanceId, int currentHp, int declaredMaxHp)
        {
            var candidate = System.Math.Max(currentHp, declaredMaxHp);
            if (_maxHpByInstanceId.TryGetValue(unityInstanceId, out var observed))
            {
                candidate = System.Math.Max(candidate, observed);
            }

            if (candidate > 0)
            {
                _maxHpByInstanceId[unityInstanceId] = candidate;
            }

            return candidate;
        }

        /// <summary>Drop ids for objects no longer present this frame.</summary>
        public void PruneDead(HashSet<int> aliveInstanceIds)
        {
            if (aliveInstanceIds == null)
            {
                throw new System.ArgumentNullException(nameof(aliveInstanceIds));
            }

            _deadInstanceIds.Clear();
            foreach (var instanceId in _objToStableId.Keys)
            {
                if (!aliveInstanceIds.Contains(instanceId))
                {
                    _deadInstanceIds.Add(instanceId);
                }
            }

            foreach (var instanceId in _deadInstanceIds)
            {
                _objToStableId.Remove(instanceId);
                _maxHpByInstanceId.Remove(instanceId);
            }
        }
    }
}
