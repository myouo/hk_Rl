namespace HKRLEnvMod.Observation
{
    using System.Collections.Generic;

    /// <summary>
    /// Builds the variable-count entity list (bosses/enemies/projectiles/hazards),
    /// delegating to the specialized observers and the EntityRegistry for stable ids.
    /// Applies top-k priority filtering by threat_score when over capacity
    /// (docs/observation_schema.md §3, PRD §7.3). Maps to HKRL.EntityState[].
    /// </summary>
    public sealed class EntityObserver
    {
        private readonly EntityRegistry _registry = new();
        private readonly BossObserver _boss = new();
        private readonly ProjectileObserver _projectile = new();
        private readonly HazardObserver _hazard = new();
        private readonly List<EntityObservation> _entities = new();
        private readonly HashSet<int> _aliveInstanceIds = new();

        public IReadOnlyList<EntityObservation> Collect(PlayerObservation player, int maxEntities = 64)
        {
            _entities.Clear();
            _aliveInstanceIds.Clear();
            _boss.ReadInto(_entities, _registry, _aliveInstanceIds, player);
            _projectile.ReadInto(_entities, _registry, _aliveInstanceIds, player);
            _hazard.ReadInto(_entities, _registry, _aliveInstanceIds, player);
            _registry.PruneDead(_aliveInstanceIds);

            if (_entities.Count <= maxEntities)
            {
                return _entities;
            }

            _entities.Sort(
                (left, right) => right.ThreatScore.CompareTo(left.ThreatScore));
            _entities.RemoveRange(maxEntities, _entities.Count - maxEntities);
            return _entities;
        }
    }
}
