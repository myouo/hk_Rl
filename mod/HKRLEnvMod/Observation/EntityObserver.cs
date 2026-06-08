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

        public IReadOnlyList<EntityObservation> Collect(PlayerObservation player, int maxEntities = 64)
        {
            var entities = new List<EntityObservation>();
            var aliveInstanceIds = new HashSet<int>();
            _boss.ReadInto(entities, _registry, aliveInstanceIds, player);
            _projectile.ReadInto(entities, _registry, aliveInstanceIds, player);
            _hazard.ReadInto(entities, _registry, aliveInstanceIds, player);
            _registry.PruneDead(aliveInstanceIds);

            if (entities.Count <= maxEntities)
            {
                return entities;
            }

            entities.Sort((left, right) => right.ThreatScore.CompareTo(left.ThreatScore));
            return entities.GetRange(0, maxEntities);
        }
    }
}
