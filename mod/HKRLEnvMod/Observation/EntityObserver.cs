namespace HKRLEnvMod.Observation
{
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

        public void Collect(/* out entities[], out entity_mask[] */)
        {
            // TODO(phase-4): gather entities, assign stable ids, compute rel_*,
            // threat_score; top-k filter; aggregate remainder into a summary token.
        }
    }
}
