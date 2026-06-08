namespace HKRLEnvMod.Observation
{
    using System.Collections.Generic;

    /// <summary>
    /// Reads projectiles/bullets (pos, vel, ttl, damage, hitbox). Feeds top-k
    /// threat filtering for bullet-hell phases (docs/model_architecture.md §3).
    /// </summary>
    public sealed class ProjectileObserver
    {
        public void ReadInto(ICollection<EntityObservation> entities)
        {
            if (entities == null)
            {
                throw new System.ArgumentNullException(nameof(entities));
            }
        }
    }
}
