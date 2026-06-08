namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads projectiles/bullets (pos, vel, ttl, damage, hitbox). Feeds top-k
    /// threat filtering for bullet-hell phases (docs/model_architecture.md §3).
    /// </summary>
    public sealed class ProjectileObserver
    {
        public void ReadInto(/* entity list */)
        {
            // TODO(phase-4): enumerate active projectiles; compute ttl + threat.
        }
    }
}
