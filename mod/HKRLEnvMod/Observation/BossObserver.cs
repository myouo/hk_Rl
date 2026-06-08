using System.Collections.Generic;

namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads boss entities via BossSceneController / HealthManager / PlayMakerFSM
    /// (hp, fsm_state_hash, phase, hitbox/hurtbox). Used standalone for the single-
    /// boss baseline (Phase 1/3) and by EntityObserver for multi-entity (Phase 4).
    /// </summary>
    public sealed class BossObserver
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
