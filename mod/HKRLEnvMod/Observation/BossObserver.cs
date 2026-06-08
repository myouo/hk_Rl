namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads boss entities via BossSceneController / HealthManager / PlayMakerFSM
    /// (hp, fsm_state_hash, phase, hitbox/hurtbox). Used standalone for the single-
    /// boss baseline (Phase 1/3) and by EntityObserver for multi-entity (Phase 4).
    /// </summary>
    public sealed class BossObserver
    {
        public void ReadInto(/* entity list */)
        {
            // TODO(phase-1): single boss; (phase-4): iterate all bosses.
        }
    }
}
