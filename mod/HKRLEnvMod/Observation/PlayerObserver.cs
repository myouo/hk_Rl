namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads PlayerState from HeroController/PlayerData incl. explicit cooldown and
    /// lock timers that make the env Markovian (docs/observation_schema.md §5,
    /// PRD §9.1). Maps to HKRL.PlayerState.
    /// </summary>
    public sealed class PlayerObserver
    {
        public void Read(/* out PlayerState fields */)
        {
            // TODO(phase-1): pos/vel, hp/soul, facing, on_ground/dashing/etc,
            // invuln_timer, attack/cast lock timers, dash_cooldown, can_* flags.
        }
    }
}
