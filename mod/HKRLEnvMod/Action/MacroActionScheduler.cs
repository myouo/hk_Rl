namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Expands high-level macro actions into primitive input sequences over several
    /// ticks (docs/action_space.md §5, PRD §6.4), keeping the env contract
    /// primitive-based. Macros: approach, retreat, jump_attack, pogo, dash_away,
    /// dash_through, cast_forward, cast_up, focus_when_safe, short_hop, long_jump.
    /// </summary>
    public sealed class MacroActionScheduler
    {
        /// <summary>Begin executing a macro; subsequent ticks emit its primitives.</summary>
        public void Begin(int macroId)
        {
            // TODO(phase-2): map macroId -> primitive plan.
        }

        /// <summary>Advance one tick of the active macro; returns primitive input.</summary>
        public void Tick(/* out primitive action */)
        {
            // TODO(phase-2)
        }

        public bool IsActive => false; // TODO(phase-2)
    }
}
