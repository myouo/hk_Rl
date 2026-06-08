using System;

namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Applies a decoded Action on the MAIN THREAD in FixedUpdate (docs/action_space.md
    /// §6, PRD §9.2). Translates the hybrid action (movement/aim/buttons/duration/
    /// macro) into in-game input via InputInjector, expanding macros first.
    /// </summary>
    public sealed class ActionApplier
    {
        private readonly InputInjector _input = new();
        private readonly MacroActionScheduler _macros = new();

        /// <summary>Apply one action for this tick (held for `duration` ticks).</summary>
        public void Apply(/* decoded Action */)
        {
            // TODO(phase-1): if macro_id >= 0 expand via scheduler, else inject
            // movement_x/aim_y/buttons; honor duration + action_repeat.
            throw new NotImplementedException();
        }
    }
}
