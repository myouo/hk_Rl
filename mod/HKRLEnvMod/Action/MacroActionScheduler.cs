namespace HKRLEnvMod.Action
{
    using System.Collections.Generic;

    /// <summary>
    /// Expands high-level macro actions into primitive input sequences over several
    /// ticks (docs/action_space.md §5, PRD §6.4), keeping the env contract
    /// primitive-based. Macros: approach, retreat, jump_attack, pogo, dash_away,
    /// dash_through, cast_forward, cast_up, focus_when_safe, short_hop, long_jump.
    /// </summary>
    public sealed class MacroActionScheduler
    {
        private readonly Queue<PrimitiveInput> _plan = new();

        /// <summary>Begin executing a macro; subsequent ticks emit its primitives.</summary>
        public void Begin(int macroId)
        {
            _plan.Clear();
            foreach (var input in BuildPlan(macroId))
            {
                _plan.Enqueue(input);
            }
        }

        /// <summary>Advance one tick of the active macro; returns primitive input.</summary>
        public PrimitiveInput Tick()
        {
            return _plan.Count == 0 ? PrimitiveInput.Noop : _plan.Dequeue();
        }

        public bool IsActive => _plan.Count > 0;

        private static IEnumerable<PrimitiveInput> BuildPlan(int macroId)
        {
            return macroId switch
            {
                0 => Repeat(new PrimitiveInput(1, 0, 0), 4), // approach
                1 => Repeat(new PrimitiveInput(-1, 0, 0), 4), // retreat
                2 => Sequence(
                    new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpTap)),
                    new PrimitiveInput(1, 0, Button(ActionMasker.ButtonAttack))), // jump_attack
                3 => Sequence(new PrimitiveInput(0, -1, Button(ActionMasker.ButtonAttack))), // pogo
                4 => Sequence(new PrimitiveInput(-1, 0, Button(ActionMasker.ButtonDash))), // dash_away
                5 => Sequence(new PrimitiveInput(1, 0, Button(ActionMasker.ButtonDash))), // dash_through
                6 => Sequence(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonCast))), // cast_forward
                7 => Sequence(new PrimitiveInput(0, 1, Button(ActionMasker.ButtonCast))), // cast_up
                8 => Repeat(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonFocusHold)), 4),
                9 => Sequence(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpTap))), // short_hop
                10 => Repeat(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpHold)), 4),
                _ => Sequence(PrimitiveInput.Noop)
            };
        }

        private static uint Button(int buttonIndex)
        {
            return 1u << buttonIndex;
        }

        private static IEnumerable<PrimitiveInput> Repeat(PrimitiveInput input, int count)
        {
            for (var i = 0; i < count; i++)
            {
                yield return input;
            }
        }

        private static IEnumerable<PrimitiveInput> Sequence(params PrimitiveInput[] inputs)
        {
            return inputs;
        }
    }
}
