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
        public const int ApproachTicks = 4;
        public const int RetreatTicks = 4;
        public const int JumpAttackTicks = 2;
        public const int PogoTicks = 5;
        public const int DashAwayTicks = 1;
        public const int DashThroughTicks = 1;
        public const int CastForwardTicks = 1;
        public const int CastUpTicks = 1;
        public const int FocusWhenSafeTicks = 120;
        public const int ShortHopTicks = 1;
        public const int LongJumpTicks = 4;

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

        /// <summary>
        /// Number of primitive ticks emitted by a known macro. Kept public so the
        /// cross-language action contract can verify Python decision timing.
        /// </summary>
        public static int ExpectedDurationTicks(int macroId)
        {
            return macroId switch
            {
                0 => ApproachTicks,
                1 => RetreatTicks,
                2 => JumpAttackTicks,
                3 => PogoTicks,
                4 => DashAwayTicks,
                5 => DashThroughTicks,
                6 => CastForwardTicks,
                7 => CastUpTicks,
                8 => FocusWhenSafeTicks,
                9 => ShortHopTicks,
                10 => LongJumpTicks,
                _ => 1
            };
        }

        /// <summary>Discard every queued primitive (reset/reconnect safety).</summary>
        public void Clear()
        {
            _plan.Clear();
        }

        private static IEnumerable<PrimitiveInput> BuildPlan(int macroId)
        {
            return macroId switch
            {
                0 => Repeat(new PrimitiveInput(1, 0, 0), ApproachTicks), // approach
                1 => Repeat(new PrimitiveInput(-1, 0, 0), RetreatTicks), // retreat
                2 => Sequence(
                    new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpTap)),
                    new PrimitiveInput(1, 0, Button(ActionMasker.ButtonAttack))), // jump_attack
                3 => Concat(
                    Repeat(
                        new PrimitiveInput(
                            0,
                            0,
                            Button(ActionMasker.ButtonJumpHold)),
                        PogoTicks - 1),
                    Sequence(
                        new PrimitiveInput(
                            0,
                            -1,
                            Button(ActionMasker.ButtonAttack)))), // pogo
                4 => Sequence(new PrimitiveInput(-1, 0, Button(ActionMasker.ButtonDash))), // dash_away
                5 => Sequence(new PrimitiveInput(1, 0, Button(ActionMasker.ButtonDash))), // dash_through
                6 => Sequence(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonCast))), // cast_forward
                7 => Sequence(new PrimitiveInput(0, 1, Button(ActionMasker.ButtonCast))), // cast_up
                8 => Repeat(
                    new PrimitiveInput(
                        0,
                        0,
                        Button(ActionMasker.ButtonFocusHold)),
                    FocusWhenSafeTicks), // focus_when_safe: one complete heal attempt
                9 => Sequence(new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpTap))), // short_hop
                10 => Repeat(
                    new PrimitiveInput(0, 0, Button(ActionMasker.ButtonJumpHold)),
                    LongJumpTicks),
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

        private static IEnumerable<PrimitiveInput> Concat(
            params IEnumerable<PrimitiveInput>[] sequences)
        {
            foreach (var sequence in sequences)
            {
                foreach (var input in sequence)
                {
                    yield return input;
                }
            }
        }
    }
}
