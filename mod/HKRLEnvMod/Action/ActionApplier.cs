using HKRLEnvMod.Transport;

namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Applies a decoded Action on the MAIN THREAD in FixedUpdate (docs/action_space.md
    /// §6, PRD §9.2). Translates the hybrid action (movement/aim/buttons/duration/
    /// macro) into in-game input via InputInjector, expanding macros first.
    /// </summary>
    public sealed class ActionApplier
    {
        private static readonly int[] DurationTicks = { 1, 2, 4, 8 };

        private readonly InputInjector _input = new();
        private readonly MacroActionScheduler _macros = new();
        private PrimitiveInput _heldInput = PrimitiveInput.Noop;
        private int _heldTicksRemaining;

        public PrimitiveInput CurrentInput => _input.Current;

        /// <summary>Apply one action for this tick (held for `duration` ticks).</summary>
        public void Apply(DecodedAction action)
        {
            if (_heldTicksRemaining > 0)
            {
                _input.Apply(_heldInput);
                _heldTicksRemaining--;
                return;
            }

            if (action.MacroId >= 0)
            {
                _macros.Begin(action.MacroId);
            }

            var macroActive = _macros.IsActive;
            var primitive = macroActive ? _macros.Tick() : ToPrimitive(action);
            _heldInput = primitive;
            _heldTicksRemaining = macroActive ? 0 : DurationFromIndex(action.DurationIdx) - 1;
            _input.Apply(primitive);
        }

        public void Clear()
        {
            _heldInput = PrimitiveInput.Noop;
            _heldTicksRemaining = 0;
            _input.Clear();
        }

        private static PrimitiveInput ToPrimitive(DecodedAction action)
        {
            return new PrimitiveInput(
                MovementFromWire(action.MovementX),
                AimFromWire(action.AimY),
                action.Buttons);
        }

        private static int MovementFromWire(byte movementX)
        {
            return movementX switch
            {
                0 => -1,
                2 => 1,
                _ => 0
            };
        }

        private static int AimFromWire(byte aimY)
        {
            return aimY switch
            {
                0 => -1,
                2 => 1,
                _ => 0
            };
        }

        private static int DurationFromIndex(byte durationIdx)
        {
            return DurationTicks[durationIdx < DurationTicks.Length ? durationIdx : 0];
        }
    }
}
