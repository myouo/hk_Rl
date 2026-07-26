using System;
using HKRLEnvMod.Transport;

namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Applies a decoded Action on the MAIN THREAD in FixedUpdate (docs/action_space.md
    /// §6, PRD §9.2). Translates the hybrid action (movement/aim/buttons/duration/
    /// macro) into in-game input via InputInjector, expanding macros first.
    /// </summary>
    public sealed class ActionApplier : IDisposable
    {
        private static readonly int[] DurationTicks = { 1, 2, 4, 8 };
        private const int MaxSuspendedInputTicks = 10;
        private const uint ContinuableHoldMask =
            (1u << ActionMasker.ButtonJumpHold)
            | (1u << ActionMasker.ButtonFocusHold)
            | (1u << ActionMasker.ButtonDreamNail)
            | (1u << ActionMasker.ButtonNailArtHold);

        private readonly InputInjector _input = new();
        private readonly MacroActionScheduler _macros = new();
        private PrimitiveInput _heldInput = PrimitiveInput.Noop;
        private int _heldTicksRemaining;
        private int _suspendedInputTicksRemaining;

        public PrimitiveInput CurrentInput => _input.Current;
        public ulong SuccessfulCommitCount => _input.SuccessfulCommitCount;
        public bool InputEnabled => _input.Enabled;

        /// <summary>Apply one action for this tick (held for `duration` ticks).</summary>
        public void Apply(DecodedAction action, bool isNewDecision = true)
        {
            _suspendedInputTicksRemaining = 0;
            if (_heldTicksRemaining > 0)
            {
                _input.Apply(_heldInput);
                _heldTicksRemaining--;
                return;
            }

            // A macro is an autonomous primitive plan. Repeated protocol ticks
            // advance an existing plan instead of restarting it at frame zero.
            if (isNewDecision && action.MacroId >= 0 && !_macros.IsActive)
            {
                _macros.Begin(action.MacroId);
            }

            var macroActive = _macros.IsActive;
            var primitive = macroActive
                ? _macros.Tick()
                : action.MacroId >= 0
                    ? PrimitiveInput.Noop
                    : ToPrimitive(action);
            _heldInput = primitive;
            _heldTicksRemaining = macroActive ? 0 : DurationFromIndex(action.DurationIdx) - 1;
            _input.Apply(primitive);
        }

        public void Clear()
        {
            _heldInput = PrimitiveInput.Noop;
            _heldTicksRemaining = 0;
            _suspendedInputTicksRemaining = 0;
            _macros.Clear();
            _input.Clear();
        }

        public void EnableInput()
        {
            _input.Enable();
        }

        public void DisableInput()
        {
            Clear();
            _input.Disable();
        }

        /// <summary>
        /// Strip edge-triggered buttons between synchronous STEP responses while
        /// briefly preserving movement/aim axes and explicit hold controls. A
        /// policy can therefore continue walking without an injected neutral
        /// physics tick, while attack/dash/cast still receive a real release edge.
        /// </summary>
        public void SuspendInput()
        {
            PrimitiveInput current = _input.Current;
            var continuation = new PrimitiveInput(
                current.MovementX,
                current.AimY,
                current.Buttons & ContinuableHoldMask);
            if (continuation.MovementX != 0
                || continuation.AimY != 0
                || continuation.Buttons != 0)
            {
                _suspendedInputTicksRemaining = MaxSuspendedInputTicks;
                _input.Apply(continuation);
                return;
            }

            _suspendedInputTicksRemaining = 0;
            _input.Clear();
        }

        /// <summary>
        /// Expire continuous state left across a response if the connected policy
        /// stops sending decisions. Ten 50-Hz ticks bridge normal local inference
        /// while bounding stale input to 200 ms.
        /// </summary>
        public void IdleTick()
        {
            if (_suspendedInputTicksRemaining <= 0)
            {
                return;
            }

            _suspendedInputTicksRemaining--;
            if (_suspendedInputTicksRemaining == 0)
            {
                _input.Clear();
            }
        }

        public void Dispose()
        {
            Clear();
            _input.Dispose();
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
