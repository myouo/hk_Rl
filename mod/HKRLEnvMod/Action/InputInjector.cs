namespace HKRLEnvMod.Action
{
    public readonly struct PrimitiveInput
    {
        public PrimitiveInput(int movementX, int aimY, uint buttons)
        {
            MovementX = ClampAxis(movementX);
            AimY = ClampAxis(aimY);
            Buttons = buttons & ButtonMask;
        }

        public const uint ButtonMask = (1u << 9) - 1u;

        public int MovementX { get; }
        public int AimY { get; }
        public uint Buttons { get; }

        public static PrimitiveInput Noop => new PrimitiveInput(0, 0, 0);

        private static int ClampAxis(int value)
        {
            if (value < 0)
            {
                return -1;
            }
            if (value > 0)
            {
                return 1;
            }

            return 0;
        }
    }

    /// <summary>
    /// Injects input directly into the game (in-mod), eliminating vgamepad timing
    /// nondeterminism (PRD §9.2). The button bit layout MUST match python/hkrl/
    /// spaces.py BUTTON_BITS and schema Action.buttons. Main-thread only.
    /// </summary>
    public sealed class InputInjector
    {
        // Button bits (mirror python/hkrl/spaces.py BUTTON_BITS):
        //  0 jump_tap  1 jump_hold  2 dash  3 attack  4 cast
        //  5 focus_hold  6 dream_nail  7 nail_art_hold  8 nail_art_release

        public PrimitiveInput Current { get; private set; } = PrimitiveInput.Noop;

        /// <summary>Set the movement axis (-1/0/+1) for this tick.</summary>
        public void SetMovementX(int dir)
        {
            Current = new PrimitiveInput(dir, Current.AimY, Current.Buttons);
        }

        /// <summary>Set the aim axis (-1 down / 0 / +1 up).</summary>
        public void SetAimY(int dir)
        {
            Current = new PrimitiveInput(Current.MovementX, dir, Current.Buttons);
        }

        /// <summary>Apply the button bitmask (tap/hold/release semantics by bit).</summary>
        public void SetButtons(uint buttons)
        {
            Current = new PrimitiveInput(Current.MovementX, Current.AimY, buttons);
        }

        public void Apply(PrimitiveInput input)
        {
            Current = input;
        }

        public void Clear()
        {
            Current = PrimitiveInput.Noop;
        }
    }
}
