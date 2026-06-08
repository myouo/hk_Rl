namespace HKRLEnvMod.Action
{
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

        /// <summary>Set the movement axis (-1/0/+1) for this tick.</summary>
        public void SetMovementX(int dir) { /* TODO(phase-1) */ }

        /// <summary>Set the aim axis (-1 down / 0 / +1 up).</summary>
        public void SetAimY(int dir) { /* TODO(phase-1) */ }

        /// <summary>Apply the button bitmask (tap/hold/release semantics by bit).</summary>
        public void SetButtons(uint buttons) { /* TODO(phase-1) */ }
    }
}
