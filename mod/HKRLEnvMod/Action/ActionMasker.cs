namespace HKRLEnvMod.Action
{
    /// <summary>Minimal player action-readiness view used by ActionMasker.</summary>
    public readonly struct PlayerActionState
    {
        public PlayerActionState(
            float dashCooldown = 0.0f,
            int soul = 0,
            float attackLockTimer = 0.0f,
            float castLockTimer = 0.0f,
            bool onGround = true,
            bool doubleJumpAvailable = true,
            bool focusing = false,
            bool canAttack = true,
            bool canCast = true,
            bool canFocus = true)
        {
            DashCooldown = dashCooldown;
            Soul = soul;
            AttackLockTimer = attackLockTimer;
            CastLockTimer = castLockTimer;
            OnGround = onGround;
            DoubleJumpAvailable = doubleJumpAvailable;
            Focusing = focusing;
            CanAttack = canAttack;
            CanCast = canCast;
            CanFocus = canFocus;
        }

        public float DashCooldown { get; }
        public int Soul { get; }
        public float AttackLockTimer { get; }
        public float CastLockTimer { get; }
        public bool OnGround { get; }
        public bool DoubleJumpAvailable { get; }
        public bool Focusing { get; }
        public bool CanAttack { get; }
        public bool CanCast { get; }
        public bool CanFocus { get; }
    }

    /// <summary>
    /// Computes the per-tick action mask from current player state (docs/action_space.md
    /// §3). The flat mask order MUST equal python/hkrl/spaces.py action_mask_layout()
    /// — drift here causes high invalid_action_ratio (docs/troubleshooting.md).
    /// </summary>
    public sealed class ActionMasker
    {
        public const int MovementCount = 3;
        public const int AimCount = 3;
        public const int ButtonCount = 9;
        public const int DurationCount = 4;
        public const int DefaultMacroCount = 11;
        public const int SpellSoulCost = 33;

        public const int MovementOffset = 0;
        public const int AimOffset = MovementOffset + MovementCount;
        public const int ButtonOffset = AimOffset + AimCount;
        public const int DurationOffset = ButtonOffset + ButtonCount;
        public const int MacroOffset = DurationOffset + DurationCount;

        public const int ButtonJumpTap = 0;
        public const int ButtonJumpHold = 1;
        public const int ButtonDash = 2;
        public const int ButtonAttack = 3;
        public const int ButtonCast = 4;
        public const int ButtonFocusHold = 5;
        public const int ButtonDreamNail = 6;
        public const int ButtonNailArtHold = 7;
        public const int ButtonNailArtRelease = 8;

        /// <summary>Return the mask in canonical layout order.</summary>
        public bool[] Compute(
            PlayerActionState? player = null,
            bool enableMacro = true,
            int macroCount = DefaultMacroCount)
        {
            if (macroCount < 0)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(macroCount),
                    "macroCount must be non-negative");
            }

            var mask = CreateAllValid(enableMacro, macroCount);
            if (!player.HasValue)
            {
                return mask;
            }

            ApplyPlayerRules(mask, player.Value);
            return mask;
        }

        public static int MaskLength(bool enableMacro = true, int macroCount = DefaultMacroCount)
        {
            return MacroOffset + (enableMacro ? macroCount + 1 : 0);
        }

        private static bool[] CreateAllValid(bool enableMacro, int macroCount)
        {
            var mask = new bool[MaskLength(enableMacro, macroCount)];
            for (var i = 0; i < mask.Length; i++)
            {
                mask[i] = true;
            }

            return mask;
        }

        private static void ApplyPlayerRules(bool[] mask, PlayerActionState player)
        {
            var attackLocked = player.AttackLockTimer > 0.0f;
            var castLocked = player.CastLockTimer > 0.0f;
            var insufficientSoul = player.Soul < SpellSoulCost;
            var cannotJump = !player.OnGround && !player.DoubleJumpAvailable;

            if (cannotJump)
            {
                MaskButton(mask, ButtonJumpTap);
                MaskButton(mask, ButtonJumpHold);
            }
            if (player.DashCooldown > 0.0f || attackLocked || player.Focusing)
            {
                MaskButton(mask, ButtonDash);
            }
            if (attackLocked || player.Focusing || !player.CanAttack)
            {
                MaskButton(mask, ButtonAttack);
                MaskButton(mask, ButtonNailArtHold);
                MaskButton(mask, ButtonNailArtRelease);
            }
            if (insufficientSoul || attackLocked || castLocked || player.Focusing || !player.CanCast)
            {
                MaskButton(mask, ButtonCast);
            }
            if (insufficientSoul || attackLocked || !player.CanFocus)
            {
                MaskButton(mask, ButtonFocusHold);
            }
        }

        private static void MaskButton(bool[] mask, int buttonIndex)
        {
            mask[ButtonOffset + buttonIndex] = false;
        }
    }
}
