using HKRLEnvMod.Transport;

namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Hard policy boundary for actions a training policy may execute. Anything
    /// outside ordinary player controls is rejected and applied as a no-op.
    /// Environment control commands (RESET/PAUSE/timescale) are not policy actions.
    /// </summary>
    public static class TrainingCapabilityPolicy
    {
        public const uint AllowedButtonMask = (1u << ActionMasker.ButtonCount) - 1u;

        public static bool IsAllowed(
            DecodedAction action,
            bool enableMacroActions,
            int macroCount)
        {
            if (action.MovementX > 2
                || action.AimY > 2
                || (action.Buttons & ~AllowedButtonMask) != 0
                || action.DurationIdx >= ActionMasker.DurationCount)
            {
                return false;
            }

            if (!enableMacroActions)
            {
                return action.MacroId < 0;
            }

            return macroCount >= 0
                && action.MacroId >= -1
                && action.MacroId < macroCount;
        }
    }
}
