namespace HKRLEnvMod.Action
{
    /// <summary>
    /// Computes the per-tick action mask from current player state (docs/action_space.md
    /// §3). The flat mask order MUST equal python/hkrl/spaces.py action_mask_layout()
    /// — drift here causes high invalid_action_ratio (docs/troubleshooting.md).
    /// </summary>
    public sealed class ActionMasker
    {
        /// <summary>Write the mask (bool[]) in canonical layout order.</summary>
        public void Compute(/* PlayerState, out bool[] mask */)
        {
            // TODO(phase-1): dash_cooldown>0 -> mask dash; soul<cost -> mask cast/focus;
            // attack_lock>0 -> mask attack/cast/dash; airborne&&!double_jump -> mask jump;
            // focusing -> mask attack/dash/cast; movement/aim mutually exclusive.
        }
    }
}
