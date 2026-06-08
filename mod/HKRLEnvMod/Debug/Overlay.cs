using UnityEngine;

namespace HKRLEnvMod.Debug
{
    /// <summary>
    /// On-screen debug overlay to visually verify observations: entity boxes,
    /// hurtboxes/hitboxes, threat scores, lifecycle state, SPS (docs/mod_dev.md §6).
    /// Toggleable; disabled during high-SPS training to save render cost.
    /// </summary>
    public sealed class Overlay : MonoBehaviour
    {
        public bool Enabled { get; set; }

        private void OnGUI()
        {
            if (!Enabled) return;
            // TODO(phase-1): draw entities/hitboxes/lifecycle/SPS.
        }
    }
}
