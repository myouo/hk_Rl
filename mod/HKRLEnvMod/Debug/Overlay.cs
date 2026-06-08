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
        public string StatusText { get; set; } = "HKRL environment server";
        public float StepsPerSecond { get; set; }

        private void OnGUI()
        {
            if (!Enabled)
            {
                return;
            }

            const int width = 360;
            const int height = 72;
            GUI.Box(new Rect(12, 12, width, height), "HKRL");
            GUI.Label(new Rect(24, 36, width - 24, 20), StatusText);
            GUI.Label(new Rect(24, 58, width - 24, 20), $"SPS: {StepsPerSecond:F1}");
        }
    }
}
