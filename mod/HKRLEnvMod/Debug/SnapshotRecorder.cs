namespace HKRLEnvMod.Debug
{
    /// <summary>
    /// Optionally records observation/action snapshots to disk for offline replay
    /// and unit tests (docs/mod_dev.md §6). Off by default; high overhead.
    /// </summary>
    public sealed class SnapshotRecorder
    {
        public bool Enabled { get; set; }

        /// <summary>Record one tick's snapshot if enabled.</summary>
        public void Record(/* snapshot */)
        {
            if (!Enabled) return;
            // TODO(phase-4): append to a snapshot file for replay/regression tests.
        }
    }
}
