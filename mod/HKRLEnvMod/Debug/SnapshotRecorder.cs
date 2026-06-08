namespace HKRLEnvMod.Debug
{
    using System;
    using System.IO;
    using UnityEngine;

    /// <summary>
    /// Optionally records observation/action snapshots to disk for offline replay
    /// and unit tests (docs/mod_dev.md §6). Off by default; high overhead.
    /// </summary>
    public sealed class SnapshotRecorder
    {
        public SnapshotRecorder(string? filePath = null)
        {
            FilePath = filePath
                ?? Path.Combine(Application.persistentDataPath, "hkrl_snapshots.jsonl");
        }

        public bool Enabled { get; set; }
        public string FilePath { get; set; }

        /// <summary>Record one tick's snapshot if enabled.</summary>
        public void Record(string jsonLine)
        {
            if (!Enabled)
            {
                return;
            }
            if (jsonLine == null)
            {
                throw new ArgumentNullException(nameof(jsonLine));
            }

            var directory = Path.GetDirectoryName(FilePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.AppendAllText(FilePath, jsonLine + Environment.NewLine);
        }
    }
}
