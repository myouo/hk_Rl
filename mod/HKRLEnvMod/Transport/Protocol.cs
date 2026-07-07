namespace HKRLEnvMod.Transport
{
    /// <summary>
    /// Wire protocol constants. MUST mirror python/hkrl/protocol.py and the
    /// definitions in schema/hkrl.fbs. Bump <see cref="SchemaVersion"/> on every
    /// schema change (schema/README.md evolution rules).
    /// </summary>
    public static class Protocol
    {
        /// <summary>Mirrors SCHEMA_VERSION in python/hkrl/protocol.py.</summary>
        public const int SchemaVersion = 3;

        /// <summary>FlatBuffers file_identifier (must equal hkrl.fbs).</summary>
        public const string FileIdentifier = "HKRL";

        // Enum values are generated from hkrl.fbs (HKRL.Command, HKRL.LifecycleState,
        // HKRL.StatusCode, ...). Do not redefine them here — use the generated types.
    }
}
