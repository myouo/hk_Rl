using System;

namespace HKRLEnvMod.Transport
{
    /// <summary>
    /// Encode/decode FlatBuffers frames using the generated HKRL.* bindings
    /// (mod/HKRLEnvMod/Schema, run `make gen-schema-cs`). Framing = uint32-LE length
    /// prefix + payload (docs/protocol.md §1). Pure data; no Unity access, so it is
    /// safe to call from either thread, though we build responses on the main thread.
    /// </summary>
    public static class MessageCodec
    {
        /// <summary>Decode a StepRequest frame (without the length prefix).</summary>
        public static void DecodeStepRequest(byte[] payload /*, out fields */)
        {
            // TODO(phase-1): HKRL.StepRequest.GetRootAsStepRequest(ByteBuffer).
            throw new NotImplementedException();
        }

        /// <summary>Build a StepResponse into a length-prefixed frame.</summary>
        public static byte[] EncodeStepResponse(/* snapshot, events, mask, flags */)
        {
            // TODO(phase-1): FlatBufferBuilder; prefix with uint32-LE length.
            throw new NotImplementedException();
        }
    }
}
