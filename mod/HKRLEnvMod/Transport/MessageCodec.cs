using System;
using System.Collections.Generic;
using System.Text;
using Google.FlatBuffers;
using HKRLEnvMod.Rewards;

namespace HKRLEnvMod.Transport
{
    public readonly struct DecodedAction
    {
        public DecodedAction(byte movementX, byte aimY, uint buttons, byte durationIdx, short macroId)
        {
            MovementX = movementX;
            AimY = aimY;
            Buttons = buttons;
            DurationIdx = durationIdx;
            MacroId = macroId;
        }

        public byte MovementX { get; }
        public byte AimY { get; }
        public uint Buttons { get; }
        public byte DurationIdx { get; }
        public short MacroId { get; }

        public static DecodedAction Noop => new DecodedAction(1, 1, 0, 0, -1);
    }

    public sealed class DecodedStepRequest
    {
        public DecodedStepRequest(
            int schemaVersion,
            int envId,
            ulong tickId,
            HKRL.Command command,
            DecodedAction action,
            byte actionRepeat,
            long policyVersion,
            double clientTime,
            int taskId,
            float timeScale)
        {
            SchemaVersion = schemaVersion;
            EnvId = envId;
            TickId = tickId;
            Command = command;
            Action = action;
            ActionRepeat = actionRepeat;
            PolicyVersion = policyVersion;
            ClientTime = clientTime;
            TaskId = taskId;
            TimeScale = timeScale;
        }

        public int SchemaVersion { get; }
        public int EnvId { get; }
        public ulong TickId { get; }
        public HKRL.Command Command { get; }
        public DecodedAction Action { get; }
        public byte ActionRepeat { get; }
        public long PolicyVersion { get; }
        public double ClientTime { get; }
        public int TaskId { get; }
        public float TimeScale { get; }
    }

    /// <summary>
    /// Encode/decode FlatBuffers frames using the generated HKRL.* bindings
    /// (mod/HKRLEnvMod/Schema, run `make gen-schema-cs`). Framing = uint32-LE length
    /// prefix + payload (docs/protocol.md §1). Pure data; no Unity access, so it is
    /// safe to call from either thread, though we build responses on the main thread.
    /// </summary>
    public static class MessageCodec
    {
        private static readonly byte[] FileIdentifierBytes = Encoding.ASCII.GetBytes(Protocol.FileIdentifier);

        /// <summary>Decode a StepRequest frame (without the length prefix).</summary>
        public static DecodedStepRequest DecodeStepRequest(byte[] payload)
        {
            if (payload == null)
            {
                throw new ArgumentNullException(nameof(payload));
            }
            if (!HasFileIdentifier(payload))
            {
                throw new InvalidOperationException("StepRequest missing HKRL file identifier");
            }

            var request = HKRL.StepRequest.GetRootAsStepRequest(new ByteBuffer(payload));
            if (request.SchemaVersion != Protocol.SchemaVersion)
            {
                throw new InvalidOperationException(
                    $"schema mismatch: local={Protocol.SchemaVersion}, remote={request.SchemaVersion}");
            }

            return new DecodedStepRequest(
                request.SchemaVersion,
                request.EnvId,
                request.TickId,
                request.Command,
                DecodeAction(request.Action),
                request.ActionRepeat,
                request.PolicyVersion,
                request.ClientTime,
                request.TaskId,
                request.TimeScale);
        }

        /// <summary>Build a StepResponse into a length-prefixed frame.</summary>
        public static byte[] EncodeStepResponse(
            DecodedStepRequest request,
            ulong serverTick,
            HKRL.LifecycleState lifecycleState,
            HKRL.StatusCode errorCode = HKRL.StatusCode.Ok,
            IReadOnlyList<RewardEventRecord>? rewardEvents = null,
            bool[]? actionMask = null,
            bool terminated = false,
            bool truncated = false,
            string? info = null)
        {
            if (request == null)
            {
                throw new ArgumentNullException(nameof(request));
            }

            return EncodeStepResponse(
                request.EnvId,
                request.TickId,
                serverTick,
                lifecycleState,
                errorCode,
                rewardEvents,
                actionMask,
                terminated,
                truncated,
                info);
        }

        public static byte[] EncodeStepResponse(
            int envId,
            ulong tickId,
            ulong serverTick,
            HKRL.LifecycleState lifecycleState,
            HKRL.StatusCode errorCode = HKRL.StatusCode.Ok,
            IReadOnlyList<RewardEventRecord>? rewardEvents = null,
            bool[]? actionMask = null,
            bool terminated = false,
            bool truncated = false,
            string? info = null)
        {
            var builder = new FlatBufferBuilder(256);
            var rewardEventVector = BuildRewardEvents(builder, rewardEvents);
            var actionMaskVector = HKRL.StepResponse.CreateActionMaskVector(
                builder,
                actionMask ?? Array.Empty<bool>());
            var infoOffset = string.IsNullOrEmpty(info)
                ? default(StringOffset)
                : builder.CreateString(info);

            var response = HKRL.StepResponse.CreateStepResponse(
                builder,
                Protocol.SchemaVersion,
                envId,
                tickId,
                serverTick,
                default,
                rewardEventVector,
                actionMaskVector,
                terminated,
                truncated,
                lifecycleState,
                errorCode,
                infoOffset);
            HKRL.StepResponse.FinishStepResponseBuffer(builder, response);
            return AddLengthPrefix(builder.SizedByteArray());
        }

        private static DecodedAction DecodeAction(HKRL.Action? action)
        {
            if (!action.HasValue)
            {
                return DecodedAction.Noop;
            }

            var value = action.Value;
            return new DecodedAction(
                value.MovementX,
                value.AimY,
                value.Buttons,
                value.DurationIdx,
                value.MacroId);
        }

        private static VectorOffset BuildRewardEvents(
            FlatBufferBuilder builder,
            IReadOnlyList<RewardEventRecord>? rewardEvents)
        {
            if (rewardEvents == null || rewardEvents.Count == 0)
            {
                return HKRL.StepResponse.CreateRewardEventsVector(
                    builder,
                    Array.Empty<Offset<HKRL.RewardEvent>>());
            }

            var offsets = new Offset<HKRL.RewardEvent>[rewardEvents.Count];
            for (var i = 0; i < rewardEvents.Count; i++)
            {
                var rewardEvent = rewardEvents[i];
                offsets[i] = HKRL.RewardEvent.CreateRewardEvent(
                    builder,
                    rewardEvent.Kind,
                    rewardEvent.EntityId,
                    rewardEvent.Amount,
                    rewardEvent.AuxInt,
                    rewardEvent.AuxInt2);
            }

            return HKRL.StepResponse.CreateRewardEventsVector(builder, offsets);
        }

        private static byte[] AddLengthPrefix(byte[] payload)
        {
            var frame = new byte[sizeof(int) + payload.Length];
            var lengthBytes = BitConverter.GetBytes(payload.Length);
            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(lengthBytes);
            }

            Buffer.BlockCopy(lengthBytes, 0, frame, 0, sizeof(int));
            Buffer.BlockCopy(payload, 0, frame, sizeof(int), payload.Length);
            return frame;
        }

        private static bool HasFileIdentifier(byte[] payload)
        {
            if (payload.Length < sizeof(int) + FileIdentifierBytes.Length)
            {
                return false;
            }

            for (var i = 0; i < FileIdentifierBytes.Length; i++)
            {
                if (payload[sizeof(int) + i] != FileIdentifierBytes[i])
                {
                    return false;
                }
            }

            return true;
        }
    }
}
