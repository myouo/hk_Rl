using System;
using System.Collections.Concurrent;

namespace HKRLEnvMod.Transport
{
    /// <summary>
    /// TCP environment server (docs/protocol.md). Runs the accept/recv/send loop on
    /// a DEDICATED NETWORK THREAD. It only marshals length-prefixed FlatBuffers
    /// frames between the socket and thread-safe queues — it NEVER touches Unity
    /// objects (docs/mod_dev.md §5). Bind to localhost/LAN only (PRD §9.10).
    /// </summary>
    public sealed class TcpServer : IDisposable
    {
        /// <summary>Inbound StepRequest frames, drained by the main thread.</summary>
        public readonly ConcurrentQueue<byte[]> InboundRequests = new();

        /// <summary>Outbound StepResponse frames, filled by the main thread.</summary>
        public readonly ConcurrentQueue<byte[]> OutboundResponses = new();

        public TcpServer(string host = "127.0.0.1", int port = 5555)
        {
            // TODO(phase-1): TcpListener, optional token-auth handshake.
        }

        /// <summary>Start the network thread (accept + recv/send loop).</summary>
        public void Start()
        {
            // TODO(phase-1): background thread reading uint32-LE length prefix then
            // payload into InboundRequests; writing OutboundResponses to the socket.
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            // TODO(phase-1): stop thread, close socket.
        }
    }
}
