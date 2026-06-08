using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

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
        private const int MaxFrameBytes = 16 * 1024 * 1024;
        private static readonly byte[] AuthPrefix = Encoding.ASCII.GetBytes("HKRL_AUTH\0");

        private readonly IPAddress _address;
        private readonly int _port;
        private readonly string? _authToken;
        private readonly object _gate = new();

        private TcpListener? _listener;
        private TcpClient? _client;
        private Thread? _thread;
        private bool _running;

        /// <summary>Inbound StepRequest frames, drained by the main thread.</summary>
        public readonly ConcurrentQueue<byte[]> InboundRequests = new();

        /// <summary>Outbound StepResponse frames, filled by the main thread.</summary>
        public readonly ConcurrentQueue<byte[]> OutboundResponses = new();

        public TcpServer(string host = "127.0.0.1", int port = 5555, string? authToken = null)
        {
            if (authToken == string.Empty)
            {
                throw new ArgumentException("authToken must not be empty", nameof(authToken));
            }

            _address = ResolveAddress(host);
            _port = port;
            _authToken = authToken;
        }

        /// <summary>Start the network thread (accept + recv/send loop).</summary>
        public void Start()
        {
            lock (_gate)
            {
                if (_running)
                {
                    return;
                }

                _running = true;
                _listener = new TcpListener(_address, _port);
                _listener.Start();
                _thread = new Thread(Run)
                {
                    IsBackground = true,
                    Name = "HKRL TcpServer"
                };
                _thread.Start();
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (!_running)
                {
                    return;
                }

                _running = false;
                CloseClient();
                _listener?.Stop();
                _listener = null;
            }

            if (_thread != null && _thread.IsAlive)
            {
                _thread.Join(millisecondsTimeout: 500);
            }
            _thread = null;
        }

        private void Run()
        {
            while (_running)
            {
                try
                {
                    var listener = _listener;
                    if (listener == null)
                    {
                        return;
                    }

                    using var client = listener.AcceptTcpClient();
                    ConfigureClient(client);
                    _client = client;
                    ServeClient(client);
                }
                catch (SocketException)
                {
                    if (_running)
                    {
                        CloseClient();
                    }
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (IOException)
                {
                    CloseClient();
                }
                finally
                {
                    CloseClient();
                }
            }
        }

        private void ServeClient(TcpClient client)
        {
            using var stream = client.GetStream();
            var authenticated = _authToken == null;

            while (_running && client.Connected)
            {
                if (authenticated)
                {
                    DrainOutbound(stream);
                }

                if (!stream.DataAvailable)
                {
                    Thread.Sleep(1);
                    continue;
                }

                var payload = ReadFrame(stream);
                if (payload == null)
                {
                    return;
                }

                if (IsAuthFrame(payload))
                {
                    authenticated = ValidateAuthFrame(payload);
                    if (!authenticated)
                    {
                        return;
                    }
                    continue;
                }

                if (!authenticated)
                {
                    return;
                }

                InboundRequests.Enqueue(payload);
            }
        }

        private void DrainOutbound(NetworkStream stream)
        {
            while (_running && OutboundResponses.TryDequeue(out var frame))
            {
                stream.Write(frame, 0, frame.Length);
            }
        }

        private static byte[]? ReadFrame(NetworkStream stream)
        {
            var header = new byte[sizeof(int)];
            if (!ReadExact(stream, header, header.Length))
            {
                return null;
            }

            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(header);
            }

            var length = BitConverter.ToInt32(header, 0);
            if (length < 0 || length > MaxFrameBytes)
            {
                throw new IOException($"invalid frame length: {length}");
            }

            var payload = new byte[length];
            return ReadExact(stream, payload, payload.Length) ? payload : null;
        }

        private bool ValidateAuthFrame(byte[] payload)
        {
            if (_authToken == null)
            {
                return true;
            }

            var token = Encoding.UTF8.GetString(payload, AuthPrefix.Length, payload.Length - AuthPrefix.Length);
            return string.Equals(token, _authToken, StringComparison.Ordinal);
        }

        private static bool IsAuthFrame(byte[] payload)
        {
            if (payload.Length < AuthPrefix.Length)
            {
                return false;
            }

            for (var i = 0; i < AuthPrefix.Length; i++)
            {
                if (payload[i] != AuthPrefix[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static bool ReadExact(Stream stream, byte[] buffer, int length)
        {
            var offset = 0;
            while (offset < length)
            {
                var read = stream.Read(buffer, offset, length - offset);
                if (read == 0)
                {
                    return false;
                }

                offset += read;
            }

            return true;
        }

        private void CloseClient()
        {
            var client = _client;
            _client = null;
            client?.Close();
        }

        private static void ConfigureClient(TcpClient client)
        {
            client.NoDelay = true;
            client.ReceiveTimeout = 1000;
            client.SendTimeout = 1000;
        }

        private static IPAddress ResolveAddress(string host)
        {
            if (string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase))
            {
                return IPAddress.Loopback;
            }

            return IPAddress.Parse(host);
        }
    }
}
