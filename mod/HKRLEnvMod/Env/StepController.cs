using HKRLEnvMod.Action;
using HKRLEnvMod.Rewards;
using HKRLEnvMod.Transport;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Heart of the environment server, driven from Unity FixedUpdate on the MAIN
    /// THREAD (docs/mod_dev.md §5). Each tick: dequeue the latest action, apply it,
    /// collect observation + reward events, and enqueue a StepResponse. Honors the
    /// command (STEP/RESET/PAUSE/...) and the episode lifecycle.
    /// </summary>
    public sealed class StepController
    {
        private readonly TcpServer _server;
        private readonly ActionApplier _actions;
        private readonly RewardEventBuffer _rewards;
        private readonly EpisodeLifecycle _lifecycle;
        private readonly ActionMasker _masker;
        private readonly Heartbeat _heartbeat;
        private ulong _serverTick;

        public StepController(TcpServer server)
            : this(
                server,
                new ActionApplier(),
                new RewardEventBuffer(),
                new EpisodeLifecycle(),
                new ActionMasker(),
                new Heartbeat())
        {
        }

        public StepController(
            TcpServer server,
            ActionApplier actions,
            RewardEventBuffer rewards,
            EpisodeLifecycle lifecycle,
            ActionMasker masker,
            Heartbeat heartbeat)
        {
            _server = server;
            _actions = actions;
            _rewards = rewards;
            _lifecycle = lifecycle;
            _masker = masker;
            _heartbeat = heartbeat;
        }

        /// <summary>Called once per FixedUpdate. Never blocks on the network.</summary>
        public void FixedTick()
        {
            _serverTick++;
            var request = DrainLatestRequest();
            if (request == null)
            {
                return;
            }

            Dispatch(request);
            var state = _lifecycle.Tick();
            if (state == HKRL.LifecycleState.ClearEvents)
            {
                _rewards.Clear();
            }

            var terminated = IsTerminal(state);
            var response = MessageCodec.EncodeStepResponse(
                request,
                _serverTick,
                state,
                _lifecycle.ErrorCode,
                _rewards.Drain(),
                _masker.Compute(),
                terminated,
                truncated: false,
                episodeId: _lifecycle.EpisodeId);
            _server.OutboundResponses.Enqueue(response);
        }

        private DecodedStepRequest? DrainLatestRequest()
        {
            DecodedStepRequest? latest = null;
            while (_server.InboundRequests.TryDequeue(out var payload))
            {
                try
                {
                    latest = MessageCodec.DecodeStepRequest(payload);
                }
                catch (System.Exception exception)
                {
                    var response = MessageCodec.EncodeStepResponse(
                        envId: 0,
                        tickId: 0,
                        serverTick: _serverTick,
                        lifecycleState: _lifecycle.State,
                        errorCode: HKRL.StatusCode.InternalError,
                        info: exception.Message,
                        episodeId: _lifecycle.EpisodeId);
                    _server.OutboundResponses.Enqueue(response);
                }
            }

            return latest;
        }

        private void Dispatch(DecodedStepRequest request)
        {
            _heartbeat.Touch((float)_serverTick);

            switch (request.Command)
            {
                case HKRL.Command.Reset:
                case HKRL.Command.SetTask:
                    _actions.Clear();
                    _rewards.Clear();
                    _lifecycle.RequestReset();
                    break;
                case HKRL.Command.Step:
                    if (_lifecycle.IsRunning)
                    {
                        _actions.Apply(request.Action);
                    }
                    break;
                case HKRL.Command.Ping:
                case HKRL.Command.Pause:
                case HKRL.Command.Resume:
                case HKRL.Command.SetTimescale:
                    break;
            }
        }

        private static bool IsTerminal(HKRL.LifecycleState state)
        {
            return state == HKRL.LifecycleState.Terminating
                || state == HKRL.LifecycleState.ReportDone;
        }
    }
}
