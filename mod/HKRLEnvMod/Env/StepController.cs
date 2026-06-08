using System.Collections.Generic;
using HKRLEnvMod.Action;
using HKRLEnvMod.Observation;
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
        private readonly ResetManager _resetManager;
        private readonly SimControl _simControl;
        private readonly ActionMasker _masker;
        private readonly Heartbeat _heartbeat;
        private readonly ObservationCollector _observations;
        private ulong _serverTick;

        public StepController(TcpServer server)
            : this(
                server,
                new ActionApplier(),
                new RewardEventBuffer(),
                new EpisodeLifecycle(),
                new ResetManager(),
                new SimControl(),
                new ActionMasker(),
                new Heartbeat(),
                new ObservationCollector())
        {
        }

        public StepController(
            TcpServer server,
            ActionApplier actions,
            RewardEventBuffer rewards,
            EpisodeLifecycle lifecycle,
            ResetManager resetManager,
            SimControl simControl,
            ActionMasker masker,
            Heartbeat heartbeat,
            ObservationCollector observations)
        {
            _server = server;
            _actions = actions;
            _rewards = rewards;
            _lifecycle = lifecycle;
            _resetManager = resetManager;
            _simControl = simControl;
            _masker = masker;
            _heartbeat = heartbeat;
            _observations = observations;
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

            var commandError = Dispatch(request);
            var state = AdvanceLifecycle();
            if (state == HKRL.LifecycleState.ClearEvents)
            {
                _rewards.Clear();
            }

            var rewardEvents = _rewards.Drain();
            if (!_lifecycle.IsRunning)
            {
                rewardEvents = System.Array.Empty<RewardEventRecord>();
            }
            else if (HasTerminalEvent(rewardEvents))
            {
                _lifecycle.RequestTerminate();
                state = _lifecycle.State;
            }

            var terminated = IsTerminal(state);
            var errorCode = commandError == HKRL.StatusCode.Ok
                ? _lifecycle.ErrorCode
                : commandError;
            var observation = _observations.Collect(request.TaskId, _lifecycle.EpisodeId);
            var response = MessageCodec.EncodeStepResponse(
                request,
                _serverTick,
                state,
                errorCode,
                rewardEvents,
                _masker.Compute(),
                terminated,
                truncated: false,
                episodeId: _lifecycle.EpisodeId,
                observation: observation);
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

        private HKRL.StatusCode Dispatch(DecodedStepRequest request)
        {
            try
            {
                _heartbeat.Touch((float)_serverTick);

                switch (request.Command)
                {
                    case HKRL.Command.Reset:
                    case HKRL.Command.SetTask:
                        _actions.Clear();
                        _rewards.Clear();
                        _resetManager.BeginReset(request.TaskId);
                        _lifecycle.RequestReset();
                        break;
                    case HKRL.Command.Step:
                        if (_lifecycle.IsRunning)
                        {
                            _actions.Apply(request.Action);
                        }
                        break;
                    case HKRL.Command.Pause:
                        _simControl.Pause();
                        break;
                    case HKRL.Command.Resume:
                        _simControl.Resume();
                        break;
                    case HKRL.Command.SetTimescale:
                        _simControl.SetTimeScale(request.TimeScale);
                        break;
                    case HKRL.Command.Ping:
                        break;
                }

                return HKRL.StatusCode.Ok;
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    $"Failed to dispatch command {request.Command}",
                    exception);
                return HKRL.StatusCode.InternalError;
            }
        }

        private HKRL.LifecycleState AdvanceLifecycle()
        {
            var resetReady = true;
            if (_resetManager.IsActive)
            {
                var resetStatus = _resetManager.Poll();
                if (resetStatus != HKRL.StatusCode.Ok)
                {
                    _resetManager.Clear();
                    _lifecycle.Fail(resetStatus);
                    return _lifecycle.State;
                }

                resetReady = _resetManager.IsComplete;
            }

            var state = _lifecycle.Tick(resetReady);
            if (state == HKRL.LifecycleState.Cleanup)
            {
                _resetManager.Clear();
            }

            return state;
        }

        private static bool HasTerminalEvent(IReadOnlyList<RewardEventRecord> rewardEvents)
        {
            for (var i = 0; i < rewardEvents.Count; i++)
            {
                var kind = rewardEvents[i].Kind;
                if (kind == HKRL.RewardEventKind.BossKilled
                    || kind == HKRL.RewardEventKind.PlayerDeath
                    || kind == HKRL.RewardEventKind.SceneChanged)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsTerminal(HKRL.LifecycleState state)
        {
            return state == HKRL.LifecycleState.Terminating
                || state == HKRL.LifecycleState.ReportDone;
        }
    }
}
