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
        private readonly ObservationRewardTracker _rewardTracker;
        private readonly EpisodeLifecycle _lifecycle;
        private readonly ResetManager _resetManager;
        private readonly SimControl _simControl;
        private readonly ActionMasker _masker;
        private readonly Heartbeat _heartbeat;
        private readonly ObservationCollector _observations;
        private ulong _serverTick;
        private ulong _episodeStartServerTick;
        private ulong _runningEpisodeId;
        private long _observedSessionId;
        private bool _observedHasClient;
        private DecodedStepRequest? _repeatRequest;
        private long _repeatSessionId;
        private int _repeatTicksRemaining;

        public StepController(TcpServer server)
            : this(
                server,
                new ActionApplier(),
                new RewardEventBuffer(),
                new ObservationRewardTracker(),
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
            ObservationRewardTracker rewardTracker,
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
            _rewardTracker = rewardTracker;
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
            try
            {
                FixedTickCore();
            }
            catch (System.Exception exception)
            {
                ClearControlState();
                global::HKRLEnvMod.Debug.Logger.Error("StepController FixedTick failed", exception);
            }
        }

        private void FixedTickCore()
        {
            _serverTick++;
            RefreshConnectionControlState();

            var pendingRequest = DrainLatestRequest();
            var commandError = HKRL.StatusCode.Ok;
            HKRL.LifecycleState state;
            if (pendingRequest.HasValue)
            {
                var request = pendingRequest.Value.Request;
                var sessionId = pendingRequest.Value.SessionId;
                if (!_server.IsCurrentSession(sessionId))
                {
                    return;
                }

                ClearControlState();
                commandError = Dispatch(request);
                state = ShouldAdvanceLifecycle(request)
                    ? AdvanceLifecycle()
                    : _lifecycle.State;
                if (ShouldDelayStepResponse(request, commandError, state))
                {
                    _repeatRequest = request;
                    _repeatSessionId = sessionId;
                    _repeatTicksRemaining = request.ActionRepeat - 1;
                    return;
                }

                EnqueueStepResponse(sessionId, request, commandError, state);
            }
            else
            {
                var request = _repeatRequest;
                if (request == null)
                {
                    return;
                }
                var sessionId = _repeatSessionId;
                if (!_server.IsCurrentSession(sessionId))
                {
                    ClearControlState();
                    return;
                }

                commandError = ApplyRepeatedStep(request);
                state = AdvanceLifecycle();
                if (ShouldContinueRepeat(commandError, state))
                {
                    return;
                }

                CancelRepeatedStep();
                EnqueueStepResponse(sessionId, request, commandError, state);
            }
        }

        private void RefreshConnectionControlState()
        {
            var currentSessionId = _server.CurrentSessionId;
            var hasClient = _server.HasClient;
            if (currentSessionId == _observedSessionId && hasClient == _observedHasClient)
            {
                return;
            }

            _observedSessionId = currentSessionId;
            _observedHasClient = hasClient;
            ClearControlState();
        }

        private void EnqueueStepResponse(
            long sessionId,
            DecodedStepRequest request,
            HKRL.StatusCode commandError,
            HKRL.LifecycleState state)
        {
            if (!_server.IsCurrentSession(sessionId))
            {
                return;
            }

            if (state == HKRL.LifecycleState.ClearEvents)
            {
                _rewards.Clear();
                _rewardTracker.Reset();
            }

            var observation = _observations.Collect(
                request.TaskId,
                _lifecycle.EpisodeId,
                CurrentEpisodeTime(state));
            if (_lifecycle.IsRunning)
            {
                _rewardTracker.Update(observation, _rewards);
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
            var actionMask = _masker.Compute(
                ToPlayerActionState(observation.Player),
                request.EnableMacroActions,
                MacroCountFor(request));
            var response = MessageCodec.EncodeStepResponse(
                request,
                _serverTick,
                state,
                errorCode,
                rewardEvents,
                actionMask,
                terminated,
                truncated: false,
                episodeId: _lifecycle.EpisodeId,
                observation: observation);
            _server.EnqueueResponse(sessionId, response);
        }

        private bool ShouldDelayStepResponse(
            DecodedStepRequest request,
            HKRL.StatusCode commandError,
            HKRL.LifecycleState state)
        {
            return request.Command == HKRL.Command.Step
                && commandError == HKRL.StatusCode.Ok
                && state == HKRL.LifecycleState.Running
                && request.ActionRepeat > 1
                && !BufferedTerminalEvent();
        }

        private static bool ShouldAdvanceLifecycle(DecodedStepRequest request)
        {
            return request.Command == HKRL.Command.Reset
                || request.Command == HKRL.Command.SetTask
                || request.Command == HKRL.Command.Step;
        }

        private bool ShouldContinueRepeat(
            HKRL.StatusCode commandError,
            HKRL.LifecycleState state)
        {
            return commandError == HKRL.StatusCode.Ok
                && state == HKRL.LifecycleState.Running
                && _repeatTicksRemaining > 0
                && !BufferedTerminalEvent();
        }

        private PendingRequest? DrainLatestRequest()
        {
            PendingRequest? latest = null;
            while (_server.InboundRequests.TryDequeue(out var frame))
            {
                try
                {
                    var request = MessageCodec.DecodeStepRequest(frame.Payload);
                    latest = new PendingRequest(frame.SessionId, request);
                }
                catch (SchemaMismatchException exception)
                {
                    EnqueueDecodeError(
                        frame.SessionId,
                        HKRL.StatusCode.SchemaMismatch,
                        exception.Message);
                }
                catch (System.Exception exception)
                {
                    EnqueueDecodeError(
                        frame.SessionId,
                        HKRL.StatusCode.InternalError,
                        exception.Message);
                }
            }

            return latest;
        }

        private void EnqueueDecodeError(long sessionId, HKRL.StatusCode errorCode, string info)
        {
            var response = MessageCodec.EncodeStepResponse(
                envId: 0,
                tickId: 0,
                serverTick: _serverTick,
                lifecycleState: _lifecycle.State,
                errorCode: errorCode,
                info: info,
                episodeId: _lifecycle.EpisodeId);
            _server.EnqueueResponse(sessionId, response);
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
                        ClearControlState();
                        _rewards.Clear();
                        _rewardTracker.Reset();
                        _runningEpisodeId = 0;
                        _resetManager.BeginReset(request.TaskId);
                        _lifecycle.RequestReset();
                        break;
                    case HKRL.Command.Step:
                        if (!_lifecycle.IsRunning)
                        {
                            return IsNoopPollStep(request)
                                ? HKRL.StatusCode.Ok
                                : HKRL.StatusCode.NotRunning;
                        }

                        ReportInvalidAction(request);
                        _actions.Apply(request.Action);
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

        private void CancelRepeatedStep()
        {
            _repeatRequest = null;
            _repeatSessionId = 0;
            _repeatTicksRemaining = 0;
        }

        private void ClearControlState()
        {
            CancelRepeatedStep();
            _actions.Clear();
        }

        private readonly struct PendingRequest
        {
            public PendingRequest(long sessionId, DecodedStepRequest request)
            {
                SessionId = sessionId;
                Request = request;
            }

            public long SessionId { get; }
            public DecodedStepRequest Request { get; }
        }

        private static bool IsNoopPollStep(DecodedStepRequest request)
        {
            var action = request.Action;
            return request.ActionRepeat == 1
                && action.MovementX == DecodedAction.Noop.MovementX
                && action.AimY == DecodedAction.Noop.AimY
                && action.Buttons == DecodedAction.Noop.Buttons
                && action.DurationIdx == DecodedAction.Noop.DurationIdx
                && action.MacroId == DecodedAction.Noop.MacroId;
        }

        private HKRL.StatusCode ApplyRepeatedStep(DecodedStepRequest request)
        {
            try
            {
                if (_lifecycle.IsRunning)
                {
                    _actions.Apply(request.Action);
                }

                if (_repeatTicksRemaining > 0)
                {
                    _repeatTicksRemaining--;
                }

                return HKRL.StatusCode.Ok;
            }
            catch (System.Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to apply repeated STEP action",
                    exception);
                return HKRL.StatusCode.InternalError;
            }
        }

        private void ReportInvalidAction(DecodedStepRequest request)
        {
            var action = request.Action;
            if (action.MovementX > 2)
            {
                AddInvalidActionEvent(actionId: 0, reason: 1);
            }
            if (action.AimY > 2)
            {
                AddInvalidActionEvent(actionId: 1, reason: 1);
            }
            if ((action.Buttons & ~PrimitiveInput.ButtonMask) != 0)
            {
                AddInvalidActionEvent(actionId: 2, reason: 2);
            }
            if (action.DurationIdx > 3)
            {
                AddInvalidActionEvent(actionId: 3, reason: 1);
            }
            var macroLimit = MacroCountFor(request);
            if (request.EnableMacroActions)
            {
                if (request.NMacroActions < 0 || action.MacroId < -1 || action.MacroId >= macroLimit)
                {
                    AddInvalidActionEvent(actionId: 4, reason: 1);
                }
            }
            else if (action.MacroId >= 0)
            {
                AddInvalidActionEvent(actionId: 4, reason: 1);
            }
        }

        private static int MacroCountFor(DecodedStepRequest request)
        {
            if (!request.EnableMacroActions || request.NMacroActions < 0)
            {
                return 0;
            }

            return request.NMacroActions > ActionMasker.DefaultMacroCount
                ? ActionMasker.DefaultMacroCount
                : request.NMacroActions;
        }

        private void AddInvalidActionEvent(int actionId, int reason)
        {
            _rewards.Add(
                HKRL.RewardEventKind.InvalidAction,
                auxInt: actionId,
                auxInt2: reason);
        }

        private static PlayerActionState ToPlayerActionState(PlayerObservation player)
        {
            return new PlayerActionState(
                dashCooldown: player.DashCooldown,
                soul: player.Soul,
                attackLockTimer: player.AttackLockTimer,
                castLockTimer: player.CastLockTimer,
                onGround: player.OnGround,
                doubleJumpAvailable: player.DoubleJumpAvailable,
                focusing: player.FocusState > 0,
                canAttack: player.CanAttack,
                canCast: player.CanCast,
                canFocus: player.CanFocus);
        }

        private float CurrentEpisodeTime(HKRL.LifecycleState state)
        {
            if (state != HKRL.LifecycleState.Running)
            {
                return 0.0f;
            }

            if (_runningEpisodeId != _lifecycle.EpisodeId)
            {
                _runningEpisodeId = _lifecycle.EpisodeId;
                _episodeStartServerTick = _serverTick;
                return 0.0f;
            }

            var elapsedTicks = _serverTick >= _episodeStartServerTick
                ? _serverTick - _episodeStartServerTick
                : 0;
            return elapsedTicks * UnityEngine.Time.fixedDeltaTime;
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

        private bool BufferedTerminalEvent()
        {
            return _rewards.Contains(IsTerminalEvent);
        }

        private static bool HasTerminalEvent(IReadOnlyList<RewardEventRecord> rewardEvents)
        {
            for (var i = 0; i < rewardEvents.Count; i++)
            {
                if (IsTerminalEvent(rewardEvents[i]))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsTerminalEvent(RewardEventRecord rewardEvent)
        {
            var kind = rewardEvent.Kind;
            return kind == HKRL.RewardEventKind.BossKilled
                || kind == HKRL.RewardEventKind.PlayerDeath
                || kind == HKRL.RewardEventKind.SceneChanged;
        }

        private static bool IsTerminal(HKRL.LifecycleState state)
        {
            return state == HKRL.LifecycleState.Terminating
                || state == HKRL.LifecycleState.ReportDone;
        }
    }
}
