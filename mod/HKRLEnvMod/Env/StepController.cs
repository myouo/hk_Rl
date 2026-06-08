using System;

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
        // TODO(phase-1): references to TcpServer, ObservationCollector, ActionApplier,
        // RewardEventBuffer, EpisodeLifecycle, ActionMasker, SimControl.

        /// <summary>Called once per FixedUpdate. Never blocks on the network.</summary>
        public void FixedTick()
        {
            // TODO(phase-1):
            //   1. Drain InboundRequests; keep only the latest action for this tick.
            //   2. Dispatch on command (STEP/RESET/PAUSE/RESUME/SET_TASK/SET_TIMESCALE/PING).
            //   3. If RUNNING and STEP: apply action (x action_repeat), advance sim.
            //   4. Collect observation + action mask + drained reward events.
            //   5. Encode StepResponse -> OutboundResponses.
            throw new NotImplementedException();
        }
    }
}
