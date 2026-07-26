using System;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Pure readiness predicates shared by the live Unity adapters and runtime
    /// smoke tests. They fail closed when a required physics component is absent.
    /// </summary>
    internal static class EpisodeReadiness
    {
        public const string ReadyHeroTransitionState = "WAITING_TO_TRANSITION";

        public static bool IsHeroReady(
            bool active,
            bool acceptingInput,
            bool controlRelinquished,
            bool gameplayState,
            bool transitioning,
            string transitionState,
            bool hasBody,
            float gravityScale,
            bool bodyKinematic,
            bool bodySimulated,
            bool positionConstraintsFree,
            bool hasCollider,
            bool colliderEnabled,
            bool tilemapTestActive,
            bool groundedJumpReady = true)
        {
            return active
                && acceptingInput
                && !controlRelinquished
                && gameplayState
                && !transitioning
                && transitionState == ReadyHeroTransitionState
                && hasBody
                && gravityScale > 0.0f
                && !float.IsNaN(gravityScale)
                && !float.IsInfinity(gravityScale)
                && !bodyKinematic
                && bodySimulated
                && positionConstraintsFree
                && hasCollider
                && colliderEnabled
                && tilemapTestActive
                && groundedJumpReady;
        }
    }

    /// <summary>
    /// Requires readiness to remain true over real (unscaled) time. This filters
    /// the transient frame between a GG transition event and its PlayMaker FSM
    /// actually taking control.
    /// </summary>
    internal sealed class ReadyStabilityGate
    {
        private readonly float _minimumSeconds;
        private float _readySince = -1.0f;

        public ReadyStabilityGate(float minimumSeconds)
        {
            if (minimumSeconds < 0.0f
                || float.IsNaN(minimumSeconds)
                || float.IsInfinity(minimumSeconds))
            {
                throw new ArgumentOutOfRangeException(nameof(minimumSeconds));
            }

            _minimumSeconds = minimumSeconds;
        }

        public bool Observe(bool ready, float unscaledTime)
        {
            if (!ready || float.IsNaN(unscaledTime) || float.IsInfinity(unscaledTime))
            {
                Reset();
                return false;
            }

            if (_readySince < 0.0f || unscaledTime < _readySince)
            {
                _readySince = unscaledTime;
                return _minimumSeconds <= 0.0f;
            }

            return unscaledTime - _readySince >= _minimumSeconds;
        }

        public void Reset()
        {
            _readySince = -1.0f;
        }
    }
}
