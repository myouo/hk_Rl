# ADR-0008: Configured learner acceleration and task-wise normalization

- Status: Accepted
- Date: 2026-07-26

## Context

The released v0.8.0 Mod makes the game-side environment trustworthy enough to
shift attention to training efficiency. The remote APPO learner already rejects
excessively stale rollout versions, but its PyTorch update path used only FP32
and ordinary Adam, had no policy-drift stop, and normalized advantages across
all Boss tasks together. The latter lets a high-variance or large-reward task
set the scale for unrelated Bosses.

The project still requires local action inference: no optimization here may move
`observation -> action -> game` onto the remote network (ADR-0004).

## Decision

Add a config-driven `TorchLearnerRuntime` for learner-side APPO updates:

1. `amp_dtype: auto` enables BF16 on compatible CUDA devices and otherwise
   FP16 with `torch.amp.GradScaler`; CPU auto mode stays FP32.
2. `compile_mode` can compile `evaluate_actions` while retaining the original
   model object for stable checkpoint keys. Generic/local configs default off;
   the fixed-shape remote GPU profile selects `reduce-overhead`.
3. `fused_optimizer: auto` selects fused Adam only on CUDA and falls back
   safely elsewhere.
4. Advantages are normalized independently per `task_id` by default.
5. A configurable `target_kl` stops the remaining PPO epochs when the
   non-negative approximate KL exceeds the guardrail.
6. Every update reports AMP/compile/fused flags, task count, optimizer-step
   count, completed epochs, and KL early-stop state.

The settings remain typed YAML and live under `learner:`. Unsupported explicit
requests fail fast instead of silently falling back; `auto` modes are portable.

This follows PyTorch's official
[AMP](https://docs.pytorch.org/docs/stable/amp.html) and
[performance-tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
guidance. It does not claim that bounded-staleness clipped PPO is the final
distributed algorithm. Sequence-preserving recurrent APPO landed in
[ADR-0009](./0009-action-aligned-sequence-appo.md); V-trace remains the next
algorithm experiment, following the
[IMPALA paper](https://arxiv.org/abs/1802.01561).

## Consequences

- GPU learner throughput can improve without touching game physics or local
  inference latency.
- Multi-Boss batches no longer share one advantage scale.
- Aggressive epochs are cut short before a single update moves the policy too
  far.
- CPU tests and smoke runs retain deterministic FP32 behavior.
- CUDA speedup must be measured on the actual learner GPU; compile cold-start
  time and steady-state learner SPS are reported separately.
- ADR-0009 supersedes the former one-step recurrent limitation with
  episode-safe truncated BPTT. APPO still lacks V-trace, so recurrent PPO
  remains the strict on-policy sequence baseline.
