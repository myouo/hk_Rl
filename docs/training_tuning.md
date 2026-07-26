# Training configuration and tuning

This project has two deliberately separate configuration layers.

1. [`configs/experiments/godhome_smart.yaml`](../configs/experiments/godhome_smart.yaml)
   is the aggregate launch manifest: task list, role configs, endpoints, worker
   identity, and output paths.
2. [`configs/base.yaml`](../configs/base.yaml) plus `configs/train/*.yaml` are
   the composed, typed `TrainConfig`: algorithm, model, optimizer, rollout, and
   learner-runtime hyperparameters.
3. `configs/tasks/*.yaml` owns Boss identity and any arena-specific overrides.
   The catalog-generated Hall of Gods tasks compose their common action,
   observation, episode, and reward settings from
   `configs/task_defaults/godhome_training.yaml`. The active three-Boss
   curriculum overlays `configs/task_defaults/godhome_engagement_v1.yaml`;
   inactive catalog tasks retain the baseline profile.

Unknown keys fail validation. Do not copy hyperparameters into the experiment
manifest; point both roles at composed train configs with the same training
contract.

## Live tuning without a restart

The authenticated checkpoint registry also exposes a narrow, loopback-only
control endpoint. `scripts/tune_training.py` submits a complete, monotonically
versioned override snapshot. The learner applies it only between updates,
publishes it in a hash-verified checkpoint, and the local worker applies that
checkpoint only after discarding any unfinished rollout prefix. Reward changes
therefore never mix two objectives in one uploaded batch.

Keep the existing SSH forward for port 5601 and load the same token used by the
worker:

```bash
set -a
source runs/remote-training/worker.env
set +a

# Inspect the requested and learner-applied versions.
python scripts/tune_training.py --show

# Change several safe knobs atomically and wait for learner acknowledgement.
python scripts/tune_training.py \
  --set reward.boss_damage=1.0 \
  --set reward.player_death=-15 \
  --set learner.entropy_coef=0.02 \
  --set worker.time_scale=3.0 \
  --note "more engagement, less edge camping" \
  --wait-applied 30

# Return one field to its startup YAML value while retaining other overrides.
python scripts/tune_training.py \
  --unset reward.player_death \
  --wait-applied 30

# Return every live field to its startup YAML value.
python scripts/tune_training.py --reset --wait-applied 30
```

The CLI accepts these fields:

- `reward.*`: `boss_damage`, `player_damage`, `soul_gained`, `heal_amount`,
  `boss_kill`, `player_death`, `time_penalty`, `invalid_action`
- learner: `learning_rate`, `entropy_coef`, `value_coef`, `clip_range`,
  `max_grad_norm`, `target_kl` (`off` disables the KL stop)
- worker: `time_scale`

The learner checks the request after each upload and on its idle intake timeout
(10 seconds by default). The worker checks for the resulting checkpoint in a
background thread every `local.checkpoint_poll_interval_s` (2 seconds in the
aggregate manifest), so remote I/O never stalls the local action loop.
`--wait-applied` confirms the learner boundary; the worker's next heartbeat
confirms the same `tuning_version` on the game host. A worker may discard one
unfinished rollout fragment, but it never uploads a mixed-version batch.

Tensor/layout and sampling-geometry fields remain restart-only:
`rollout_steps`, `minibatch_size`, `epochs`, `sequence_length`, `burn_in`,
`batches_per_update`, model dimensions, observation/action layouts,
`action_repeat`, transport, and task enrollment. Changing those fields requires
a new composed YAML and a controlled restart because existing buffers, compiled
graphs, or checkpoints may have incompatible shapes.

Each request and applied acknowledgement is atomically persisted as
`live_tuning.json` / `live_tuning_status.json` in the checkpoint registry. The
full snapshot is embedded in every later checkpoint, so learner and worker
restarts preserve the active values. `tuning_version` is also carried by every
rollout; the learner rejects batches from a different objective version.

## Where each knob lives

| Goal | Setting | File to tune |
|---|---|---|
| temporal horizon | `gamma`, `gae_lambda` | `configs/base.yaml` or train override |
| PPO trust region | `clip_range`, `learner.target_kl` | train config |
| optimizer | `learning_rate`, `max_grad_norm` | train config |
| exploration | `entropy_coef` | train/task-profile override |
| critic balance | `value_coef` | train config |
| collection/update size | `rollout_steps`, `minibatch_size`, `epochs` | train config |
| GRU credit horizon | `sequence_length` | train config |
| local recurrent PPO warm-up | `burn_in` | recurrent-PPO config |
| remote GPU aggregation | `learner.batches_per_update` | remote train config |
| stale rollout bound | `learner.max_staleness` | remote train config |
| checkpoint cadence | `learner.publish_every_updates` | remote train config |
| CUDA acceleration | `amp_dtype`, `amp_init_scale`, `compile_mode`, `fused_optimizer` | remote train config |
| model capacity | `model.entity_hidden`, attention layers/heads, `rnn_hidden` | train config |
| game decision cadence | `action.action_repeat` | task config |
| simulation wall-clock multiplier | `local.time_scale` | experiment manifest |
| background tuning latency | `local.checkpoint_poll_interval_s` | experiment manifest |
| reward composition | `reward.*` | task config |

APPO sequence chunks start from the exact recorded behavior hidden state, so its
remote profile requires `burn_in: 0` (other values fail validation). Local
`recurrent_ppo` may still use burn-in when a chunk begins before its loss
window.
Variable duration and macro choices use semi-Markov GAE:
`discount_exponent = elapsed_ticks / task.action.action_repeat`. `gamma` and
`gae_lambda` are therefore defined per base task decision interval, not per
option regardless of its duration. Time-limit truncation bootstraps from the
terminal observation but ends the trace before the reset episode.

## Current production starting point

The remote profile groups four 2,048-transition worker rollouts into one
8,192-transition update. It trains fixed 32-step sequences with
`minibatch_size: 1024`, `epochs: 2`, task-wise advantage normalization, a
`target_kl: 0.03` guard, AMP, fused Adam, and `torch.compile` on the CUDA role.
AMP `auto` selects native BF16 on Ampere-or-newer GPUs and FP16 plus GradScaler
on Volta GPUs such as V100; it does not treat a software BF16 fallback as native
support.
Live scene/actor/prefab/FSM hashes are signed int32-scale values. The structured
encoders remove those columns from continuous MLP inputs and use field-specific
4,096-bucket embeddings, keeping V100 FP16 forwards finite without discarding
Boss/state identity.
The SSH profile starts FP16 scaling at 1,024 on Volta. Recoverable scaled-gradient
overflow skips only that optimizer step, lowers the scale, and reports
`amp_step_skipped` plus exact succeeded/skipped step counts.
If the installed Python/PyTorch pair cannot run TorchDynamo (notably PyTorch
2.3 on Python 3.12), launch
`configs/experiments/godhome_smart_eager.yaml`. It keeps CUDA AMP and fused Adam
but selects the explicit eager learner role instead of deferring failure until
the first update.
The last sequence minibatch is padding-masked to keep the compiled learner shape
stable. The four batches may come from four game instances or sequentially from
fewer workers; four parallel instances reduce update latency when the game host
can sustain them.

The current engagement-v1 comparison changes reward weights only: damage dealt
remains `1.0`, while player damage/death are reduced to `-2`/`-20`.
`gamma=0.995`, `gae_lambda=0.95`, and `entropy_coef=0.01` stay fixed for the
first controlled comparison. Start it from a fresh optimizer/checkpoint series
and preserve the death-heavy baseline.

Keep `fixedDeltaTime` at 0.02 seconds: it is the 50 Hz physics integration step,
not an FPS cap. Tune `local.time_scale` with
`scripts/live_performance_benchmark.py` (for example 1×, 2×, then 3×) and retain
the fastest value that preserves reset success, action validity, and game SPS.
The local launcher restores 1× when a finite run exits normally.

All 44 catalog Bosses have standalone files under `configs/tasks/`, but merely
creating a task file does not enroll it. The active task set remains the explicit
three-entry list in `configs/experiments/godhome_smart.yaml`; add tasks there only
when their curriculum stage and fixed-seed evaluation gate are ready.

Tune from fixed-seed evaluation, not reward:

1. Hold task set/seeds and Mod/game settings constant.
2. Compare game SPS and learner samples/s separately.
3. Reject changes that lower per-Boss win rate or raise damage taken, even if
   training reward rises.
4. Change one family at a time. First adjust batch/epoch/learning rate, then
   sequence length/model capacity, then reward/exploration.

If policy KL repeatedly hits the guard, reduce learning rate or epochs. If CUDA
is underutilized and KL is healthy, increase `minibatch_size` or
`batches_per_update`. If workers are frequently rejected as stale, reduce
checkpoint interval or worker group size; do not simply raise staleness without
evaluating off-policy drift.

## Launch

On the remote GPU host:

```bash
export HKRL_AUTH_TOKEN='the-shared-secret'
python scripts/run_remote_training.py \
  --experiment configs/experiments/godhome_smart.yaml
```

On the game host, after the Mod and SSH forwards are ready:

```bash
# Keep the rollout/checkpoint services private; replace USER@GPU_HOST.
ssh -N \
  -L 5600:127.0.0.1:5600 \
  -L 5601:127.0.0.1:5601 \
  USER@GPU_HOST

export HKRL_AUTH_TOKEN='the-shared-secret'
python scripts/run_local_inference.py \
  --experiment configs/experiments/godhome_smart.yaml
```

Both support `--dry-run`, which validates and prints a secret-free resolved
plan. The local script never sends per-step observations to the remote GPU:
`obs -> local policy -> action -> game` remains on the game host.

## Algorithm status

- PPO MLP remains the single-Boss ablation floor.
- Recurrent PPO remains the strict on-policy sequence baseline.
- APPO now performs episode-safe truncated BPTT, bounded-staleness clipped PPO,
  task-wise normalization, and multi-worker GPU batching.
- V-trace/IMPALA is intentionally not enabled yet. Add it as a registry-selected
  algorithm only after policy-lag measurements justify the extra off-policy
  correction and its discount/bootstrap tests are complete.

See [ADR-0009](./adr/0009-action-aligned-sequence-appo.md) for the action and
sequence correctness decision.
