# Metrics

> Implements PRD §13 + §9.6. Code: `hkrl/utils/metrics.py`, `hkrl/utils/logging.py`.

## 1. Must-record

```text
episode_reward        win_rate              episode_length
damage_dealt          damage_taken          heal_count
death_reason          invalid_action_ratio  action_entropy
policy_kl             value_loss            policy_loss
explained_variance    SPS                   reset_success_rate
reset_duration        worker_crash_count
per_boss_win_rate     per_boss_damage_ratio
```

## 2. Reward is not capability (PRD §13)

> Training reward ≠ real ability.

Decisions are driven by **shaping-free** metrics, in priority order:

1. per-boss win rate
2. damage taken
3. time to kill
4. invalid action ratio
5. generalization to untrained bosses
6. old-task regression (catastrophic forgetting)

The evaluator ([`../python/hkrl/eval/evaluator.py`](../python/hkrl/eval/evaluator.py))
computes these on fixed seeds/tasks, isolated from training, to catch the
"reward up, win rate down" failure (PRD §9.4).

## 3. SPS, not FPS (PRD §9.6)

High game FPS ≠ efficient training. Track **samples per second**. Levers:
`Time.timeScale` / `fixedDeltaTime`, `action_repeat`, parallel instances, reduced
render quality, fast reset. `reset_duration` is a first-class SPS factor.

## 4. Backends

`logging.py` abstracts the sink (stdout/JSONL/CSV always; TensorBoard / WandB
optional via the `logging` extra). Every episode emits a complete JSONL/CSV
record (PRD §2.1, Phase 2 milestone): reward, damage dealt/taken, win/loss,
length, SPS, reset status.
