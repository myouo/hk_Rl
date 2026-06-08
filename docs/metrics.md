# Metrics

> Implements PRD §13 + §9.6. Code: `hkrl/utils/metrics.py`, `hkrl/utils/logging.py`.

## 1. Must-record

```text
episode_reward        win_rate              episode_length
damage_dealt          damage_taken          heal_count
heal_amount           death_rate            death_reason
time_to_kill          invalid_action_ratio  action_entropy
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
Because evaluator output is keyed by task/boss, each task record includes
`per_boss_win_rate` as an alias of `win_rate` and `per_boss_damage_ratio` as
`damage_taken / damage_dealt` with a zero value when no damage was dealt.
`scripts/run_eval.py --replay-jsonl FILE` can additionally emit per-step replay
records with task/seed/episode/step, action, reward, terminal flags, and
event-derived metrics. Replay JSONL is debugging evidence; capability decisions
still use the aggregated shaping-free metrics above.

## 3. SPS, not FPS (PRD §9.6)

High game FPS ≠ efficient training. Track **samples per second**. Levers:
`Time.timeScale` / `fixedDeltaTime`, `action_repeat`, parallel instances, reduced
render quality, fast reset. `HKRLEnv.set_timescale(scale)` sends the protocol
command that mod `SimControl` applies on the Unity main thread. `reset_duration`
is a first-class SPS factor.

## 4. Backends

`logging.py` abstracts the sink (stdout/JSONL/CSV always; TensorBoard / WandB
optional via the `logging` extra). Every episode emits a complete JSONL/CSV
record (PRD §2.1, Phase 2 milestone): reward, damage dealt/taken, win/loss,
length, SPS, reset status. The default CSV sink uses stable
`type,step,key,value,record` columns; episode payloads are stored as compact JSON
in `record`, while custom `fieldnames` can produce a fixed wide export. The
stdout sink emits the same scalar/episode payloads as JSON lines.
