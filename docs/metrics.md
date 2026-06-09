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
worker_learner_upload_submitted_batches
worker_learner_upload_accepted_batches
worker_learner_upload_rejected_batches
worker_learner_upload_failed_batches
worker_policy_lag_max worker_checkpoint_lag_max
stale_policy_worker_count stale_checkpoint_worker_count
recovering_worker_count
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
Regression reports accept either `win_rate` or `per_boss_win_rate` for baseline
and current metrics.
`scripts/run_eval.py --replay-jsonl FILE` can additionally emit per-step replay
records with task/seed/episode/step, action, reward, terminal flags, and
event-derived metrics. Replay JSONL is debugging evidence; capability decisions
still use the aggregated shaping-free metrics above.
`scripts/run_eval.py --eval-workers N --ports P0 P1 ...` evaluates tasks through
a task-level worker pool so multi-boss regression checks can use multiple live
env instances when available. The default is `1` worker and the single `--port`
value to preserve deterministic single-instance behavior.
`scripts/render_eval_report.py --eval-json runs/eval.json` renders the fixed-seed
metrics and optional regression deltas into stable JSON/Markdown release
artifacts. If `win_rate` is absent or invalid for a task, the report uses
`per_boss_win_rate` as the canonical win-rate fallback. `make
phase8-eval-report` writes `runs/eval-report.json` and `runs/eval-report.md`
from the most recent evaluator output. Non-object per-task metric payloads are
reported as critical findings instead of being silently treated as valid zeros;
win-rate summaries are computed over valid task rows and include separate
valid/malformed task counts. If every task row is malformed, the report also
emits a critical no-valid-task finding so release evidence cannot pass without
at least one usable fixed-seed metric row.

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

For Phase 8 fleet monitoring, `scripts/render_phase8_dashboard.py` renders a
static HTML/JSON dashboard from `run_coordinator.py` or `run_phase8_smoke.py`
summary JSON. The dashboard summarizes fleet SPS, crash/recovery counts,
policy/checkpoint lag, worker table state, worker-side learner upload counters,
learner intake counters, sampler weights, and evaluator win-rate inputs;
`make phase8-dashboard` writes the default offline smoke dashboard to
`runs/phase8-smoke/`.
Dashboard health is degraded for lost workers, recovering workers, crash churn,
unassigned active workers, stale/missing policy or checkpoint versions, or active
workers reporting zero fleet SPS. Worker learner upload failures/rejections and
learner rejected/queued batches are also reported as dashboard health issues.
Lost-worker health checks use both aggregate counts and per-worker `alive`
state, so stale or partial summaries still surface heartbeat-expired workers.
`scripts/render_profile_report.py` renders a static JSON/Markdown profile report
from the same summaries. It normalizes fleet SPS, per-worker rollout timing,
per-worker alive/status state, lost/recovering workers, crash counts,
unassigned workers, and stale/missing policy or checkpoint versions into
bottleneck findings, with worker upload failures/rejections and learner
rejected/queued batches reported as intake/backpressure findings.
This report defines a CI-friendly Phase 8 profiling format; live Unity CPU/GPU
profiling is still performed on the game machine.
