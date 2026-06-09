# Troubleshooting & Root-Cause Catalog

> Implements PRD §9. Each item is a *root* problem with its standing mitigation —
> not a symptom log. Link here from code comments where a guard implements a fix.

| # | Problem | Mitigation | Spec |
|---|---|---|---|
| 9.1 | **Partial observability** — missing hitbox/cooldown/ttl breaks Markov property | explicit state fields + cooldown/lock timers + RNN; don't rely on frame stacking | [observation_schema](./observation_schema.md) §5 |
| 9.2 | **Incomplete action semantics** — buttons ≠ tap/hold/release/duration | hybrid action + explicit duration + action mask + macros + in-mod injection | [action_space](./action_space.md) |
| 9.3 | **Reset contamination** — stale events / unready scene/boss poison data | clean lifecycle + reset ack + lifecycle_state + episode_id + event clear + ready checks | [episode_lifecycle](./episode_lifecycle.md) |
| 9.4 | **Reward hacking** — agent farms shaping instead of winning | terminal ≫ shaping; decouple events/scalar; shaping-free eval; judge by win rate | [reward_design](./reward_design.md) §4 |
| 9.5 | **PPO vs async sampling** — stale policy data hurts PPO | policy_version per batch; sync PPO or APPO/IMPALA; drop/down-weight stale | [distributed_training](./distributed_training.md) §4 |
| 9.6 | **SPS not FPS** — high FPS ≠ efficient training | measure SPS; timeScale/fixedDeltaTime; action_repeat; parallel; fast reset | [metrics](./metrics.md) §3 |
| 9.7 | **Generalization / catastrophic forgetting** — new boss erases old | task/boss embedding; balanced sampler; old-task replay; per-boss eval; curriculum; shared encoder + adapter; MoE | [model_architecture](./model_architecture.md), PRD §9.7 |
| 9.8 | **Privileged info too strong** — FSM/hitbox is superhuman | declare scope (game-state agent); ablate privileged/reduced/human-visible; report each | [observation_schema](./observation_schema.md) §7 |
| 9.9 | **Mod fragility** — game/API/FSM/scene changes break hooks | schema_version; mod-version lock; health checks; fallback fields; unit tests; try/catch + overlay | [mod_dev](./mod_dev.md) §6 |
| 9.10 | **Remote comms security** — network service + weight sync | LAN/localhost only; token auth; no public port; checkpoint hash/sign; command whitelist | [protocol](./protocol.md) §8 |

## Operational quick checks

- **Env won't reach RUNNING** → check `error_code` (BossNotFound / SceneLoadFailed),
  verify scene name and boss `HealthManager`; see lifecycle waits.
- **`SchemaMismatch`** → regenerate bindings (`make gen-schema`); bump
  `SCHEMA_VERSION` on both ends.
- **High invalid_action_ratio** → action mask layout drift between
  `hkrl/spaces.py` and mod `ActionMasker`; they MUST share index order.
- **Reward up, win rate flat/down** → reward hacking; tighten terminal/shaping
  ratio; trust the evaluator, not the reward curve.
- **SPS low** → run `make phase8-profile` for fleet heartbeat bottlenecks, then
  profile live `reset_duration`, `Time.timeScale`/`fixedDeltaTime`,
  `action_repeat`, render cost, and Python inference on the game machine.
