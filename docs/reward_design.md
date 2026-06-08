# Reward Design

> Implements PRD §5.6 + §9.4. Events: `RewardEvent` in
> [`../schema/hkrl.fbs`](../schema/hkrl.fbs). Scalar: `hkrl/reward.py`.

## 1. Event/reward decoupling

The mod **never** computes a final scalar reward. It reports typed **events**;
Python composes the scalar. This keeps reward shaping out of the mod, lets us run
multiple reward functions over the same trajectory, and makes shaping-free
evaluation trivial.
Core events may come from direct game hooks or from conservative observation
deltas (player hp/soul and entity hp changes). Both paths write the same
`RewardEvent` records; scalar shaping stays Python-side.

```text
mod  ──RewardEvent[]──▶  hkrl/reward.py  ──scalar──▶  rollout buffer
                          (config-weighted)
```

## 2. Events

| Kind | Payload (schema fields) |
|---|---|
| `DamageDealt` | `entity_id` (target), `amount`, `aux_int` (damage_type) |
| `DamageTaken` | `entity_id` (source), `amount` |
| `Heal` | `amount` |
| `SoulGained` | `amount` |
| `BossKilled` | `entity_id` |
| `PlayerDeath` | `aux_int` (reason) |
| `SceneChanged` | `aux_int` (from), `aux_int2` (to) |
| `InvalidAction` | `aux_int` (action_id), `aux_int2` (reason) |
| `Stagger` | `entity_id` |

For `InvalidAction`, `aux_int` identifies the invalid component
(`0=movement_x`, `1=aim_y`, `2=buttons`, `3=duration`, `4=macro`) and
`aux_int2=1` means out-of-range while `aux_int2=2` means button bits outside the
canonical 9-bit layout.

## 3. Default reward function (PRD §5.6)

```text
reward =
  + 1.0   * damage_dealt
  - 8.0   * damage_taken
  + 0.5   * soul_gained
  + 2.0   * heal_amount
  + 100   * boss_kill
  - 100   * player_death
  - 0.001 * time_step
  - 0.01  * invalid_action
  + optional shaping (distance, positioning)
```

Weights live in task configs (`configs/tasks/*.yaml`), not in code.

## 4. Anti-reward-hacking (PRD §9.4)

1. **Terminal >> shaping.** `boss_kill` / `player_death` dominate intermediate
   shaping so the agent can't farm mid-rewards instead of winning.
2. **Decoupled events.** Shaping changes never require touching the mod.
3. **Shaping-free evaluation.** The evaluator always reports metrics that ignore
   shaping (win rate, damage ratio, time-to-kill). See [`metrics.md`](./metrics.md).
4. **Core metric ≠ reward.** Decisions are driven by per-boss win rate, not the
   training reward curve.

## 5. Lifecycle correctness

Reward events are buffered in the mod and **cleared on reset**
([`episode_lifecycle.md`](./episode_lifecycle.md)). After `done`, no new events
are collected. Each event belongs to exactly one `episode_id`. This prevents the
classic reset-contamination bug (PRD §9.3).

Mod `StepController` only emits reward events while the lifecycle is `RUNNING`.
`BossKilled`, `PlayerDeath`, and unexpected `SceneChanged` events route the
lifecycle to `TERMINATING` and are included in the terminal `StepResponse`.
