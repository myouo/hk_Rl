# Observation Schema

> Implements PRD §5.5. Wire types: [`../schema/hkrl.fbs`](../schema/hkrl.fbs).
> This doc specifies **semantics, units, ranges, and normalization** — the
> schema only specifies layout.

## 1. Structure

```text
Observation
  ├─ GlobalState   (1)   scene/task/episode context
  ├─ PlayerState   (1)   hero state with explicit cooldown/lock timers
  ├─ entities[]    (N)   variable-count entity list
  └─ entity_mask[] (N)   parallel; true = valid slot
```

## 2. Normalization (Python-side, in `hkrl/spaces.py` / wrappers)

The mod emits raw game units; the policy consumes normalized features. Keep the
normalization in one place so privileged/reduced/human-visible ablations
(PRD §9.8) stay consistent.

| Field group | Transform |
|---|---|
| positions | player-centric: `rel = entity.pos - player.pos`, then `/ ARENA_SCALE` |
| velocities | `/ VEL_SCALE` |
| hp / soul | `/ max_*` → [0,1] |
| timers (cooldown/lock/ttl/invuln) | `clamp(t / T_MAX, 0, 1)` |
| booleans/flags | 0/1; flags bit-unpacked to a vector |
| hashes (scene/fsm/prefab) | embedding lookup, NOT fed as raw int |

`ARENA_SCALE`, `VEL_SCALE`, `T_MAX` are constants in `hkrl/spaces.py`; document
any change here.
`GlobalState.time_in_episode` is measured from the episode's first `RUNNING`
tick, not from Unity scene load time, so same-scene resets start at zero.

## 3. Entity list

Capacity (first version, PRD §3.1):

```text
max_bosses = 4, max_enemies = 8, max_projectiles = 32, max_hazards = 16
# or unified: max_entities = 64, disambiguated by entity_type
```

Required mechanics:

- **`entity_mask`** — model attends only over valid slots (mask in attention +
  pooling). Padded slots are zeroed and masked.
- **`type_embedding`** — `EntityType` → learned vector (boss/enemy/projectile/
  hazard/platform/...).
- **`team`** — enemy / neutral / player-created projectile.
- **`stable_entity_id`** — the mod's `EntityRegistry` keeps identity consistent
  across frames so velocity/history are coherent.
- **`threat_score` + top-k** — when entities exceed capacity, keep all bosses,
  then highest-threat / nearest / fastest projectiles & hazards; aggregate the
  remainder into a single summary token (PRD §7.3).

## 4. `flags` bitfield (EntityState.flags)

Bit layout (extend append-only; mirror in `hkrl/spaces.py`):

```text
bit 0: is_attacking
bit 1: is_invulnerable
bit 2: is_staggered
bit 3: is_airborne
bit 4: spawns_projectiles
bit 5: is_summon
... (reserved)
```

## 5. Markov completeness (PRD §9.1)

Partial observability is mitigated by **explicit state**, not only frame stacking:
cooldowns, lock timers, hitbox-active flags, projectile `ttl`, invuln windows are
all in the schema. Remaining temporal structure (boss wind-up, trajectory history)
is handled by the recurrent memory ([`model_architecture.md`](./model_architecture.md)).
The mod reads player hp/soul plus readiness/timer fields from `PlayerData` and
`HeroController` via reflection with safe fallbacks, so minor Hollow
Knight/Modding API field-name drift degrades to conservative defaults instead
of crashing the main loop.

## 6. Health checks (PRD §9.9)

The worker validates each observation: mask length == entities length, hp ≤
max_hp, finite floats, at least one boss present in a boss task. Failures surface
as `info` warnings and metrics, not silent corruption.

## 7. Ablation tiers (PRD §9.8)

Wrappers expose three observation tiers for honest evaluation:

- **privileged** — full schema (fsm/hitbox/cooldown).
- **reduced** — drop internal fsm/hitbox; keep positions/hp/timers.
- **human-visible** — only what a human could perceive on screen.

Report per-tier separately; this project is explicitly a *game-state agent*.
