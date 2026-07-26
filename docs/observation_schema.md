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
| hashes (scene/fsm/prefab) | field-specific 4,096-bucket embedding lookup, NOT fed as raw int |

`ARENA_SCALE`, `VEL_SCALE`, `T_MAX` are constants in `hkrl/spaces.py`; document
any change here.
The model zeroes hash columns before every continuous MLP and obtains their
signal only from the learned bucket embeddings. This ordering is required for
FP16: live signed hashes have int32-scale magnitudes that exceed the finite
range of IEEE half precision.
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
of crashing the main loop. Boolean readiness also supports zero-argument game
methods such as `CanAttack()` / `CanCast()`. The privileged tail of
`PlayerState` adds read-only action telemetry:

| index | field | semantics |
|---:|---|---|
| 25 | `actor_state_hash` | stable hash of the Hero actor state |
| 26 | `action_flags` | attack/up/down/nail-charge/cyclone/quake/double-jump bits |
| 27 | `spell_fsm_state_hash` | `Spell Control` active state |
| 28 | `dream_nail_fsm_state_hash` | `Dream Nail` active state |
| 29 | `nail_arts_fsm_state_hash` | `Nail Arts` active state |
| 30 | `nail_charge_timer` | ordinary attack-button hold time in seconds |
| 31 | `applied_input_buttons` | canonical button bits committed by the input bridge |

`action_flags` uses bit 0 `attacking`, bit 1 `up_attacking`, bit 2
`down_attacking`, bit 3 `nail_charging`, bit 4 `nail_art_cyclone`, bit 5
`spell_quake`, and bit 6 `double_jumping`. These fields are observations only:
the policy cannot write FSMs, flags, or this diagnostic echo.
`applied_input_buttons` makes duration/hold continuity observable without
inspecting or mutating gameplay state. Player FSM references are resolved once
per persistent Hero instance, not scanned every tick.

Boss collection reads
`BossSceneController`'s configured boss list first and uses name heuristics only
as a fallback. For HealthManager variants without a max-hp member, the stable
entity registry caches the highest observed hp for that object so normalization
and observation-delta rewards remain usable.
Reset readiness separately requires HeroController's `acceptingInput` gate,
non-relinquished gameplay control, and a dynamic/simulated Rigidbody2D with
unfrozen position constraints, positive gravity, an enabled collider, and
active terrain-ingress checks. The first training action is therefore not
sampled during a scene/Boss intro lock or after a corrupt same-scene transition.

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

## 8. Current coverage and known limits

The wire shape is broad, but not every field is equally complete yet:

| Area | Current status |
|---|---|
| player position/velocity/hp/soul/facing/movement flags | live game values |
| player attack/cast/focus/dash/dream/nail-art readiness | live, side-effect-free `Can*` methods; conservative fallback |
| player action variant / spell / dream-nail / nail-art state | live read-only flags and cached FSM-state hashes |
| Boss identity/position/velocity/current hp/FSM hash | live values; verified for Gruz Mother |
| Boss max hp | live member when present, otherwise highest hp observed in the episode |
| Boss phase | currently `0`; not yet mapped per Boss |
| Boss hitbox/hurtbox | collider/DamageHero approximation, not every named attack box |
| regular enemies | no dedicated general enemy observer yet |
| projectiles | heuristic discovery; ttl is a placeholder and damage is approximate |
| hazards | heuristic discovery; damage is approximate |
| platforms/pickups/effects | schema tags exist, collectors are not implemented |

The current data is sufficient for a single-Boss baseline and basic entity
avoidance, but it is not a complete mechanical state for every Boss. Fields
marked approximate or placeholder must be excluded from a claimed privileged
benchmark until Boss-specific adapters and data-quality tests land.
