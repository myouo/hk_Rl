# Action Space

> Implements PRD §6. Wire type: `Action` in [`../schema/hkrl.fbs`](../schema/hkrl.fbs).
> Python construction: `hkrl/spaces.py`.
> Normative allow/deny boundary:
> [`training_capability_policy.md`](./training_capability_policy.md).

## 1. Why hybrid (not MultiBinary)

Hollow Knight actions carry semantics that raw button bits cannot express:
short/long jump, attack direction, pogo (down-slash), dash timing, cast
direction, focus hold, nail-art hold/release, attack/dash/cast lock windows,
buffering/cancel windows, cooldown/resource limits. Pure `MultiBinary` produces
many invalid combos (`left+right`, `dash while cooldown`, `cast without soul`).

## 2. The hybrid space

```text
movement_x : Discrete(3)   # 0=left 1=neutral 2=right
aim_y      : Discrete(3)   # 0=down 1=neutral 2=up
buttons    : MultiBinary(9)
             [jump_tap, jump_hold, dash, attack, cast,
              focus_hold, dream_nail, nail_art_hold, nail_art_release]
duration   : Discrete(4)   # index into {1, 2, 4, 8} ticks
macro      : Discrete(M+1) # 0=none, 1..M = macro action (optional, see §5)
```

`M` defaults to `11` (`hkrl.spaces.DEFAULT_N_MACROS`) and is exposed as
`action.n_macro_actions` in task YAML. It may be reduced per task or disabled
with `action.enable_macro_actions=false`, but it cannot exceed the mod-side
`ActionMasker.DefaultMacroCount` / `MacroActionScheduler` set for the current
mod build.

On the wire these pack into `Action{movement_x, aim_y, buttons(bitmask),
duration_idx, macro_id}`. The model has one head per component
([`model_architecture.md`](./model_architecture.md)).

The PyTorch training path packs sampled actions into an integer tensor with this
fixed order:

```text
[movement_x, aim_y, button[0], ..., button[8], duration, macro?]
```

`macro` is present only when the policy has macro actions enabled. Keep this
order aligned with `hkrl.models.heads.CompositeActionDistribution` and rollout
buffer action storage.

### Button bit layout (mirror in `hkrl/spaces.py` and mod `InputInjector`)

```text
bit 0 jump_tap    bit 1 jump_hold   bit 2 dash
bit 3 attack      bit 4 cast        bit 5 focus_hold
bit 6 dream_nail  bit 7 nail_art_hold  bit 8 nail_art_release
```

## 3. Action mask (PRD §6.3)

The mod computes a mask each tick from current state; the policy sets masked
logits to `-inf` before sampling (per-head). Invalid attempts that slip through
are reported as `InvalidAction` reward events. The mod always records wire-level
invalid actions (out-of-range movement/aim/duration/macro ids or button bits
outside the 9-bit layout) before executing a no-op. Invalid wire actions never
reach input injection.

```text
dash_cooldown > 0 or !CanDash()    -> mask dash
soul < cast_cost                   -> mask cast / focus
spell not learned                  -> mask cast
attack_lock > 0                    -> mask attack / cast / dash
not grounded and no double_jump    -> mask jump
focusing                           -> mask attack / dash / cast
!CanDreamNail()                    -> mask dream_nail
!CanNailCharge()                   -> mask nail-art hold/release
movement_x: left XOR right         (mutually exclusive by construction)
aim_y:      up XOR down             (mutually exclusive by construction)
```

Mask layout on the wire is a flat `action_mask[]` bool array; the canonical
index order (movement, aim, each button, duration, macro) is defined as a
constant in `hkrl/spaces.py` and MUST match the mod's `ActionMasker`.
Every `StepRequest` carries `enable_macro_actions` and `n_macro_actions`; the
mod uses those task-layout fields to size the returned action mask and to report
macro ids outside the current task's range as `InvalidAction`.
StepResponse masks are computed from the same tick's observed player readiness:
`soul`, grounded/double-jump, dash, ordinary attack, spell/focus, dream-nail,
and side-effect-free nail-charge gates. The Mod invokes the same read-only
`HeroController.Can*` predicates used by the game. It deliberately never calls
`CanNailArt()` while observing because that method consumes the current charge
timer. Cooldown/lock timers remain explicit observation fields.
The macro slice uses the same readiness rules as the primitive buttons:
`macro:0` is the no-macro/primitive path, while `macro:1..M` map to mod macro
ids `0..M-1` and are masked when their primitive sequence would require an
unavailable jump, dash, attack, cast, or focus input.

## 4. Duration & action_repeat

`duration` selects how many ticks a button is held (tap vs hold vs nail-art
charge). `action_repeat` is the protocol-level count of FixedUpdate ticks before
returning the StepResponse, unless a terminal event ends it early. They remain
different fields, but the Gym environment aligns them per policy decision:

```text
request.action_repeat =
    max(task.action_repeat, selected duration ticks, selected macro-plan ticks)
```

This prevents the next sampled/recorded PPO action from overtaking a hold or
macro that the Mod is still executing. The task value is a minimum cadence; a
1-tick primitive in a 2-tick task is still repeated for two physics ticks.
`info.action_repeat` reports the requested value and `info.elapsed_ticks`
reports the actual server-tick delta.
The GameWorker converts that delta to `discount_exponent =
elapsed_ticks / task.action_repeat`, so GAE applies time-aware
`gamma^discount_exponent` and `lambda^discount_exponent`. Long holds/macros are
therefore not treated as zero-cost one-step transitions.

## 5. Macro actions (PRD §6.4)

Optional high-level actions to bootstrap early training:

```text
approach, retreat, jump_attack, pogo, dash_away, dash_through,
cast_forward, cast_up, focus_when_safe, short_hop, long_jump
```

Curriculum over abstraction level:

```text
Phase 1: macro-heavy
Phase 2: macro + primitive mixed
Phase 3: mostly primitive with learned duration
```

Macros expand to primitive sequences in the mod's `MacroActionScheduler`, so the
environment contract stays primitive-based.

The policy treats macros hierarchically. `macro=0` selects the primitive path
and its behavior probability includes all primitive branches. For `macro>0`,
the Mod ignores sampled primitive values, so PPO log-probability includes only
the selected macro branch; the stored primitive fields are canonicalized to
neutral values. This avoids noisy recurrent context and assigning gradient to
actions the game did not execute. See
[ADR-0009](./adr/0009-action-aligned-sequence-appo.md).

## 6. Semantic combination discovery

Meaningful combinations remain compositions of the factorized primitives; they
are **not** another action variable. The immutable version-1 catalog in
[`../python/hkrl/action_combinations.py`](../python/hkrl/action_combinations.py)
currently names 18 motifs:

| Family | Catalog entries |
|---|---|
| movement + attack | `moving_slash_left`, `moving_slash_right` |
| jump + nail | `jump_side_slash_left/right`, `jump_up_slash`, `jump_down_slash` |
| jump + dash | `air_dash_left`, `air_dash_right` |
| jump + spell | `jump_fireball_left/right`, `jump_scream_up`, `jump_quake_down` |
| jump + nail art | `jump_great_slash_left/right`, `jump_cyclone_up/down` |
| dash + nail art | `dash_slash_left`, `dash_slash_right` |

Clients discover the stable list through
`HKRLEnv.action_combination_catalog`. When
`action.expose_action_combinations=true`, reset/step `info` includes:

```text
action_combination_catalog_version  # append-only catalog contract
action_combination_bits             # bit N => combo_id N is startable now
action_combination_count            # popcount of the bitset
```

The availability bitset is derived locally from the authoritative per-head
action mask plus current grounded/soul state. It is a conservative start check:
later phases must still satisfy the live mask. The catalog is never copied into
the FlatBuffers hot path and the bitset calculation is `O(K)` for `K=18`, with
no per-step string/list allocation.

Training keeps the small factorized heads and recurrent memory. Entropy
regularization explores primitive compositions; GRU state learns multi-tick
charge/jump/release timing. `CombinationCoverageBandit` supplies UCB1
prioritization only for separately labelled smoke/curriculum collection, where
it tests unseen motifs before repeating covered ones. It must not replace PPO's
own action samples because PPO is on-policy. See
[ADR-0007](./adr/0007-factorized-action-combinations.md).

The live explorer covers six particularly timing-sensitive aggregations:

```bash
python scripts/live_action_explorer.py \
  --family combination \
  --reset-between-cases \
  --fail-on-failed \
  --output runs/live/action-exploration-combinations.json
```

Jump-height response to all duration choices is measured independently with:

```bash
python scripts/live_jump_profile.py \
  --clean-trials 3 \
  --max-attempts 6 \
  --fail-on-invalid \
  --output runs/live/jump-amplitude-duration.json
```

## 7. Input injection (PRD §9.2)

Actions are injected **inside the mod**, not via an external virtual gamepad.
`StepController.FixedTick()` selects the current primitive input on the Unity
main thread. `InputInjector` subscribes to InControl's public
`InputManager.OnUpdate` event and commits that primitive into Hollow Knight's
`PlayerAction` set immediately after physical action sets refresh. Both
`HeroController` and independent PlayMaker input consumers therefore observe
the same value in that input tick.

Movement/aim write `left/right/up/down` and refresh `moveVector`; spell `cast`
maps to the game's `quickCast`, while `focus_hold` maps to the game's `cast`
action. Nail-art hold asserts `attack`, and `nail_art_release` clears it to
produce the release edge. The bridge uses InControl's same-tick internal
`SetValue` + `Commit` when available so pending physical bindings do not OR into
the agent value, then falls back to public `CommitWithState` only for compatible
older builds.

All reflection is resolved lazily because the game input singleton is not ready
at Mod initialization. Hook bodies are exception-contained and log a single
compatibility error. A monotonically increasing successful-commit counter
acknowledges the write back to `StepController`; it does not encode the next
observation until the commit is visible. Repeated ticks belonging to one STEP
advance the active macro plan instead of restarting it; once a plan completes,
the remaining repeats are neutral. A new policy decision may then start the
macro again. The `pogo` plan performs a real takeoff before its down-slash, and
`focus_when_safe` holds long enough for one ordinary heal attempt.
Reset/reconnect clears every active plan.
Between STEP responses, movement/aim axes and explicit hold controls
(`jump_hold`, `focus_hold`, `dream_nail`, and `nail_art_hold`) remain continuous
for at most 10 FixedUpdates. This bridges local inference/transport latency
without injecting a neutral walking tick or a synthetic hold release. Transient
edge controls such as attack, dash, cast, and nail-art release are stripped at
the response boundary, so they cannot repeat merely because the next policy
decision is late. An absent/hung policy cannot leave any bridged input asserted
for more than 200 ms at the default 50 Hz.
Disposing the driver commits neutral input and removes the hook, preventing held
controls or duplicate callbacks after a Mod reload.
The bridge is disabled throughout RESET/scene transition and enabled only once
the environment reaches `RUNNING`, so it cannot overwrite inputs used by the
game's own Godhome entry flow.
