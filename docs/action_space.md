# Action Space

> Implements PRD §6. Wire type: `Action` in [`../schema/hkrl.fbs`](../schema/hkrl.fbs).
> Python construction: `hkrl/spaces.py`.

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
`action.n_macro_actions` in task YAML. It must match the mod-side
`ActionMasker.DefaultMacroCount` / `MacroActionScheduler` set for the current
mod build; otherwise the flat action-mask length will drift.

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
outside the 9-bit layout) before clamping/ignoring them for safe input
injection.

```text
dash_cooldown > 0                  -> mask dash
soul < cast_cost                   -> mask cast / focus
attack_lock > 0                    -> mask attack / cast / dash
not grounded and no double_jump    -> mask jump
focusing                           -> mask attack / dash / cast
movement_x: left XOR right         (mutually exclusive by construction)
aim_y:      up XOR down             (mutually exclusive by construction)
```

Mask layout on the wire is a flat `action_mask[]` bool array; the canonical
index order (movement, aim, each button, duration, macro) is defined as a
constant in `hkrl/spaces.py` and MUST match the mod's `ActionMasker`.
StepResponse masks are computed from the same tick's observed player readiness
where available (`soul`, grounded/double-jump, and can-attack/cast/focus flags),
with cooldown/lock timers defaulting open until the mod reader exposes them.
The macro slice uses the same readiness rules as the primitive buttons:
`macro:0` is the no-macro/primitive path, while `macro:1..M` map to mod macro
ids `0..M-1` and are masked when their primitive sequence would require an
unavailable jump, dash, attack, cast, or focus input.

## 4. Duration & action_repeat

`duration` selects how many ticks a button is held (tap vs hold vs nail-art
charge). Distinct from `action_repeat` (protocol-level, repeats the *same*
StepRequest N FixedUpdate ticks before returning the StepResponse, unless a
terminal reward event ends the episode early). Both exist; don't conflate them.

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

## 6. Input injection (PRD §9.2)

Actions are injected **inside the mod** (main-thread `FixedUpdate`), not via an
external virtual gamepad, eliminating `vgamepad` timing nondeterminism.
