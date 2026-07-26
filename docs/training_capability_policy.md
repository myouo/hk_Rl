# Training Capability Policy

This policy is the hard boundary between ordinary Hollow Knight controls and
privileged environment administration. Training code may observe privileged
state, but the learned policy may only act through controls available to a
normal player.

## 1. Policy-callable abilities

The policy may select only:

| Group | Allowed values |
|---|---|
| horizontal movement | left, neutral, right |
| vertical aim | down, neutral, up |
| jump | tap or hold |
| dash | normal in-game dash |
| nail | attack, directional attack, nail-art hold/release |
| spell | normal quick-cast, subject to soul and game locks |
| focus | normal focus/heal hold, subject to soul and game locks |
| dream nail | normal dream-nail input |
| duration | hold a permitted input for 1, 2, 4, or 8 ticks |
| macros | configured macros that expand only to the primitives above |
| combinations | simultaneous/temporal compositions of the primitives above; catalog metadata is not a new capability |

`TrainingCapabilityPolicy` rejects out-of-range axes, unknown button bits,
invalid durations, and unavailable macro ids. Rejected requests emit
`InvalidAction` and execute a no-op.

## 2. Environment-control abilities

These are available to the local environment manager, never to the learned
policy head:

- `RESET` / `SET_TASK`: clean episode setup and Godhome scene selection;
- `PAUSE` / `RESUME`: operator and recovery control;
- `SET_TIMESCALE`: SPS tuning with unchanged simulation fixed-step semantics;
- `PING`: liveness only.

Arena auto-reset belongs to the local supervisor/GameWorker and invokes only
the same `RESET` command after preserving the terminal transition. It is not
selectable by the learned policy.

RESET may load a configured save and use Hollow Knight's own
`GameManager.BeginSceneTransition` plus `BossSceneController.SetupEvent`.
It mirrors the Hall of Gods workshop's `Change Scene` state and uses the
`GodsAndGlory` visualization, without globally broadcasting its local
transition-cinematic events. If the save loads at a bench, RESET uses
Hollow Knight's own `Bench Control` FSM by sending its canonical `GET UP`
event. The FSM clears the bench marker and restores body/control state. This is
lifecycle-only and does not directly write Transform/Rigidbody state or expose
a policy capability. Input injection is disabled during the transition. The
Mod waits for a new scene instance, a stable completed Godhome transition, a
controllable Hero with restored gravity/collision, and live bosses before
reporting `RUNNING`.
For direct/same-scene loads it also marks GameManager's scene-entry completion
handshake pending, matching the lifecycle step omitted by
`BeginSceneTransition`; `HeroController.FinishedEnteringScene()` completes it.
This flag is not Boss state. RESET waits for the unmodified Boss to demonstrate
natural motion or control-state activity and never forces a Boss wake event.

## 3. Allowed observation access

Read-only access may include player/Boss/entity position, velocity, hp, soul,
cooldowns, FSM hashes, collision geometry, projectiles, hazards, lifecycle, and
reward events. This includes player action-state/FSM telemetry used to verify
directional attacks, spells, dream nail, and nail arts; it never supplies a
write path. Ablation tiers decide which subset reaches a model.

## 4. Forbidden capabilities

Neither policy actions nor macros may:

- write `Transform`, `Rigidbody2D`, velocity, gravity, collision, or animation
  state;
- teleport, fly, freeze the Hero, noclip, or directly move a Boss;
- write player/Boss hp, soul, damage, invulnerability, cooldowns, charms, nail
  damage, inventory, or save progression;
- force FSM states, stagger, phase transitions, Boss death, or reward events;
- load scenes, reset, pause, or change timescale through the policy action
  space;
- call remote services in the local `observation → action → game` loop.

The only action-side game mutation is the scoped InControl `PlayerAction`
bridge: policy primitives while lifecycle state is `RUNNING`, plus the fixed
reset-owned bench-exit input above. Observation reflection is read-only.

## 5. Review rule

Any new policy ability requires all of:

1. it is reproducible through ordinary player controls;
2. schema/action-mask/docs updates remain byte-aligned across Python and C#;
3. a test proves it does not write physics, health, FSM, or progression state;
4. an ADR if it changes the boundary above.
