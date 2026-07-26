# Live Mod Validation

Use this procedure to compare the same real-game action sequence before and
after a mod change. It uses HKRLEnvMod's protocol and input injection; it does
not emulate an external keyboard.

## Prerequisites

1. Build and install `HKRLEnvMod.dll`.
2. Start Hollow Knight and wait for the mod TCP port to listen.
3. Export the same `HKRL_AUTH_TOKEN` configured for the mod.
4. Use a Python environment with the `hkrl` development package installed.

First verify connectivity without changing the current scene:

```bash
python scripts/check_env.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml
```

## Deterministic action run

The driver resets into the configured Boss scene and prints a JSON snapshot
after every command. The optional number is the exact protocol
`action_repeat`/physics-tick count. This diagnostic driver caps a command at
200 ticks so a 50 Hz step stays below the environment's five-second receive
timeout.

```bash
python scripts/live_action_demo.py \
  --command "right 35" \
  --command "left 35" \
  --command "jump 16" \
  --command "noop 8" \
  --command "noop 8" \
  --command "attack 2" \
  --command "up_attack 2"
```

Run it without `--command` for an interactive prompt. Useful boundary and
gravity probes are:

```text
left 40
left 20
reset
jump 6
noop 6
noop 8
jump 16
noop 8
noop 8
```

For boundary validation, compare the two left-action snapshots: the second
must stop changing `player.x` once the wall is reached. For gravity validation,
`player.y` and `player.vy` must form a rise/fall sequence and finish with
`on_ground=true`. The Boss `position`, `velocity`, `hp`, and FSM hash are
reported alongside the player to catch a globally frozen simulation.

## Complete combat-action exploration

Keep the deterministic driver above as the compact movement regression. For
the complete semantic action matrix, use the separate explorer:

```bash
python scripts/live_action_explorer.py \
  --output runs/live/action-exploration.json \
  --fail-on-failed
```

List or select cases without connecting to the game:

```bash
python scripts/live_action_explorer.py --list
python scripts/live_action_explorer.py \
  --case aerial_down_slash \
  --case scream_up \
  --case cyclone_slash_up
```

Use `--reset-between-cases` when a long charge/focus case should start from a
fresh natural episode. This invokes the normal RESET lifecycle; it still does
not write Boss health, position, velocity, or FSM state.

The catalog covers meaningful equivalence classes rather than contradictory
Cartesian button combinations: movement; short/long/left/right/double jump;
ground/air dash; ground/running/aerial/side/up/down ordinary attacks; Dream
Nail; left/right fireball, up scream, airborne down quake; focus; Great Slash,
Cyclone Slash, and Dash Slash; all four 1/2/4/8-tick duration choices; plus all
11 configured macros.

Schema-v6 read-only action telemetry verifies the applied input bits, actual
Hero attack flags, and Spell/Dream-Nail/Nail-Arts FSM transitions. Soul and
missing health are acquired only through ordinary combat input. The explorer
may request a clean RESET but does not expose pause, timescale, teleport,
health/soul writes, Boss events, Boss FSM transitions, or Boss physics controls.
A case is `skipped` rather than faked when the loaded save lacks an ability or a
natural prerequisite cannot be obtained.

## Full Hall of Gods compatibility sweep

`configs/godhome_bosses.yaml` is the versioned compatibility catalog for all 44
distinct Hall of Gods fights and named variants. Ascended/Radiant arenas are
difficulty modes of those fights and are not duplicated as additional Bosses.
The list follows the [Hall of Gods
roster](https://hollowknight.wiki/w/Hall_of_Gods), while exact `GG_*` scene
tokens are checked against the installed game's `globalgamemanagers` file.
Community automation such as
[HollowKnight.BossAttacks' end-to-end
test](https://github.com/royitaqi/HollowKnight.BossAttacks/blob/master/BossAttacks/Utils/E2eBossFightTest.cs)
provides a useful external example of driving Godhome Boss scenes, but the local
build table and live RESET observations remain authoritative here.

Run the complete restart-safe sweep:

```bash
export HKRL_GLOBALGAMEMANAGERS=/path/to/hollow_knight_Data/globalgamemanagers
export HKRL_MOD_DLL=/path/to/Mods/HKRLEnvMod/HKRLEnvMod.dll
python scripts/live_godhome_sweep.py \
  --mod-dll "$HKRL_MOD_DLL" \
  --globalgamemanagers "$HKRL_GLOBALGAMEMANAGERS" \
  --output runs/live/godhome-all-boss-sweep-v0.8.0.json \
  --report runs/live/godhome-all-boss-sweep-v0.8.0.md \
  --fail-on-failed
```

Every Boss gets an isolated TCP connection and two clean RESET handshakes. The
acceptance checks cover the requested scene hash and task id, `RUNNING`
lifecycle, current-scene Boss entities, left/right Hero movement, jump
input/takeoff/gravity/landing, attack input/state, empty reset events, and a new
`episode_id` after same-scene reload. The evidence envelope is versioned as
`hkrl.godhome_sweep.v2`; resume refuses a different probe version even when the
Mod/game/catalog fingerprints match.

RESET is deliberately an object/control gate and sends no policy action.
After each RESET acknowledgement, the test uses a bounded, action-mask-aware
paced-right/neutral plus jump/dash primitive sequence when the preceding
control probe has not already activated the Boss. The explicit neutral actions
give platform traversal a stable horizontal duty cycle without depending on a
Mod-side input gap. Each entry must expose natural position, velocity, FSM, HP,
or combat-entity-set activity and a positive/full combat-health baseline. The
same checks run again after reload, and the sorted `max_hp` capacity vector must
match. `0/0` transition/cinematic placeholders remain in telemetry but are not
combat-health baselines. The catalog requires two simultaneous combat-ready
Boss rows for Oblobbles. Each row records stable id, type/team, prefab and FSM
hashes, world/relative position, velocity, HP, hurtbox/hitbox state, phase,
threat score, and flags; duplicate ids or missing metadata fail the sweep.
Failure to activate is a hard failure, not a warning.

JSON and Markdown are atomically updated after each fight. Continue an
interrupted run without retesting completed fights:

```bash
python scripts/live_godhome_sweep.py \
  --mod-dll "$HKRL_MOD_DLL" \
  --globalgamemanagers "$HKRL_GLOBALGAMEMANAGERS" \
  --output runs/live/godhome-all-boss-sweep-v0.8.0.json \
  --report runs/live/godhome-all-boss-sweep-v0.8.0.md \
  --resume
```

Useful bounded/diagnostic selections are:

```bash
python scripts/live_godhome_sweep.py --list
python scripts/live_godhome_sweep.py \
  --mod-dll "$HKRL_MOD_DLL" --boss pure_vessel --boss absolute_radiance
python scripts/live_godhome_sweep.py \
  --mod-dll "$HKRL_MOD_DLL" --start-at watcher_knights --max-bosses 5
```

The driver exposes no pause/timescale control and has no Boss health, position,
physics, or FSM mutation path. Boss damage, Hero damage, phase transitions, and
terminal outcomes can only occur through the game's naturally running fight.

### 2026-07-26 all-Boss acceptance record

The installed Proton build completed the catalog with **44/44 verified** and
all 44 final results on their first recorded attempt. Independent evidence
validation confirmed:

- every catalog scene token exists in the installed `globalgamemanagers`;
- all Hero left/right/jump/gravity/landing/attack checks passed with no
  `INVALID_ACTION`;
- both entry and same-scene RESET naturally activated every Boss and exposed
  matching full-health capacity vectors;
- every RESET advanced `episode_id` and returned no stale reward event.

The tested v0.8.0 DLL SHA-256 is
`05950f3bf0ed83c5f48c6ae0f3d80759db089653a22894e33345001c6ad64955`.
Median first-entry RESET was `1.7180 s`; median same-scene RESET was `1.7219 s`.
Absolute Radiance was the longest scene load (`19.7401 s` first entry,
`19.7346 s` reload), within the 60-second gate. Winged Nosk required the
largest natural activation traversal (`288` paced steps first entry, `231` after
reload). Oblobbles exposed two simultaneous `450/450` combat Boss rows with
unique stable ids and the complete v2 meta field set. Troupe Master Grimm
correctly transitioned from a `0/0` entry object to `1000/1000`; Nightmare King
Grimm kept one explicit `0/0` transition placeholder alongside its `1250/1250`
combat entity. Neither case required a Boss event or state write.

Evidence:
[`godhome-all-boss-sweep-v0.8.0.json`](../runs/live/godhome-all-boss-sweep-v0.8.0.json)
and
[`godhome-all-boss-sweep-v0.8.0.md`](../runs/live/godhome-all-boss-sweep-v0.8.0.md).

### 2026-07-26 complete action acceptance record

The real Gruz Mother arena verified all 48 catalog cases:

| Family | Verified cases |
|---|---:|
| movement | 2 |
| jump | 7 |
| dash | 4 |
| ordinary attack variants | 8 |
| Dream Nail | 1 |
| directional spells | 4 |
| focus | 1 |
| nail arts | 6 |
| duration (1/2/4/8 ticks) | 4 |
| bootstrap macros | 11 |
| **Total** | **48** |

Evidence is split into bounded runs under `runs/live/`: `action-exploration-core-final.json`,
`action-exploration-abilities-final.json`,
`action-exploration-nail-arts-final.json`,
`action-exploration-duration-final.json`,
`action-exploration-macros-nonresource-final.json`, and
`action-exploration-macros-resource-final.json`. Two natural Boss collisions
interrupted an aerial slash and a Cyclone charge; the failed attempts remain in
those raw files, while
`action-exploration-aerial-side-right-retry.json` and
`action-exploration-cyclone-down-retry-final.json` record clean ordinary RESET
retries. The union contains 48 unique `verified` cases and no unverified case.
`action-exploration-post-performance-final.json` additionally rechecks six
representative paths after the observation-allocation optimization.

## Visual evidence on XWayland

Find the Hollow Knight window and capture it while the synchronous environment
is between commands:

```bash
xdotool search --name "Hollow Knight" getwindowname %@
xdotool search --name "Hollow Knight"
import -window WINDOW_ID /tmp/hkrl-live-validation.png
```

Keep the action snapshots, screenshot, latest mod log, installed DLL hash, and
the exact command list together when recording a regression.

## 2026-07-26 performance acceptance record

### Response-boundary movement continuity

`scripts/live_walk_smoothness.py` compares one 24-tick right-move STEP against
12 ordinary `action_repeat=2` policy decisions. It divides displacement by
observed `server_tick * fixed_delta_time`, so inference/transport gaps remain
part of the measured game time. Both before/after trials were clean 9→9 HP runs
with no reward event or Boss/simulation mutation:

| Metric | Before | After |
|---|---:|---:|
| Continuous reference speed | 7.615 u/s | 7.608 u/s |
| Ordinary-decision speed | 5.275 u/s | 7.829 u/s |
| Speed retention | 69.27% | 102.91% |
| Ordinary-decision game time | 0.72 s | 0.72 s |
| Fixed timestep | 0.02 s | 0.02 s |

The old response suspension cleared horizontal input, inserting one neutral
physics tick between decisions. The fix uses bounded zero-order hold for
movement/aim and genuine hold buttons, while attack/dash/cast/release buttons
are cleared at the response boundary. The slight result above 100% comes from
the longer segmented trial spending a smaller fraction of time accelerating;
steady response velocity remained 8.3 u/s rather than changing movement
physics. A separate live regression stopped immediately on `noop`, remained at
the left wall (`x=86.272`, `vx=0`) for another 40 ticks, completed a
rise/fall/landing jump, and released attack normally.

Evidence:
[`walk-smoothness-pre-fix.json`](../runs/live/walk-smoothness-pre-fix.json) and
[`walk-smoothness-post-fix.json`](../runs/live/walk-smoothness-post-fix.json).

### Main-thread throughput

The same Gruz Mother scene was measured for 400 requests at `time_scale=1`,
`fixed_delta_time=0.02`, and `action_repeat=2`, with the Steam FPS overlay and
an uncapped render rate. The before run used the per-observation scene scans and
per-frame reflected input invocation; the after run used cached/bounded entity
discovery and pre-bound input delegates.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Active-scene render FPS snapshot | 104 | 341 | +227.9% |
| STEP mean | 131.587 ms | 59.994 ms | -54.4% |
| STEP p50 | 123.567 ms | 58.619 ms | -52.6% |
| STEP p95 | 146.636 ms | 66.332 ms | -54.8% |
| STEP p99 | 400.230 ms | 67.458 ms | -83.1% |
| STEP maximum | 430.694 ms | 68.480 ms | -84.1% |
| Requests/second | 7.599 | 16.668 | +119.3% |
| Effective fixed updates/second | 47.838 | 50.003 | +4.5% |

The optimized run covered exactly 1,200 server ticks in 23.9988 seconds. Its
observed `fixed_delta_time` remained `0.02`, demonstrating that the improvement
did not coarsen collision or gravity integration. A sustained-left regression
stopped at the arena wall with zero X velocity; a long jump reached positive
Y velocity and returned to `on_ground=true` under gravity.

### Post-action-telemetry hit-stop diagnosis

After schema-v6 action telemetry landed, a 240-request run with
`action_repeat=2` measured four 339–349 ms outliers. The enhanced benchmark
recorded the event and HP delta for every slow request: all four outliers were
exactly the four `DAMAGE_TAKEN` steps (`hp_delta=-1`). Hollow Knight's ordinary
damage path runs its native `FreezeMoment` hit-stop; preserving it is part of
the unmodified Boss/player dynamics required for training.

Excluding only those labeled damage frames, the same run measured:

| Metric | Non-damage STEP |
|---|---:|
| mean | 59.953 ms |
| p50 | 60.081 ms |
| p95 | 67.760 ms |
| p99 | 70.111 ms |
| maximum | 72.716 ms |

The server still reported `fixed_delta_time=0.02` and `time_scale=1`. Thus the
50 Hz fixed physics interval is not a render cap or periodic Mod stall. The
remaining long samples are observable combat hit-stop caused by the naturally
running Boss. The Mod does not disable it, freeze the Boss, grant
invulnerability, or edit health to improve benchmark numbers.

The final hot path also reuses entity/Boss/dead-id buffers, reads public
Hero/PlayerData state directly, reuses a per-thread 16 KiB FlatBuffers builder,
and emits one size-prefixed response allocation instead of copying a second
payload. These changes reduce allocation pressure without changing the wire
format or game state.

## 2026-07-26 jump, combination, and arena acceptance

All tests below used the installed schema-v6 Mod in `GG_Gruz_Mother`. They sent
ordinary Hero inputs only. Every evidence bundle declares
`boss_mutation_allowed=false`; no pause, health write, position write, Boss FSM
event, or Boss physics write was used.

### Jump amplitude by hold duration

Three uncontaminated resets/trials were collected for each policy duration at
the native `fixed_delta_time=0.02`. Natural Boss-damage trials would have been
discarded and retried; all 12 attempts were clean.

| Requested hold | Verified input samples | Median height | Three heights | Median airtime |
|---:|---:|---:|---:|---:|
| 1 tick | 1 | 1.2117 u | 1.2065 / 1.2175 / 1.2117 | 0.32 s |
| 2 ticks | 2 | 1.2111 u | 1.2140 / 1.2025 / 1.2111 | 0.32 s |
| 4 ticks | 4 | 2.4946 u | 2.5005 / 2.4654 / 2.4946 | 0.48 s |
| 8 ticks | 8 | 4.6132 u | 4.6132 / 4.5989 / 4.6157 | 0.76 s |

Peak observed upward velocity was `15.702 u/s` in every profile. Height was
monotonic non-decreasing, and 8-tick hold gained `3.4015 u` over 1 tick.
The 1- and 2-tick controls produced distinct committed hold counts but the same
physical height band in this arena/build. Keep the wire layout stable for
checkpoint compatibility; treat the 2-tick branch as a possible redundant
choice and only prune it after multi-arena evaluation.

Evidence:
[`jump-amplitude-duration-3trial-final.json`](../runs/live/jump-amplitude-duration-3trial-final.json).

### Aggregated action input

The dedicated combination family verified 6/6 cases:

- jump + left/right aerial side slash;
- jump + up-slash and jump + down-slash;
- nail charge + simultaneous jump/nail hold + airborne Cyclone release, with
  both up and down continuation inputs.

Success required more than movement: the verifier checked airborne state,
directional attack flags, committed input bits, nail-charge time, Cyclone flag,
and Nail Arts FSM transition. Each case began with an ordinary clean RESET;
the Boss continued its natural movement throughout.

Evidence:
[`action-exploration-combinations-final.json`](../runs/live/action-exploration-combinations-final.json).

### Death auto-reset

The contact policy deliberately walked into the live Boss without attacking.
The Hero naturally lost 9 hp and died after 349 decisions / 20.94 seconds.
The arena preserved terminal episode `20`, including `PlayerDeath` and damage,
then completed the standard RESET handshake and returned fresh episode `21`.
The result recorded one death and one successful auto-reset.

This run validates episode recovery, not combat capability: the deliberately
losing policy has `hitless_win_rate=0` and did not meet the 30-second target.
Hitless fast-clear success must come from trained fixed-seed evaluation.

Evidence:
[`boss-arena-death-autoreset-final.json`](../runs/live/boss-arena-death-autoreset-final.json).
