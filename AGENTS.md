# AGENTS.md — Project Charter for Coding Agents

> This file orients any coding agent (Codex, Claude, etc.) working in this repo.
> Read it fully before editing. Authoritative product vision:
> [`hollow_knight_rl_prd.md`](./hollow_knight_rl_prd.md). Engineering specs:
> [`docs/`](./docs/). When in doubt, the PRD + ADRs win; if code disagrees with
> them, that's a bug to reconcile (and record in a new ADR).

---

## 1. What this project is

A **game-state reinforcement-learning agent** for Hollow Knight. A C# mod
(`HKRLEnvMod`) exposes structured game state and accepts actions, acting as an
*environment server*. A Python package (`hkrl`) runs a Gymnasium env, models, and
training. The agent learns to defeat bosses in Godhome / Hall of Gods, scaling
from a single boss → simultaneous multiple enemies → linear multi-boss flows.

It reads internal game state (positions, hp, FSM, hitboxes, cooldowns) — it is
**not** a vision/human agent. That scope is deliberate (PRD §9.8); ablation tiers
(privileged/reduced/human-visible) keep evaluation honest.

**Guiding principle (PRD §15): get the environment right first, then scale the
model.** Order: clean env → single boss stable → entity list → attention → local
worker → remote learner → curriculum.

## 2. Architecture in one diagram

```text
Hollow Knight + HKRLEnvMod (C#)          Game PC                  Remote GPU
  Observation collect                    GameWorker  ─rollouts─▶  Learner (PPO/APPO)
  Action apply (FixedUpdate)   ◀──FB────  local infer            Checkpoint registry
  Reward events                  TCP/SHM  Gym env    ◀─weights──  Coordinator/Evaluator
  Clean episode lifecycle
```

Components & responsibilities: [`docs/architecture.md`](./docs/architecture.md) §3.

## 3. Six invariants — do not violate

1. **Local action loop never crosses the remote network.** `obs → local_policy →
   action → game` is on the Game PC; remote GPU only does batch training.
   ([ADR-0004](./docs/adr/0004-local-inference-remote-training.md))
2. **`schema/hkrl.fbs` is the single source of truth.** Change the schema, then
   `make gen-schema`. Never hand-edit generated bindings.
   ([ADR-0002](./docs/adr/0002-serialization-flatbuffers.md), [`schema/README.md`](./schema/README.md))
3. **Transport is pluggable.** Everything goes through `Transport`; TCP and
   shared-memory are interchangeable.
4. **Config-driven + registry.** Register components (`@register_model`, etc.) and
   select them from YAML. No core edits to add a boss/model/transport/reward.
   ([`hkrl/utils/registry.py`](./python/hkrl/utils/registry.py))
5. **Clean episode lifecycle.** State machine + reset ack + `episode_id` + event
   clear. No `STEP` before `RUNNING`; reset failures return an error code.
   ([`docs/episode_lifecycle.md`](./docs/episode_lifecycle.md))
6. **Model is decomposed + mask-aware.** `encoders → attention → memory → heads`,
   `entity_mask` everywhere. ([`docs/model_architecture.md`](./docs/model_architecture.md))

## 4. Directory map

```text
schema/            ★ hkrl.fbs single source of truth + gen_schema
mod/HKRLEnvMod/    C# env-server mod (HK Modding API)
  Transport/ Env/ Observation/ Action/ Rewards/ Debug/ Schema(gen)/
python/hkrl/       Python package
  transport/ schema(gen)/ models/ training/ worker/ learner/ coordinator/ eval/ utils/
  env.py spaces.py reward.py wrappers.py protocol.py
python/tests/      pytest (skeleton: import + layout consistency; xfail behavioral)
configs/           tasks/*.yaml, train/*.yaml, base.yaml
docs/              specs + adr/
scripts/           gen_schema.sh, train.py, run_worker.py, run_learner.py, run_eval.py
```

Where to make a change: [`docs/architecture.md`](./docs/architecture.md) §6 (the
"touch only…" table).

## 5. Roadmap

Source of truth for phases: PRD §10. Status tracker below — **update the marker
when a phase lands.**

```text
Phase 0  Modding env + Hello-World mod, read player pos/scene
Phase 1  HKRLMod v0: TCP + step/reset protocol + player+boss obs
Phase 2  Python Gymnasium env + random/scripted policy + logging
Phase 3  Single-boss PPO (MLP) baseline + per-boss eval
Phase 4  Entity-ized observation (entities/projectiles/hazards, stable ids)
Phase 5  Attention + recurrent policy (entity encoder + GRU/LSTM)
Phase 6  Local inference + remote training (worker/learner/checkpoints)
Phase 7  Multi-boss curriculum + anti-forgetting (per-boss eval, replay)
Phase 8  Multi-instance scale-out + monitoring + crash recovery     ◀── CURRENT
```

**Implementation priority (PRD §14):**
- **P0 (do first):** bidirectional step/reset protocol, Gym env, clean lifecycle,
  action mask, reward events, single-boss baseline.
- **P1:** entity list, local-infer/remote-train, recurrent policy, per-boss eval,
  metrics dashboard.
- **P2:** attention encoder, macro/primitive mixed action, curriculum, multi-worker,
  APPO/IMPALA, linear boss sequence.
- **P3 (experiments):** imitation/BC, replay recorder, MoE, reduced-vs-privileged,
  vision+state hybrid.

Placeholders are tagged `# TODO(phase-N):` / `throw new NotImplementedException()`.
Grep `TODO(phase-` to find the next concrete work for a phase.

## 6. Schema-first workflow (mandatory)

Any change to observation/action/protocol:

1. Edit [`schema/hkrl.fbs`](./schema/hkrl.fbs) (append-only — see evolution rules).
2. Bump `SCHEMA_VERSION` in both [`python/hkrl/protocol.py`](./python/hkrl/protocol.py)
   and [`mod/HKRLEnvMod/Transport/Protocol.cs`](./mod/HKRLEnvMod/Transport/Protocol.cs).
3. `make gen-schema` (regenerates C# + Python bindings).
4. Update the field-semantics docs ([`observation_schema.md`](./docs/observation_schema.md)
   / [`action_space.md`](./docs/action_space.md)) and `CHANGELOG.md`.

Two layouts MUST stay byte-identical across languages: the **button bit layout**
and the **action-mask index order** (`hkrl/spaces.py` ↔ mod `InputInjector` /
`ActionMasker`). Drift = high `invalid_action_ratio`.

## 7. Conventions

**Python**
- 3.10+, full type annotations, `from __future__ import annotations`.
- Lint/type/test: `make lint` (ruff), `make typecheck` (mypy), `make test` (pytest).
- Register new components via the registry; select via config. Don't hardcode.
- Import heavy deps (torch/gymnasium) at module scope only where the module truly
  needs them, so `import hkrl` stays light.
- New behavior ships with a test. Skeleton tests assert imports + layout
  consistency; behavioral tests are `xfail(strict=True)` until their phase, then
  flip to real assertions.

**C#**
- **Network thread never touches Unity objects.** All game access on the main
  thread in `FixedUpdate` via `StepController.FixedTick()`. Cross the boundary with
  `ConcurrentQueue` / ring buffers. ([`docs/mod_dev.md`](./docs/mod_dev.md) §5)
- Wrap every game hook in try/catch and log via `Debug/Logger` (PRD §9.9).
- Use the generated `HKRL.*` types; never redefine schema enums by hand.
- Keep machine-specific assembly paths out of source (`.csproj.user` / props).

**Docs/decisions**
- Significant architectural choices get an ADR in [`docs/adr/`](./docs/adr/).
- Keep specs and code in sync; cross-link with `file:line` where useful.

## 8. Performance discipline (the model must train *well*)

- Optimize **SPS, not FPS** ([`docs/metrics.md`](./docs/metrics.md) §3). Profile
  `reset_duration` first; then `Time.timeScale`/`fixedDeltaTime`, `action_repeat`,
  parallel instances, reduced render quality.
- Hot path is **zero-copy FlatBuffers**; never put debug JSON (`info`) on it.
- Training path reserves **`torch.compile` + AMP** and contiguous **sequence
  (truncated-BPTT) batching** for recurrent models.
- `entity_mask` must gate attention/pooling — padded slots leak nothing.
- Prefer shared-memory transport for single-machine high-SPS once TCP is stable.

## 9. Evaluation > reward (read this twice)

Training reward is **not** capability. Judge progress by shaping-free metrics, in
order: per-boss win rate → damage taken → time-to-kill → invalid-action ratio →
generalization → old-task regression. The evaluator computes these on fixed seeds,
isolated from training. A rising reward with flat/falling win rate means reward
hacking — trust the evaluator. ([`docs/reward_design.md`](./docs/reward_design.md) §4,
[`docs/metrics.md`](./docs/metrics.md) §2)

## 10. Do NOT

- Hard-code the final scalar reward in the mod (report **events**; compose scalar
  in `hkrl/reward.py`).
- Allow reset contamination (stale events / un-ready scene/boss / `STEP` before
  `RUNNING`).
- Break `schema_version` compatibility (non-append-only schema edits).
- Run real-time inference across the remote network.
- Edit generated bindings under `*/Schema/` or `hkrl/schema/hkrl/`.
- Let the action-mask layout drift between Python and C#.
- Expose a public network port (LAN/localhost + token auth + hash-verified
  checkpoints only — PRD §9.10).

## 11. Verify your changes

```bash
make gen-schema     # if you touched schema/hkrl.fbs (needs flatc)
make lint           # ruff
make typecheck      # mypy
make test           # pytest (collection must succeed; xfail = not-yet-implemented)
# behavioral / integration (needs a live game + mod):
python scripts/train.py --config configs/train/ppo_mlp.yaml --task configs/tasks/gruz_mother.yaml --smoke
```

Full dev install: `pip install -e "python[dev]"`. Heavy deps (torch/gymnasium) are
required for the model/env modules and their tests; the package core
(`import hkrl`, protocol/spaces) imports without them.
