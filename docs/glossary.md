# Glossary

| Term | Meaning |
|---|---|
| **HKRLEnvMod** | The C# Hollow Knight mod acting as the environment server. |
| **GameWorker** | Game-PC process: local inference + Gym env + rollout upload. |
| **Learner** | Remote-GPU process: large-batch PPO/APPO training. |
| **Coordinator** | Manages workers, task assignment, curriculum, registries. |
| **Evaluator** | Fixed-seed, shaping-free, per-boss evaluation. |
| **tick / tick_id** | One fixed-update step; `tick_id` matches request↔response. |
| **server_tick** | Mod-side `FixedUpdate` counter. |
| **SPS** | Samples per second — the real training-throughput metric (not FPS). |
| **action_repeat** | Protocol-level repeat of one StepRequest N ticks to raise SPS. |
| **duration** | Action-level hold length (1/2/4/8 ticks); distinct from action_repeat. |
| **entity** | Any in-arena object: boss/enemy/projectile/hazard/platform/pickup/effect. |
| **entity_mask** | Boolean array marking valid slots in the variable-length entity list. |
| **stable_entity_id** | Identity kept consistent across frames by the mod's EntityRegistry. |
| **threat_score** | Per-entity priority used for top-k filtering when over capacity. |
| **summary token** | Single aggregated embedding for entities dropped past top-k. |
| **action mask** | Per-component validity mask; masked logits set to -inf. |
| **macro action** | High-level action expanded to primitives by MacroActionScheduler. |
| **lifecycle_state** | Position in the clean episode state machine. |
| **episode_id** | Unique id per episode; scopes reward events. |
| **policy_version** | Version tag on rollouts/weights for staleness handling. |
| **RolloutBatch** | The on-the-wire training sample bundle (obs/actions/values/...). |
| **truncated BPTT** | Backprop-through-time over fixed-length sequence chunks. |
| **burn-in** | Leading timesteps used only to warm the RNN hidden state, no loss. |
| **privileged / reduced / human-visible** | Observation ablation tiers (PRD §9.8). |
| **schema_version** | Wire-format version carried per message; gates compatibility. |
| **registry** | Name→class lookup enabling config-driven component selection. |
| **ADR** | Architecture Decision Record (see `docs/adr/`). |
