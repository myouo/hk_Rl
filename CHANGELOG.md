# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
the project version tracks the **schema_version** + roadmap phase.

## [Unreleased]

### Added
- Initial architecture scaffold: full directory tree, interface-level
  placeholders, specification docs, ADRs, and `AGENTS.md`.
- FlatBuffers schema `schema/hkrl.fbs` as the single source of truth for
  observation / action / protocol (schema_version 1).
- Python package `hkrl` skeleton: transport, env, spaces, reward, models
  (entity-attention + recurrent), training (PPO / RecurrentPPO / APPO),
  worker / learner / coordinator / eval / utils.
- C# mod skeleton `HKRLEnvMod` (HK Modding API) covering Transport / Env /
  Observation / Action / Rewards / Debug.
- Phase 0 mod bootstrap: Modding API-backed logger, persistent FixedUpdate
  driver, and periodic scene/player-position snapshot logging.
- Conda development environment (`environment.yml`) for Python runtime, dev,
  logging, and distributed extras.
- Python training foundations: config `defaults` composition, default reward
  composition, deterministic seeding, running metrics, and GAE.
- Configs (`tasks/`, `train/`) and scripts (`gen_schema`, `train`,
  `run_worker`, `run_learner`, `run_eval`).

### Decisions
- ADR-0001 RL framework: self-built PyTorch PPO.
- ADR-0002 serialization: FlatBuffers single source of truth.
- ADR-0003 mod framework: HK Modding API.
- ADR-0004 local inference + remote training decoupled.
