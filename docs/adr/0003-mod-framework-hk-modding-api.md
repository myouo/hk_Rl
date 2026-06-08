# ADR-0003: Mod framework — Hollow Knight Modding API

- Status: **Accepted**
- Date: 2026-06-08

## Context

The environment server runs inside Hollow Knight as a mod. It must hook game
internals (`HeroController`, `HealthManager`, `BossSceneController`,
`PlayMakerFSM`), drive Godhome / Hall of Gods boss fights, inject input, and
manage scene/reset lifecycle. Candidates: HK Modding API; BepInEx; a compatibility
abstraction over both.

## Decision

Standardize on the **Hollow Knight Modding API** (MonoMod-based).

## Rationale

- **Community standard** for Hollow Knight; most boss/Godhome mods use it, so
  patterns, examples, and support are mature.
- Direct, well-trodden access to HK-specific systems (Godhome statue/boss flow,
  `HealthManager`, `BossSceneController`).
- Cleaner integration with HK's mod loader lifecycle than a generic plugin host.

## Consequences

- `HKRLEnvMod` subclasses the Modding API `Mod` type; references the game's
  managed assemblies + the API (paths are machine-local, kept out of source).
- Harmony patches for hooks (damage/heal/death/scene).
- Compilation needs a configured game install — deferred to Phase 0; the C# tree
  is interface-level placeholder until then.

## Alternatives rejected

- **BepInEx** — broad Unity ecosystem, but HK-specific hooks need more manual work
  and there's less HK-focused community material.
- **Dual abstraction layer** — maximally flexible in theory, but high upfront
  complexity for low near-term payoff; revisit only if a second framework is ever
  truly required.
