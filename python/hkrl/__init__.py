"""hkrl — Hollow Knight reinforcement learning package.

Layout (see ../README.md and ../../docs/architecture.md):

    protocol      wire constants + (de)serialization helpers over schema/hkrl.fbs
    transport/    pluggable Transport (tcp, shared_memory)
    env           Gymnasium env (HKRLEnv)
    spaces        hybrid action space, entity-list obs space, action mask layout
    reward        RewardEvent[] -> scalar (config-weighted)
    wrappers      observation tiers, normalization, frame ops
    models/       encoders, entity attention, recurrent ActorCritic, heads
    training/     PPO / RecurrentPPO / APPO, rollout buffers, GAE
    worker/       GameWorker, checkpoint client
    learner/      learner server, checkpoint registry
    coordinator/  worker management, task sampling, curriculum
    eval/         evaluator, scripted policies
    utils/        config, logging, metrics, seeding, registry

Everything is interface-level placeholder at this stage; implementations land
per the roadmap (see ../../AGENTS.md#roadmap).
"""

from __future__ import annotations

from hkrl.protocol import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0"
