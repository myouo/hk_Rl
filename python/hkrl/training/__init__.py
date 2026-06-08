"""Training algorithms and buffers (PRD §8, docs/distributed_training.md).

    gae                 generalized advantage estimation
    rollout_buffer      flat on-policy buffer + RolloutBatch
    batch_io            pickle-free RolloutBatch NPZ serialization
    recurrent_buffer    sequence buffer for truncated BPTT (+ burn-in)
    ppo                 synchronous PPO
    recurrent_ppo       PPO over sequences with hidden-state handling
    appo                asynchronous PPO (policy_version staleness handling)

Algorithms register via @register_algo and are selected from TrainConfig.algorithm.
"""
