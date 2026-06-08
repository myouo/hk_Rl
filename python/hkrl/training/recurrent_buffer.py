"""Sequence buffer for recurrent PPO (docs/model_architecture.md §4).

Chunks trajectories into fixed-length sequences for truncated BPTT, stores the
hidden state at each sequence boundary, masks padded timesteps, and supports an
optional burn-in prefix used only to warm the RNN (no loss on burn-in steps).
"""

from __future__ import annotations

from typing import Any


class RecurrentRolloutBuffer:
    """Stores transitions + per-step rnn_state and yields (seq_len, batch) chunks.

    Critical correctness points (docs/troubleshooting.md): hidden state reset at
    episode boundaries, padded-timestep masking in the loss, and burn-in steps
    excluded from the policy/value loss.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        sequence_length: int = 32,
        burn_in: int = 0,
        obs_spec: dict[str, Any] | None = None,
    ) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        # TODO(phase-5): preallocate arrays incl. rnn_states + seq_mask.

    def add(self, **transition: Any) -> None:
        raise NotImplementedError  # TODO(phase-5)

    def is_full(self) -> bool:
        raise NotImplementedError  # TODO(phase-5)

    def iter_sequences(self) -> object:
        """Yield (obs, actions, ..., rnn_state0, loss_mask) sequence minibatches.

        ``loss_mask`` is False on padding and burn-in steps.
        TODO(phase-5): implement chunking with episode-boundary awareness.
        """
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError  # TODO(phase-5)
