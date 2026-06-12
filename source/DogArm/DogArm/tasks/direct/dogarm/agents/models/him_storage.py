# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM-extended rollout storage.

Adds a ``next_critic_obs`` buffer so the HIM estimator can access
the successor critic observation during HIO training.
"""

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
    """Rollout storage that also records *next* privileged observations.

    Used by :class:`HIMPPO` to feed ``next_critic_obs`` to
    :meth:`HIMEstimator.update` during the HIO step.
    """

    next_critic_obs: torch.Tensor | None
    """Buffer of shape (T, E, critic_obs_dim) recording the critic
    observation that follows each transition step."""
    critic_obs_dim: int | None
    """Dimension of the critic observation; None if not recording."""

    class Transition(RolloutStorage.Transition):
        """Extended transition that carries ``next_critic_observations``."""

        next_critic_observations: torch.Tensor | None

        def __init__(self) -> None:
            super().__init__()
            self.next_critic_observations = None

        def clear(self) -> None:
            super().clear()
            self.next_critic_observations = None

    class Batch(RolloutStorage.Batch):
        """Extended batch that includes ``next_critic_observations``."""

        next_critic_observations: torch.Tensor | None

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
            next_critic_observations: torch.Tensor | None = None,
        ) -> None:
            super().__init__(
                observations=observations,
                actions=actions,
                values=values,
                advantages=advantages,
                returns=returns,
                old_actions_log_prob=old_actions_log_prob,
                old_distribution_params=old_distribution_params,
                hidden_states=hidden_states,
                masks=masks,
                privileged_actions=privileged_actions,
                dones=dones,
            )
            self.next_critic_observations = next_critic_observations

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        critic_obs_dim: int | None = None,
    ) -> None:
        """Allocate standard storage plus ``next_critic_observations`` buffer.

        Args:
            critic_obs_dim: Total dimension of critic (privileged)
                observations. If provided, allocates a buffer of shape
                (T, E, critic_obs_dim). If ``None``, the storage won't
                record next critic observations (degraded mode).
        """
        super().__init__(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
        )

        self.critic_obs_dim = critic_obs_dim
        if critic_obs_dim is not None:
            self.next_critic_obs = torch.zeros(
                num_transitions_per_env, num_envs, critic_obs_dim, device=device
            )
        else:
            self.next_critic_obs = None

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add transition with next-critic-obs to the storage."""
        super().add_transition(transition)

        if self.next_critic_obs is not None and isinstance(transition, HIMRolloutStorage.Transition):
            nc = transition.next_critic_observations
            if nc is not None:
                self.next_critic_obs[self.step - 1].copy_(nc)

    def clear(self) -> None:
        """Clear storage including next_critic_obs."""
        super().clear()

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[HIMRolloutStorage.Batch, None, None]:
        """Yields mini-batches with ``next_critic_observations``."""
        if self.training_type != "rl":
            raise ValueError("HIM training requires 'rl' training_type.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        # Flatten all buffers (all guaranteed non-None after __init__)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        assert observations is not None
        assert actions is not None
        assert values is not None
        assert returns is not None
        assert old_actions_log_prob is not None
        assert advantages is not None

        # distribution_params is lazily initialised on the first transition.
        # In RL mode, mini_batch_generator is only called after data exists,
        # so it is guaranteed non-None at this point.
        if self.distribution_params is None:
            raise RuntimeError(
                "distribution_params not initialized — add transitions before generating batches"
            )
        old_distribution_params = tuple(
            p.flatten(0, 1) for p in self.distribution_params
        )

        next_critic_obs_f: torch.Tensor | None
        if self.next_critic_obs is not None:
            next_critic_obs_f = self.next_critic_obs.flatten(0, 1)
        else:
            next_critic_obs_f = None

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield HIMRolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    next_critic_observations=(
                        next_critic_obs_f[batch_idx]
                        if next_critic_obs_f is not None
                        else None
                    ),
                )
