# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM-augmented Actor model for RSL-RL.

This module provides :class:`HIMActorModel`, an extension of the RSL-RL
:class:`~rsl_rl.models.MLPModel` that injects HIM (Hybrid Internal Model)
features into the actor's latent representation before the MLP head.

Integration::

    raw obs (TensorDict) ──► get_latent()
        │                        │
        │   flat_obs (B, obs_dim * history)  ──► HIM Estimator
        │                        │                    │
        │                        │              vel_hat (B,3) + latent_z (B,16)
        │                        │                    │
        │                        ▼                    ▼
        │              obs_normalizer(flat_obs) ─── concat ──► MLP head
        │
        ▼
    critic obs ──► standard MLPModel (unchanged)

The critic model remains a standard :class:`~rsl_rl.models.MLPModel`
(no HIM injection), as it already sees privileged information.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState

from .him_estimator import HIMEstimator


class HIMActorModel(MLPModel):
    """Actor model that prepends HIM features to the observation latent.

    This model wraps a standard MLP actor and injects:
    - ``vel_hat`` (3): estimated body-frame linear velocity
    - ``latent_z`` (latent_dim): implicit terrain/environment embedding

    The HIM estimator is stored as a sub-module and updated via
    :meth:`HIMEstimator.update` by an external HIO-aware training loop.

    Parameters beginning with ``him_`` are forwarded to
    :class:`HIMEstimator`.  All other parameters match
    :class:`~rsl_rl.models.MLPModel`.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # -- HIM-specific parameters --
        him_temporal_steps: int = 5,
        him_num_one_step_obs: int = 61,
        him_vel_start_idx: int = 3,
        him_enc_hidden_dims: list[int] | None = None,
        him_tar_hidden_dims: list[int] | None = None,
        him_latent_dim: int = 16,
        him_num_prototype: int = 16,
        him_temperature: float = 3.0,
        him_learning_rate: float = 1e-3,
        him_device: str = "cpu",
        **_extra: Any,
    ):
        """Initialize the HIM-augmented actor model.

        Args:
            obs: Observation TensorDict (used to infer dims).
            obs_groups: Mapping from observation set names to group lists.
            obs_set: Which observation set this model consumes.
            output_dim: Action dimensionality.
            hidden_dims: MLP hidden layer sizes.
            activation: Activation function name.
            obs_normalization: Whether to normalize observations.
            distribution_cfg: Output distribution configuration.

            him_temporal_steps: Number of history steps for HIM estimator.
            him_num_one_step_obs: Dimension of a single-step observation.
            him_vel_start_idx: Start index of base_lin_vel in single-step obs.
            him_enc_hidden_dims: Source encoder hidden dims.
            him_tar_hidden_dims: Target encoder hidden dims.
            him_latent_dim: Dimension of implicit latent embedding.
            him_num_prototype: Number of SwAV prototypes.
            him_temperature: SwAV soft-assignment temperature.
            him_learning_rate: Estimator learning rate.
            him_device: Device for estimator parameters.
        """
        # Filter out unknown kwargs — rsl-rl may pass legacy/deprecated fields
        if _extra:
            him_kwargs = {
                k: _extra.pop(k)
                for k in list(_extra.keys())
                if k.startswith("him_")
            }
            # Apply late him_ kwargs if any were captured
            for k, v in him_kwargs.items():
                setattr(self, f"_{k}", v)
            if _extra:
                print(
                    f"[HIMActorModel] Ignoring unknown kwargs: {list(_extra.keys())}"
                )

        # -- Resolve HIM encoder dimensions --
        _enc_hidden_dims: list[int]
        if him_enc_hidden_dims is None:
            _enc_hidden_dims = [256, 64, him_latent_dim]
        else:
            _enc_hidden_dims = him_enc_hidden_dims
            him_latent_dim = _enc_hidden_dims[-1]

        _tar_hidden_dims: list[int] = him_tar_hidden_dims if him_tar_hidden_dims is not None else [128, 64]

        # -- Store HIM config BEFORE super().__init__() --
        # (super().__init__ calls _get_latent_dim which needs these values)
        self._him_num_one_step_obs: int = him_num_one_step_obs
        self._him_temporal_steps: int = him_temporal_steps
        self._him_latent_dim: int = him_latent_dim

        # Save HIM constructor args for deferred module creation.
        # The estimator must be created AFTER super().__init__() because
        # nn.Module.__init__() must run first.
        self._him_init_kwargs: dict[str, Any] = dict(
            temporal_steps=him_temporal_steps,
            num_one_step_obs=him_num_one_step_obs,
            vel_start_idx=him_vel_start_idx,
            enc_hidden_dims=_enc_hidden_dims,
            tar_hidden_dims=_tar_hidden_dims,
            activation=activation,
            learning_rate=him_learning_rate,
            num_prototype=him_num_prototype,
            temperature=him_temperature,
            device=him_device,
        )
        # Placeholder — replaced after super().__init__() below
        self._him_estimator: HIMEstimator  # type hint only; assigned post-super

        # Call the parent initializer (which calls _get_latent_dim)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # -- Now that self is a fully-initialized nn.Module,
        #    create the HIMEstimator as a proper sub-module. --
        self._him_estimator = HIMEstimator(**self._him_init_kwargs)
        del self._him_init_kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def him_estimator(self) -> HIMEstimator:
        """Return the HIM estimator module."""
        return self._him_estimator

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _get_latent_dim(self) -> int:
        """Return the dimension consumed by the MLP head.

        Overrides the parent to account for the extra HIM feature
        dimensions: velocity estimate (3) + latent embedding (latent_dim).
        """
        return self.obs_dim + 3 + self._him_latent_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState | None = None,
    ) -> torch.Tensor:
        """Build the latent for the MLP head.

        Steps:
        1. Concatenate observation groups (standard MLPModel behavior).
        2. Normalize the observation portion.
        3. Extract the observation history from the flattened tensor.
        4. Run the HIM estimator to produce ``vel_hat`` and ``latent_z``.
        5. Concatenate ``[normalized_obs, vel_hat, latent_z]``.

        Returns:
            latent: (B, obs_dim + 3 + latent_dim) tensor.
        """
        # Standard observation concatenation
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)  # (B, obs_dim)

        # Normalize the observation portion
        normalized_obs = self.obs_normalizer(flat_obs)

        # Run HIM estimator on the flattened history.
        # The estimator expects (B, temporal_steps * num_one_step_obs),
        # so we take only the most recent `temporal_steps` worth of frames.
        him_input_dim = self._him_temporal_steps * self._him_num_one_step_obs
        obs_history = flat_obs[:, -him_input_dim:]  # most recent steps

        with torch.no_grad():
            vel_hat, latent_z = self._him_estimator(obs_history)

        # Concatenate [normalized_obs, vel_hat, latent_z]
        return torch.cat([normalized_obs, vel_hat, latent_z], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics.

        Only normalizes the raw observation portion (not HIM features).
        """
        if not self.obs_normalization:
            return
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        mlp_obs = torch.cat(obs_list, dim=-1)
        # Guard: when obs_normalization=True, obs_normalizer is always
        # EmpiricalNormalization (has .update); nn.Identity does not.
        from rsl_rl.modules.normalization import EmpiricalNormalization
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(mlp_obs)
