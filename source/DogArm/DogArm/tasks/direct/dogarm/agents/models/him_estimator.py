# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid Internal Model (HIM) Estimator.

Reference:
    Long et al. "Hybrid Internal Model: Learning Agile Legged Locomotion
    with Simulated Robot Response." ICLR 2024.
    https://arxiv.org/abs/2312.11460

The HIM Estimator extracts a hybrid internal embedding from a history of
proprioceptive observations:

1. **Explicit velocity estimate** (3-dim): supervised with MSE against
   ground-truth body-frame linear velocity.
2. **Implicit latent embedding** (16-dim): optimized via SwAV-style
   contrastive learning to be predictive of the robot's successor state.

The estimator is trained via Hybrid Internal Optimization (HIO) which
runs before each PPO policy gradient step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HIMEstimator(nn.Module):
    """Hybrid Internal Model estimator with SwAV contrastive learning.

    The source encoder processes a history of observations to predict
    (a) the robot's current velocity and (b) a latent embedding that
    encodes terrain/environment properties. The target encoder embeds
    the next observation for contrastive learning.

    Architecture::

        obs_history (B, T * obs_dim)
            │
            ▼
        Source Encoder (MLP)  ──► vel_hat (B, 3)
            │                     ──► z_s (B, latent_dim)  ── L2 norm
            │
        Target Encoder (MLP)  ◄── next_obs (B, obs_dim)
            │
            ▼
            z_t (B, latent_dim) ── L2 norm

        Prototypes (K, latent_dim)
            │
            ▼
        SwAV swapped-assignment loss + Velocity MSE loss
    """

    encoder: nn.Sequential
    """Source encoder: history → vel(3) + latent(latent_dim)."""
    target: nn.Sequential
    """Target encoder: next_obs → latent(latent_dim)."""
    proto: nn.Embedding
    """Prototype embeddings (K, latent_dim)."""
    optimizer: optim.Adam
    """Estimator optimizer (separate from PPO optimizer)."""

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        vel_start_idx: int = 3,
        enc_hidden_dims: list[int] | None = None,
        tar_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 16,
        temperature: float = 3.0,
        device: str = "cpu",
        **_extra: Any,
    ):
        """Initialize the HIM Estimator.

        Args:
            temporal_steps: Number of history steps stacked (e.g., 5).
            num_one_step_obs: Dimension of a single observation step.
            vel_start_idx: Start index of ground-truth velocity in the
                single-step observation (default: 3, after base_ang_vel).
            enc_hidden_dims: Hidden layer sizes for the source encoder.
                Default: [256, 64, 16] where the last value is the latent
                dimension (additional 3 outputs are reserved for velocity).
            tar_hidden_dims: Hidden layer sizes for the target encoder.
                Default: [128, 64].
            activation: Activation function name.
            learning_rate: Learning rate for the estimator optimizer.
            max_grad_norm: Max gradient norm for estimator clipping.
            num_prototype: Number of prototypes for SwAV clustering.
            temperature: Temperature for soft assignment in SwAV.
            device: Device to place parameters on.
        """
        if _extra:
            print(
                "[HIMEstimator] Ignoring unexpected arguments: "
                + str(list(_extra.keys()))
            )
        super().__init__()

        # Resolve defaults (narrow types with explicit locals)
        _enc_hidden_dims: list[int] = enc_hidden_dims if enc_hidden_dims is not None else [256, 64, 16]
        _tar_hidden_dims: list[int] = tar_hidden_dims if tar_hidden_dims is not None else [128, 64]

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.vel_start_idx = vel_start_idx
        self.num_latent: int = _enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        act_fn = _get_activation(activation)

        # -- Source Encoder: history → vel(3) + latent(latent_dim) --
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers: list[nn.Module] = []
        for li in range(len(_enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, _enc_hidden_dims[li]), act_fn]
            enc_input_dim = _enc_hidden_dims[li]
        enc_layers += [nn.Linear(enc_input_dim, _enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # -- Target Encoder: next_obs → latent(latent_dim) --
        tar_input_dim = self.num_one_step_obs
        tar_layers: list[nn.Module] = []
        for li in range(len(_tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, _tar_hidden_dims[li]), act_fn]
            tar_input_dim = _tar_hidden_dims[li]
        tar_layers += [nn.Linear(tar_input_dim, _enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # -- Prototype embeddings --
        self.proto = nn.Embedding(num_prototype, _enc_hidden_dims[-1])

        # -- Optimizer --
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------
    # Forward / Inference
    # ------------------------------------------------------------------

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference-mode forward pass (gradient-free).

        Args:
            obs_history: (B, temporal_steps * num_one_step_obs) flattened
                history of observations.

        Returns:
            vel_hat: (B, 3) estimated body-frame linear velocity.
            latent_z: (B, latent_dim) L2-normalized implicit embedding.
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def encode(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as forward() — used during rollout for clarity."""
        return self.forward(obs_history)

    # ------------------------------------------------------------------
    # HIO Update
    # ------------------------------------------------------------------

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """Run one HIO (Hybrid Internal Optimization) step.

        Args:
            obs_history: (B, temporal_steps * num_one_step_obs) source input.
            next_critic_obs: (B, critic_obs_dim) the NEXT privileged
                observation.  The velocity ground-truth is extracted from
                ``next_critic_obs[:, vel_start_idx:vel_start_idx+3]``.
            lr: Optional override for the estimator learning rate.

        Returns:
            estimation_loss: Scalar MSE loss for velocity prediction.
            swap_loss: Scalar SwAV contrastive loss.
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Ground-truth velocity from the next critic observation.
        vel_gt = next_critic_obs[
            :, self.vel_start_idx : self.vel_start_idx + 3
        ].detach()

        # Target encoder input: full single-step observation
        next_obs = next_critic_obs.detach()[
            :, : self.num_one_step_obs
        ]

        # Source encoding
        z_s = self.encoder(obs_history)  # (B, latent_dim + 3)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        # Target encoding
        z_t = self.target(next_obs)

        # L2 normalize features
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # Normalize prototypes
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        # Cosine similarities → cluster assignment scores
        score_s = z_s @ self.proto.weight.T  # (B, K)
        score_t = z_t @ self.proto.weight.T

        # Sinkhorn-Knopp targets (no gradient)
        with torch.no_grad():
            q_s = _sinkhorn(score_s)
            q_t = _sinkhorn(score_t)

        # Log-probabilities under soft assignment
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # SwAV swapped-assignment loss
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        # Velocity estimation loss
        estimation_loss = F.mse_loss(pred_vel, vel_gt)

        # Combined loss
        total_loss = estimation_loss + swap_loss

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "crelu": nn.ReLU,
    "silu": nn.SiLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(act_name: str) -> nn.Module:
    """Resolve an activation function name to a module."""
    cls = _ACTIVATION_REGISTRY.get(act_name)
    if cls is not None:
        return cls()
    print(f"[HIMEstimator] Invalid activation '{act_name}', using ELU.")
    return nn.ELU()


@torch.no_grad()
def _sinkhorn(
    out: torch.Tensor, eps: float = 0.05, iters: int = 3
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for balanced cluster assignment.

    Args:
        out: (B, K) logits or cosine scores.
        eps: Entropy regularization strength.
        iters: Number of Sinkhorn iterations.

    Returns:
        Q: (B, K) soft assignment matrix (rows sum to 1/K).
    """
    Q = torch.exp(out / eps).T  # (K, B)
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for _ in range(iters):
        # Normalize each row: total weight per prototype = 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        # Normalize each column: total weight per sample = 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    return (Q * B).T  # (B, K)
