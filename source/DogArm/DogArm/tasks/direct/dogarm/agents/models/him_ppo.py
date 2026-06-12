# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM-augmented PPO algorithm.

Extends :class:`rsl_rl.algorithms.PPO` with a Hybrid Internal Optimization
(HIO) step that updates the HIM estimator before each policy gradient step.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .him_actor_critic import HIMActorModel
from .him_estimator import HIMEstimator
from .him_storage import HIMRolloutStorage


class HIMPPO(PPO):
    """PPO with a HIM (Hybrid Internal Model) estimator.

    The estimator is updated via HIO **before** the PPO policy gradient
    step on each mini-batch.  This allows the policy to consume fresh
    terrain/environment embeddings during training.

    Reference:
        Long et al. "Hybrid Internal Model." ICLR 2024.
    """

    actor: HIMActorModel
    """The HIM-augmented actor (contains the estimator)."""
    critic: MLPModel
    """The standard critic model."""
    storage: HIMRolloutStorage
    """HIM-extended rollout storage."""
    him_estimator: HIMEstimator
    """Convenience reference to ``actor.him_estimator``."""
    transition: HIMRolloutStorage.Transition
    """Extended transition with next_critic_observations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Grab a reference to the estimator from the actor
        self.him_estimator = self.actor.him_estimator
        # Use HIM-specific transition type for next_critic_obs storage.
        # This replaces the parent's RolloutStorage.Transition with the
        # extended version that carries next_critic_observations.
        self.transition = HIMRolloutStorage.Transition()

    # ------------------------------------------------------------------
    # Override: record next_critic_obs during rollout
    # ------------------------------------------------------------------

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record transition with next_critic_observations for HIO."""
        # Update normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Compute next_critic_obs for HIO.
        # The critic observation is stored in the TensorDict under
        # the critic's configured observation groups.
        critic_obs_list = [obs[group] for group in self.critic.obs_groups]
        next_critic_obs = torch.cat(critic_obs_list, dim=-1).detach()

        self.transition.next_critic_observations = next_critic_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on timeouts
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record and clear
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    # ------------------------------------------------------------------
    # Override: HIO step + PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run HIO step then standard PPO update over mini-batches."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        mean_rnd_loss: float | None = 0.0 if self.rnd else None
        mean_symmetry_loss: float | None = 0.0 if self.symmetry else None

        # Clip returns and advantages — prevents value explosion from
        # terrain-induced reward spikes propagating through GAE.
        # (clone first: tensors may be inference-mode, rejecting in-place ops)
        self.storage.returns = self.storage.returns.clamp(-100.0, 100.0)
        self.storage.advantages = self.storage.advantages.clamp(-100.0, 100.0)

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            # All batch fields are guaranteed non-None in RL mode.
            # Use explicit cast after verifying the key fields.
            batch_obs = batch.observations
            batch_actions = batch.actions
            batch_values = batch.values
            batch_advantages = batch.advantages
            batch_returns = batch.returns
            batch_old_log_prob = batch.old_actions_log_prob
            batch_old_dist_params = batch.old_distribution_params

            if batch_obs is None or batch_actions is None:
                continue
            if batch_values is None or batch_advantages is None or batch_returns is None:
                continue
            if batch_old_log_prob is None or batch_old_dist_params is None:
                continue

            original_batch_size = batch_obs.batch_size[0]

            # Normalize advantages per mini-batch if configured
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )

            # Symmetry augmentation (from parent)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch_obs, batch_actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch_obs,
                    actions=batch_actions,
                )
                num_aug = int(batch_obs.batch_size[0] / original_batch_size)
                batch_old_log_prob = batch_old_log_prob.repeat(num_aug, 1)
                batch_values = batch_values.repeat(num_aug, 1)
                batch_advantages = batch_advantages.repeat(num_aug, 1)
                batch_returns = batch_returns.repeat(num_aug, 1)

            # ---- Forward pass through actor / critic ----
            self.actor(
                batch_obs,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch_actions)
            values = self.critic(
                batch_obs,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            # ---- KL-divergence adaptive LR ----
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch_old_dist_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ---- HIO: Update HIM Estimator ----
            obs_list = [
                batch_obs[group]
                for group in self.actor.obs_groups
            ]
            flat_obs_batch = torch.cat(obs_list, dim=-1)
            him_input_dim = (
                self.him_estimator.temporal_steps
                * self.him_estimator.num_one_step_obs
            )
            obs_history = flat_obs_batch[:, -him_input_dim:]

            next_critic_batch: torch.Tensor | None = getattr(
                batch, "next_critic_observations", None
            )

            if next_critic_batch is not None:
                est_loss, swap_loss = self.him_estimator.update(
                    obs_history=obs_history,
                    next_critic_obs=next_critic_batch,
                    lr=self.learning_rate,
                )
                mean_estimation_loss += est_loss
                mean_swap_loss += swap_loss

            # ---- PPO surrogate loss ----
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch_old_log_prob))
            surrogate = -torch.squeeze(batch_advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch_advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- Value loss ----
            if self.use_clipped_value_loss:
                value_clipped = batch_values + (values - batch_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch_returns).pow(2)
                value_losses_clipped = (value_clipped - batch_returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch_returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            # ---- Symmetry loss ----
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch_obs, _ = data_augmentation_func(
                        obs=batch_obs, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch_obs.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # ---- RND loss ----
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch_obs[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ---- Gradient step (PPO: actor + critic only) ----
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # ---- Accumulate metrics ----
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ---- Normalize by number of updates ----
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict: dict[str, float] = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimation": mean_estimation_loss,
            "swap": mean_swap_loss,
        }
        if self.rnd and mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry and mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    # ------------------------------------------------------------------
    # Save / Load with estimator state
    # ------------------------------------------------------------------

    def save(self) -> dict[str, Any]:
        """Add estimator state to saved dict."""
        saved_dict = super().save()
        saved_dict["him_estimator_state_dict"] = self.him_estimator.state_dict()
        saved_dict["him_estimator_optimizer_state_dict"] = (
            self.him_estimator.optimizer.state_dict()
        )
        return saved_dict

    def load(
        self,
        loaded_dict: dict[str, Any],
        load_cfg: dict | None,
        strict: bool,
    ) -> bool:
        """Load estimator state alongside PPO state."""
        result = super().load(loaded_dict, load_cfg, strict)
        if "him_estimator_state_dict" in loaded_dict:
            self.him_estimator.load_state_dict(
                loaded_dict["him_estimator_state_dict"], strict=strict
            )
        if "him_estimator_optimizer_state_dict" in loaded_dict:
            self.him_estimator.optimizer.load_state_dict(
                loaded_dict["him_estimator_optimizer_state_dict"]
            )
        return result

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict[str, Any],
        device: str,
    ) -> HIMPPO:
        """Construct a HIMPPO algorithm with HIMActorModel and HIMRolloutStorage.

        This is a drop-in replacement for :meth:`PPO.construct_algorithm`.
        Callers should set ``cfg["algorithm"]["class_name"]`` to
        ``HIMPPO``, ``cfg["actor"]["class_name"]`` to ``HIMActorModel``,
        and ``cfg["critic"]["class_name"]`` to ``MLPModel``.
        """
        # Resolve classes (pop class_name — consumed here, not forwarded).
        # resolve_callable accepts both strings and callable objects.
        raw_actor_cls = cfg["actor"].pop("class_name")
        raw_critic_cls = cfg["critic"].pop("class_name")
        raw_alg_cls = cfg["algorithm"].pop("class_name")

        actor_class: type[HIMActorModel] = cast(
            type[HIMActorModel], resolve_callable(raw_actor_cls)
        )
        critic_class: type[MLPModel] = cast(
            type[MLPModel], resolve_callable(raw_critic_cls)
        )
        alg_class: type[HIMPPO] = cast(
            type[HIMPPO], resolve_callable(raw_alg_cls)
        )

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND / symmetry (they may mutate cfg["algorithm"])
        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"], obs, cfg["obs_groups"], env
        )
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Build HIM-augmented actor
        actor: HIMActorModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"HIM Actor Model: {actor}")

        # Standard critic
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        # Determine critic obs dimension for HIM storage
        critic_obs_list = [obs[group] for group in cfg["obs_groups"]["critic"]]
        critic_obs_dim: int = int(sum(o.shape[-1] for o in critic_obs_list))

        # HIM storage
        storage = HIMRolloutStorage(
            training_type="rl",
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=[env.num_actions],
            device=device,
            critic_obs_dim=critic_obs_dim,
        )

        # Build algorithm
        alg: HIMPPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

        return alg
