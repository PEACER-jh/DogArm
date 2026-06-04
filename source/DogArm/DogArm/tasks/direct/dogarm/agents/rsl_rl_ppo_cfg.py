# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for DogArm DirectRL task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


# Standard Gaussian distribution
_GAUSSIAN = RslRlMLPModelCfg.GaussianDistributionCfg(
    class_name="GaussianDistribution",
    init_std=1.0,
    std_type="log",  # LeggedManip_Lab: prevents std collapse
)


@configclass
class DogarmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for DogArm Go2+arm task."""

    # -- Runner --
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 1000
    check_for_nan = True  # LeggedManip_Lab: catch nan early
    experiment_name = "go2arm_direct"
    run_name = ""
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # -- Observation groups --
    # Actor sees history-stacked observations
    # Critic sees privileged information
    obs_groups: dict[str, list[str]] = {
        "actor": ["obs"],
        "critic": ["critic"],
    }

    # -- Actor: MLP [512, 256, 128] --
    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=_GAUSSIAN,
    )

    # -- Critic: MLP [512, 256, 128] --
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # -- PPO Algorithm --
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,   # official Go2: stronger exploration pressure
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
