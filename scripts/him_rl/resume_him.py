# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Resume HIM training from a checkpoint.

Two modes:
  1. Same-folder (default): logs continue in the checkpoint's parent directory.
  2. Fresh-folder (--fresh):  read model from any path, create new timestamped
     log directory like a fresh train.

Usage::

    python scripts/rsl_rl/resume_him.py --task=Template-Dogarm-Direct-v0 \\
        --num_envs 5000 --resume_path logs/.../model_5000.pt

    python scripts/rsl_rl/resume_him.py --task=Template-Dogarm-Direct-v0 \\
        --num_envs 5000 --resume_path logs/.../model_5000.pt --fresh
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime

from isaaclab.app import AppLauncher

# ---- argparse ----
parser = argparse.ArgumentParser(description="Resume HIM RL training from a checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume_path", type=str, required=True, help="Path to model .pt checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--fresh", action="store_true", default=False,
    help="Create a new timestamped log directory instead of continuing in the checkpoint folder",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest."""

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # isort: skip
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from DogArm.tasks.direct.dogarm.agents.models import (
    HIMActorModel,
    HIMPPO,
)

logger = logging.getLogger(__name__)
import DogArm.tasks  # isort: skip

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # ---- Load configs ----
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    agent_cfg.seed = args_cli.seed

    import importlib.metadata as metadata
    rsl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)

    # ---- Checkpoint path ----
    checkpoint_path = os.path.abspath(args_cli.resume_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")

    # ---- Log directory ----
    if args_cli.fresh:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            ts += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root, ts)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Fresh log directory: {log_dir}")
    else:
        log_dir = os.path.dirname(checkpoint_path)
        print(f"[INFO] Logging to same folder: {log_dir}")

    env_cfg.log_dir = log_dir

    # ---- Create env ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- Build HIM-compatible config ----
    train_cfg = agent_cfg.to_dict()

    obs_sample = env.get_observations()
    actor_obs_groups = train_cfg["obs_groups"]["actor"]
    single_step_flat_dim = sum(int(obs_sample[group].shape[-1]) for group in actor_obs_groups)
    num_obs_history = env_cfg.num_obs_history_steps
    single_step_dim = single_step_flat_dim // num_obs_history

    train_cfg["actor"]["class_name"] = HIMActorModel
    train_cfg["critic"]["class_name"] = MLPModel
    train_cfg["algorithm"]["class_name"] = HIMPPO

    train_cfg["actor"]["him_temporal_steps"] = 5
    train_cfg["actor"]["him_num_one_step_obs"] = single_step_dim
    train_cfg["actor"]["him_vel_start_idx"] = 3
    train_cfg["actor"]["him_enc_hidden_dims"] = [256, 64, 16]
    train_cfg["actor"]["him_tar_hidden_dims"] = [128, 64]
    train_cfg["actor"]["him_latent_dim"] = 16
    train_cfg["actor"]["him_num_prototype"] = 16

    print(f"[INFO] HIM actor: single_step_dim={single_step_dim}, "
          f"mlp_input={single_step_flat_dim + 3 + 16}")

    # ---- Create runner ----
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # ---- Load checkpoint ----
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path)

    if args_cli.max_iterations is not None:
        start_it = runner.current_learning_iteration
        remaining = max(1, args_cli.max_iterations - start_it)
        agent_cfg.max_iterations = remaining
        total = start_it + remaining
        print(f"[INFO] Resuming from iter {start_it}, will train {remaining} more -> total {total}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
