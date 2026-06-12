#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train RL agent with RSL-RL using HIM (Hybrid Internal Model).

This script is a drop-in replacement for ``train.py`` that augments the
standard PPO pipeline with a HIM estimator.

The HIM estimator learns an implicit terrain/environment embedding from
proprioceptive history via contrastive learning (SwAV) and explicitly
predicts body-frame velocity via MSE supervision.

Usage::

    # Same CLI as train.py, plus HIM-specific flags
    python scripts/rsl_rl/train_him.py --task=Template-Dogarm-Direct-v0 \\
        --num_envs 4096 --max_iterations 5000

    # Resume from a standard checkpoint (HIM estimator will be fresh)
    python scripts/rsl_rl/train_him.py --task=Template-Dogarm-Direct-v0 \\
        --num_envs 4096 --resume --load_run <run_name>

Reference:
    Long et al. "Hybrid Internal Model: Learning Agile Legged Locomotion
    with Simulated Robot Response." ICLR 2024.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL + HIM.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

# -- HIM-specific arguments --
parser.add_argument(
    "--him_temporal_steps", type=int, default=5,
    help="Number of history steps for HIM estimator (default: 5)."
)
parser.add_argument(
    "--him_latent_dim", type=int, default=16,
    help="Dimension of implicit latent embedding (default: 16)."
)
parser.add_argument(
    "--him_num_prototype", type=int, default=16,
    help="Number of SwAV prototypes (default: 16)."
)
parser.add_argument(
    "--him_temperature", type=float, default=3.0,
    help="SwAV soft-assignment temperature (default: 3.0)."
)
parser.add_argument(
    "--him_enc_hidden_dims", type=str, default="256,64",
    help="Comma-separated source encoder hidden dims (default: 256,64). Final is latent_dim."
)
parser.add_argument(
    "--him_learning_rate", type=float, default=1e-3,
    help="Learning rate for HIM estimator (default: 1e-3)."
)
parser.add_argument(
    "--him_vel_start_idx", type=int, default=3,
    help="Start index of base_lin_vel in single-step actor obs (default: 3)."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# HIM modules (from DogArm agents)
from DogArm.tasks.direct.dogarm.agents.models import (
    HIMActorModel,
    HIMEstimator,
    HIMPPO,
    HIMRolloutStorage,
)

# import logger
logger = logging.getLogger(__name__)

import DogArm.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Parse HIM-specific arguments
# ------------------------------------------------------------------

def _parse_him_hidden_dims(dims_str: str) -> list[int]:
    """Parse comma-separated hidden dims like '256,64' into a list of ints."""
    return [int(x.strip()) for x in dims_str.split(",") if x.strip()]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL + HIM agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. "
            "No IO descriptors will be exported."
        )

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # Build the training configuration with HIM components injected
    # ------------------------------------------------------------------
    train_cfg = agent_cfg.to_dict()

    # -- Determine single-step observation dimension from the environment --
    # The env returns "obs" containing history-stacked observations.
    # We extract single_step_dim from the actor observation group(s).
    obs_sample = env.get_observations()
    actor_obs_groups = train_cfg["obs_groups"]["actor"]
    single_step_flat_dim = 0
    for group in actor_obs_groups:
        single_step_flat_dim += obs_sample[group].shape[-1]

    # For history-stacked observations, single_step_dim = flattened_dim / history_steps
    num_obs_history = env_cfg.num_obs_history_steps
    single_step_dim = single_step_flat_dim // num_obs_history
    print(f"[INFO] HIM: single-step obs dim = {single_step_dim} "
          f"(flattened={single_step_flat_dim}, history_steps={num_obs_history})")

    # -- Parse HIM encoder hidden dims --
    him_enc_dims = _parse_him_hidden_dims(args_cli.him_enc_hidden_dims)
    him_enc_dims.append(args_cli.him_latent_dim)  # [256, 64, 16]
    him_tar_dims = [128, 64]  # target encoder defaults

    # -- Inject HIM classes and parameters into the config --
    train_cfg["actor"]["class_name"] = HIMActorModel
    train_cfg["critic"]["class_name"] = MLPModel
    train_cfg["algorithm"]["class_name"] = HIMPPO

    # HIM estimator parameters (passed through actor config)
    train_cfg["actor"]["him_temporal_steps"] = args_cli.him_temporal_steps
    train_cfg["actor"]["him_num_one_step_obs"] = single_step_dim
    train_cfg["actor"]["him_vel_start_idx"] = args_cli.him_vel_start_idx
    train_cfg["actor"]["him_enc_hidden_dims"] = him_enc_dims
    train_cfg["actor"]["him_tar_hidden_dims"] = him_tar_dims
    train_cfg["actor"]["him_latent_dim"] = args_cli.him_latent_dim
    train_cfg["actor"]["him_num_prototype"] = args_cli.him_num_prototype
    train_cfg["actor"]["him_temperature"] = args_cli.him_temperature
    train_cfg["actor"]["him_learning_rate"] = args_cli.him_learning_rate

    print(f"[INFO] HIM config: temporal_steps={args_cli.him_temporal_steps}, "
          f"latent_dim={args_cli.him_latent_dim}, "
          f"num_prototype={args_cli.him_num_prototype}, "
          f"enc_hidden={him_enc_dims}")

    # -- Create the runner (uses HIMPPO.construct_algorithm internally) --
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # Note: resuming from a non-HIM checkpoint will initialize the
        # HIM estimator from scratch while loading the actor/critic weights.
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # pyright: ignore[reportCallIssue]  # hydra injects args via decorator
    # close sim app
    simulation_app.close()
