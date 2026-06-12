# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent trained with RSL-RL + HIM.

This is the HIM-aware counterpart of ``play.py``.  Standard ``play.py``
cannot load HIM-trained checkpoints because the model architecture differs
(extra ``_him_estimator`` sub-module and larger MLP input dimension).

Usage::

    python scripts/rsl_rl/play_him.py --task=Template-Dogarm-Direct-v0 --resume
    python scripts/rsl_rl/play_him.py --task=Template-Dogarm-Direct-v0 \\
        --load_run 2026-06-10_12-34-56
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent trained with RSL-RL + HIM.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # isort: skip  # register task configurations
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from DogArm.tasks.direct.dogarm.agents.models import (
    HIMActorModel,
    HIMPPO,
)

import DogArm.tasks  # isort: skip  # register DogArm task


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL + HIM agent."""
    # grab task name for checkpoint path
    task_name: str = args_cli.task.split(":")[-1]
    train_task_name: str = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path: str = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path: str
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir: str = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # Build HIM-compatible config (same injection as train_him.py)
    # ------------------------------------------------------------------
    train_cfg = agent_cfg.to_dict()
    actor_obs_groups: list[str] = train_cfg["obs_groups"]["actor"]

    # Determine single-step observation dimension from the environment
    obs_sample = env.get_observations()
    single_step_flat_dim: int = sum(
        int(obs_sample[group].shape[-1]) for group in actor_obs_groups
    )
    num_obs_history: int = env_cfg.num_obs_history_steps
    single_step_dim: int = single_step_flat_dim // num_obs_history

    # Inject HIM classes
    train_cfg["actor"]["class_name"] = HIMActorModel
    train_cfg["critic"]["class_name"] = MLPModel
    train_cfg["algorithm"]["class_name"] = HIMPPO

    # HIM parameters (must match training configuration)
    train_cfg["actor"]["him_temporal_steps"] = 5
    train_cfg["actor"]["him_num_one_step_obs"] = single_step_dim
    train_cfg["actor"]["him_vel_start_idx"] = 3
    train_cfg["actor"]["him_enc_hidden_dims"] = [256, 64, 16]
    train_cfg["actor"]["him_tar_hidden_dims"] = [128, 64]
    train_cfg["actor"]["him_latent_dim"] = 16
    train_cfg["actor"]["him_num_prototype"] = 16

    print(
        f"[INFO] HIM actor: single_step_dim={single_step_dim}, "
        f"mlp_input={single_step_flat_dim + 3 + 16}"
    )

    # ------------------------------------------------------------------
    # Create runner and load checkpoint
    # ------------------------------------------------------------------
    runner: OnPolicyRunner | DistillationRunner
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------------------------------------------
    # Export (skipped — standard JIT/ONNX path omits the HIM estimator
    # and feature concat, causing input shape mismatch: 610 vs 629).
    # ------------------------------------------------------------------
    print("[INFO] Skipping JIT/ONNX export (unsupported for HIM models).")

    dt: float = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep: int = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            # reset recurrent states (no-op for feed-forward MLP policies)
            policy.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time: float = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]  # hydra injects args via decorator
    simulation_app.close()
