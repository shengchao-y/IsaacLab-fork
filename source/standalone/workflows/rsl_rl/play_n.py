# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
import json

# local imports
import cli_args  # isort: skip
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--model_path_dir", type=str, default=None, help="Path to the model directory.")
parser.add_argument("--demo", action="store_true", default=False, help="If true, run in demo mode.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import time

def main():
    """Play with RSL-RL agent."""
    # Check how many models are there
    model_pathes = []
    for model_name in sorted(os.listdir(args_cli.model_path_dir)):
        if model_name.endswith(".pt"):
            print(f"[INFO] Found model: {model_name}")
            model_pathes.append(os.path.join(args_cli.model_path_dir, model_name))
    print(f"[INFO] Found {len(model_pathes)} models in the directory: {args_cli.model_path_dir}")
    
    if args_cli.demo:
        new_model_pathes = []
        for model_path in model_pathes:
            if "gage75" in model_path:
                new_model_pathes.append(model_path)
        model_pathes = new_model_pathes
        print(f"[INFO] Use Demo Mode")
    # Set number of environments to the number of models
    # args_cli.num_envs = len(model_pathes)
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=len(model_pathes), use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Loading experiment from directory: {log_root_path}")



    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # resume_baseline_path = get_checkpoint_path(log_root_path, args_cli.run_name_baseline, args_cli.checkpoint_baseline)
    # log_dir = os.path.dirname(args_cli.model_path_dir)
    log_dir = args_cli.model_path_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {args_cli.model_path_dir}")
    # load previously trained model
    runners = [OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device) for i in range(len(model_pathes))]
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner_baseline = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner_baseline.load(resume_baseline_path)
    # ppo_runner.load(resume_pathh)
    policies = []
    for i in range(len(model_pathes)):
        runners[i].load(model_pathes[i])
        policies.append(runners[i].get_inference_policy(device=env.unwrapped.device))

    # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # policy_baseline = ppo_runner_baseline.get_inference_policy(device=env.unwrapped.device)
    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.cat([policies[j](obs)[[j]] for j in range(len(policies))], dim=0)
            # actions_1 = policy(obs)
            # actions_2 = policy_baseline(obs)
            # print(actions_1.shape)
            # actions = torch.cat([actions_1[[0]], actions_2[[1]]], dim=0)
            time.sleep(0.1)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
