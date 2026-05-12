# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RL-Games checkpoint and write a base-motion trace CSV."""

import argparse
import csv
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace a checkpoint of an RL-Games agent.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of policy steps to run.")
parser.add_argument("--trace_csv", type=Path, required=True, help="CSV path for trace output.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--fixed_forward_velocity", type=float, default=None, help="Override command +Y velocity range.")
parser.add_argument("--fixed_lateral_velocity", type=float, default=None, help="Override command +X velocity range.")
parser.add_argument("--fixed_yaw_velocity", type=float, default=None, help="Override command yaw velocity range.")
parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions instead of policy mean.")
parser.add_argument("--joint_trace_csv", type=Path, default=None, help="Optional long-format per-joint action/limit trace.")
parser.add_argument("--joint_trace_every", type=int, default=1, help="Write per-joint trace every N policy steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hexapod_lab.tasks  # noqa: F401


def _load_joint_limits() -> dict[str, tuple[float, float]]:
    repo_root = Path(__file__).resolve().parents[2]
    urdf_path = repo_root / "assets" / "robots" / "hexapod" / "urdf" / "hexapod.urdf"
    limits = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.iter("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.fixed_forward_velocity is not None:
        env_cfg.forward_velocity_range = (args_cli.fixed_forward_velocity, args_cli.fixed_forward_velocity)
    if args_cli.fixed_lateral_velocity is not None:
        env_cfg.lateral_velocity_range = (args_cli.fixed_lateral_velocity, args_cli.fixed_lateral_velocity)
    if args_cli.fixed_yaw_velocity is not None:
        env_cfg.yaw_velocity_range = (args_cli.fixed_yaw_velocity, args_cli.fixed_yaw_velocity)

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    base_env = gym_env.unwrapped

    env = RlGamesVecEnvWrapper(gym_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    if args_cli.stochastic:
        agent.is_deterministic = False

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    fixed_command = (
        args_cli.fixed_lateral_velocity,
        args_cli.fixed_forward_velocity,
        args_cli.fixed_yaw_velocity,
    )

    def apply_fixed_command():
        if any(value is not None for value in fixed_command):
            if fixed_command[0] is not None:
                base_env._commands[:, 0] = fixed_command[0]
            if fixed_command[1] is not None:
                base_env._commands[:, 1] = fixed_command[1]
            if fixed_command[2] is not None:
                base_env._commands[:, 2] = fixed_command[2]

    def patch_obs_command(obs_tensor):
        if any(value is not None for value in fixed_command):
            obs_tensor = obs_tensor.clone()
            if fixed_command[0] is not None:
                obs_tensor[:, 9] = fixed_command[0]
            if fixed_command[1] is not None:
                obs_tensor[:, 10] = fixed_command[1]
            if fixed_command[2] is not None:
                obs_tensor[:, 11] = fixed_command[2]
        return obs_tensor

    apply_fixed_command()
    obs = patch_obs_command(obs)
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    joint_limits = _load_joint_limits()
    joint_names = list(getattr(base_env._robot.data, "joint_names", []))
    if not joint_names:
        joint_names = list(getattr(base_env._robot, "joint_names", []))
    if not joint_names:
        joint_names = [f"joint_{idx}" for idx in range(base_env._actions.shape[1])]

    args_cli.trace_csv.parent.mkdir(parents=True, exist_ok=True)
    joint_file = None
    joint_writer = None
    if args_cli.joint_trace_csv is not None:
        args_cli.joint_trace_csv.parent.mkdir(parents=True, exist_ok=True)
        joint_file = args_cli.joint_trace_csv.open("w", newline="")
        joint_writer = csv.DictWriter(
            joint_file,
            fieldnames=[
                "step",
                "time_s",
                "joint_name",
                "action",
                "action_bound_margin",
                "target_pos",
                "joint_pos",
                "lower",
                "upper",
                "target_lower_margin",
                "target_upper_margin",
                "joint_lower_margin",
                "joint_upper_margin",
            ],
        )
        joint_writer.writeheader()

    try:
        trace_file = args_cli.trace_csv.open("w", newline="")
        writer = csv.DictWriter(
            trace_file,
            fieldnames=[
                "step",
                "time_s",
                "done",
                "x_w",
                "y_w",
                "z_w",
                "vx_w",
                "vy_w",
                "vz_w",
                "vx_b",
                "vy_b",
                "vz_b",
                "cmd_x",
                "cmd_y",
                "cmd_yaw",
                "action_mean_abs",
                "action_max_abs",
                "action_delta_mean_abs",
                "joint_pos_delta_mean_abs",
                "joint_vel_mean_abs",
            ],
        )
        writer.writeheader()
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                apply_fixed_command()
                obs = patch_obs_command(obs)
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
                action_delta = actions - base_env._actions
                obs, _, dones, _ = env.step(actions)
                done = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])

            root_pos_w = base_env._robot.data.root_pos_w[0].detach().cpu()
            root_lin_vel_w = base_env._robot.data.root_lin_vel_w[0].detach().cpu()
            root_lin_vel_b = base_env._robot.data.root_lin_vel_b[0].detach().cpu()
            commands = base_env._commands[0].detach().cpu()
            joint_pos_delta = (base_env._robot.data.joint_pos[0] - base_env._robot.data.default_joint_pos[0]).detach().cpu()
            joint_vel = base_env._robot.data.joint_vel[0].detach().cpu()
            action_cpu = actions[0].detach().cpu()
            action_delta_cpu = action_delta[0].detach().cpu()
            target_pos = (base_env.cfg.action_scale * actions[0] + base_env._robot.data.default_joint_pos[0]).detach().cpu()
            joint_pos = base_env._robot.data.joint_pos[0].detach().cpu()
            writer.writerow(
                {
                    "step": step,
                    "time_s": f"{step * base_env.step_dt:.4f}",
                    "done": int(done),
                    "x_w": f"{root_pos_w[0].item():.6f}",
                    "y_w": f"{root_pos_w[1].item():.6f}",
                    "z_w": f"{root_pos_w[2].item():.6f}",
                    "vx_w": f"{root_lin_vel_w[0].item():.6f}",
                    "vy_w": f"{root_lin_vel_w[1].item():.6f}",
                    "vz_w": f"{root_lin_vel_w[2].item():.6f}",
                    "vx_b": f"{root_lin_vel_b[0].item():.6f}",
                    "vy_b": f"{root_lin_vel_b[1].item():.6f}",
                    "vz_b": f"{root_lin_vel_b[2].item():.6f}",
                    "cmd_x": f"{commands[0].item():.6f}",
                    "cmd_y": f"{commands[1].item():.6f}",
                    "cmd_yaw": f"{commands[2].item():.6f}",
                    "action_mean_abs": f"{torch.mean(torch.abs(action_cpu)).item():.6f}",
                    "action_max_abs": f"{torch.max(torch.abs(action_cpu)).item():.6f}",
                    "action_delta_mean_abs": f"{torch.mean(torch.abs(action_delta_cpu)).item():.6f}",
                    "joint_pos_delta_mean_abs": f"{torch.mean(torch.abs(joint_pos_delta)).item():.6f}",
                    "joint_vel_mean_abs": f"{torch.mean(torch.abs(joint_vel)).item():.6f}",
                }
            )
            if joint_writer is not None and step % args_cli.joint_trace_every == 0:
                for joint_idx, joint_name in enumerate(joint_names):
                    lower, upper = joint_limits.get(joint_name, (float("nan"), float("nan")))
                    target = target_pos[joint_idx].item()
                    pos = joint_pos[joint_idx].item()
                    action_value = action_cpu[joint_idx].item()
                    joint_writer.writerow(
                        {
                            "step": step,
                            "time_s": f"{step * base_env.step_dt:.4f}",
                            "joint_name": joint_name,
                            "action": f"{action_value:.6f}",
                            "action_bound_margin": f"{1.0 - abs(action_value):.6f}",
                            "target_pos": f"{target:.6f}",
                            "joint_pos": f"{pos:.6f}",
                            "lower": f"{lower:.6f}",
                            "upper": f"{upper:.6f}",
                            "target_lower_margin": f"{target - lower:.6f}",
                            "target_upper_margin": f"{upper - target:.6f}",
                            "joint_lower_margin": f"{pos - lower:.6f}",
                            "joint_upper_margin": f"{upper - pos:.6f}",
                        }
                    )
            if isinstance(obs, dict):
                obs = obs["obs"]
    finally:
        trace_file.close()
        if joint_file is not None:
            joint_file.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
