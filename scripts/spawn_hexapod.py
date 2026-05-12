# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Spawn and visually preview the custom hexapod articulation.

Run with:

    /workspace/isaaclab/_isaac_sim/python.sh scripts/spawn_hexapod.py --headless --steps 120

For a short visual landing check with Isaac Sim / WebRTC, run without --headless:

    /workspace/isaaclab/_isaac_sim/python.sh scripts/spawn_hexapod.py --seconds 8 --hold-open --livestream 2
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn the hexapod and step the simulation briefly.")
parser.add_argument("--steps", type=int, default=None, help="Number of physics steps to run before exiting.")
parser.add_argument("--seconds", type=float, default=5.0, help="Duration to simulate when --steps is not provided.")
parser.add_argument("--hold-open", action="store_true", help="Keep the app open after the preview simulation finishes.")
parser.add_argument("--report-interval", type=float, default=0.5, help="Seconds between base-height status prints.")
parser.add_argument("--base-height", type=float, default=None, help="Override initial base/root height in meters.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from hexapod_lab.assets import HEXAPOD_CFG


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.4, 1.2], [0.0, 0.0, 0.15])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(HEXAPOD_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    root_state = robot.data.default_root_state.clone()
    if args_cli.base_height is not None:
        root_state[:, 2] = args_cli.base_height
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    print("[INFO]: Spawned hexapod", flush=True)
    print("[INFO]: Joint names:", robot.data.joint_names, flush=True)
    print("[INFO]: Body names:", robot.data.body_names, flush=True)
    print("[INFO]: Default base height:", float(robot.data.default_root_state[0, 2].item()), flush=True)

    sim_dt = sim.get_physics_dt()
    num_steps = args_cli.steps if args_cli.steps is not None else max(1, int(args_cli.seconds / sim_dt))
    report_every = max(1, int(args_cli.report_interval / sim_dt))
    joint_target = robot.data.default_joint_pos.clone()
    for step in range(num_steps):
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if step % report_every == 0 or step == num_steps - 1:
            base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            print(
                f"[INFO]: step {step + 1:04d}/{num_steps} "
                f"base xyz=({base_pos[0]:+.3f}, {base_pos[1]:+.3f}, {base_pos[2]:+.3f})",
                flush=True,
            )

    base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    joint_pos = robot.data.joint_pos[0].detach().cpu()
    print("[INFO]: Final base position:", [round(x, 4) for x in base_pos], flush=True)
    print("[INFO]: Final joint range:", round(float(torch.min(joint_pos)), 4), round(float(torch.max(joint_pos)), 4), flush=True)

    while args_cli.hold_open and simulation_app.is_running():
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


if __name__ == "__main__":
    main()
    if not args_cli.hold_open:
        os._exit(0)
    simulation_app.close()
