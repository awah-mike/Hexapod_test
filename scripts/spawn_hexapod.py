# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Spawn-smoke test for the custom hexapod articulation.

Run with:

    /workspace/isaaclab/_isaac_sim/python.sh scripts/spawn_hexapod.py --headless --steps 120
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn the hexapod and step the simulation briefly.")
parser.add_argument("--steps", type=int, default=120, help="Number of physics steps to run before exiting.")
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
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    print("[INFO]: Spawned hexapod", flush=True)
    print("[INFO]: Joint names:", robot.data.joint_names, flush=True)
    print("[INFO]: Body names:", robot.data.body_names, flush=True)
    print("[INFO]: Default base height:", float(robot.data.default_root_state[0, 2].item()), flush=True)

    sim_dt = sim.get_physics_dt()
    joint_target = robot.data.default_joint_pos.clone()
    for _ in range(args_cli.steps):
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

    base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    joint_pos = robot.data.joint_pos[0].detach().cpu()
    print("[INFO]: Final base position:", [round(x, 4) for x in base_pos], flush=True)
    print("[INFO]: Final joint range:", round(float(torch.min(joint_pos)), 4), round(float(torch.max(joint_pos)), 4), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
