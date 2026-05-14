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
import csv
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn the hexapod and step the simulation briefly.")
parser.add_argument("--steps", type=int, default=None, help="Number of physics steps to run before exiting.")
parser.add_argument("--seconds", type=float, default=5.0, help="Duration to simulate when --steps is not provided.")
parser.add_argument("--hold-open", action="store_true", help="Keep the app open after the preview simulation finishes.")
parser.add_argument("--report-interval", type=float, default=0.5, help="Seconds between base-height status prints.")
parser.add_argument("--base-height", type=float, default=None, help="Override initial base/root height in meters.")
parser.add_argument(
    "--screenshot-dir",
    type=Path,
    default=None,
    help="Directory where initial and final viewport screenshots should be written.",
)
parser.add_argument(
    "--snapshot-steps",
    type=str,
    default="",
    help="Comma-separated physics step numbers to capture when --screenshot-dir is set.",
)
parser.add_argument(
    "--diagnostics-csv",
    type=Path,
    default=None,
    help="Optional CSV path for logging base height, joint errors, torques, and torque saturation.",
)
parser.add_argument(
    "--diagnostics-every",
    type=int,
    default=1,
    help="Write one diagnostics row every N physics steps when --diagnostics-csv is set.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from hexapod_lab.assets import HEXAPOD_CFG


def capture_screenshot(path: Path) -> None:
    """Capture the active Isaac Sim viewport to a PNG file."""
    import asyncio
    import inspect

    import omni.kit.app
    import omni.renderer_capture
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    path.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport is available for screenshot capture.")

    app = omni.kit.app.get_app()
    for _ in range(4):
        app.update()
    capture = capture_viewport_to_file(viewport, file_path=str(path))
    result = capture.wait_for_result(10)
    if inspect.isawaitable(result):
        try:
            asyncio.get_event_loop().run_until_complete(asyncio.wait_for(result, timeout=10.0))
        except TimeoutError:
            print(f"[WARN]: Timed out while waiting for screenshot capture: {path}", flush=True)
            return
    capture_iface = omni.renderer_capture.acquire_renderer_capture_interface()
    for _ in range(3):
        capture_iface.wait_async_capture()
        app.update()
    print(f"[INFO]: Wrote screenshot: {path}", flush=True)


def _as_env_joint_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return a tensor with shape (num_envs, num_joints)."""
    return value.unsqueeze(0) if value.ndim == 1 else value


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.4, 1.2], [0.0, 0.0, 0.15])

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
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
    if args_cli.screenshot_dir is not None:
        capture_screenshot(args_cli.screenshot_dir / "hexapod_initial_standing_pose.png")

    num_steps = args_cli.steps if args_cli.steps is not None else max(1, int(args_cli.seconds / sim_dt))
    report_every = max(1, int(args_cli.report_interval / sim_dt))
    snapshot_steps = {
        int(item.strip())
        for item in args_cli.snapshot_steps.split(",")
        if item.strip()
    }
    joint_target = robot.data.default_joint_pos.clone()
    joint_names = list(robot.data.joint_names)
    max_abs_applied_by_joint = torch.zeros(len(joint_names), device=robot.device)
    max_abs_computed_by_joint = torch.zeros(len(joint_names), device=robot.device)
    max_abs_error_by_joint = torch.zeros(len(joint_names), device=robot.device)
    max_saturated_count = 0
    diagnostics_file = None
    diagnostics_writer = None
    if args_cli.diagnostics_csv is not None:
        args_cli.diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_file = args_cli.diagnostics_csv.open("w", newline="")
        diagnostics_writer = csv.DictWriter(
            diagnostics_file,
            fieldnames=[
                "step",
                "time_s",
                "base_z_m",
                "max_abs_joint_pos_error_rad",
                "max_abs_joint_vel_rad_s",
                "max_abs_applied_torque_nm",
                "max_abs_computed_torque_nm",
                "max_effort_limit_nm",
                "saturated_joint_count",
            ],
        )
        diagnostics_writer.writeheader()

    for step in range(num_steps):
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        joint_pos_error = torch.abs(joint_target - robot.data.joint_pos)
        joint_vel = torch.abs(robot.data.joint_vel)
        applied_torque = torch.abs(robot.data.applied_torque)
        computed_torque = torch.abs(robot.data.computed_torque)
        effort_limits = torch.abs(_as_env_joint_tensor(robot.data.joint_effort_limits))
        saturated = applied_torque >= (0.98 * effort_limits)

        max_abs_applied_by_joint = torch.maximum(max_abs_applied_by_joint, applied_torque[0])
        max_abs_computed_by_joint = torch.maximum(max_abs_computed_by_joint, computed_torque[0])
        max_abs_error_by_joint = torch.maximum(max_abs_error_by_joint, joint_pos_error[0])
        saturated_count = int(torch.count_nonzero(saturated[0]).item())
        max_saturated_count = max(max_saturated_count, saturated_count)

        if diagnostics_writer is not None and step % max(1, args_cli.diagnostics_every) == 0:
            diagnostics_writer.writerow(
                {
                    "step": step + 1,
                    "time_s": f"{(step + 1) * sim_dt:.6f}",
                    "base_z_m": f"{float(robot.data.root_pos_w[0, 2].item()):.6f}",
                    "max_abs_joint_pos_error_rad": f"{float(torch.max(joint_pos_error).item()):.6f}",
                    "max_abs_joint_vel_rad_s": f"{float(torch.max(joint_vel).item()):.6f}",
                    "max_abs_applied_torque_nm": f"{float(torch.max(applied_torque).item()):.6f}",
                    "max_abs_computed_torque_nm": f"{float(torch.max(computed_torque).item()):.6f}",
                    "max_effort_limit_nm": f"{float(torch.max(effort_limits).item()):.6f}",
                    "saturated_joint_count": saturated_count,
                }
            )

        if step % report_every == 0 or step == num_steps - 1:
            base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            print(
                f"[INFO]: step {step + 1:04d}/{num_steps} "
                f"base xyz=({base_pos[0]:+.3f}, {base_pos[1]:+.3f}, {base_pos[2]:+.3f})",
                flush=True,
            )
        if args_cli.screenshot_dir is not None and (step + 1) in snapshot_steps:
            capture_screenshot(args_cli.screenshot_dir / f"hexapod_step_{step + 1:04d}.png")

    base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    joint_pos = robot.data.joint_pos[0].detach().cpu()
    print("[INFO]: Final base position:", [round(x, 4) for x in base_pos], flush=True)
    print("[INFO]: Final joint range:", round(float(torch.min(joint_pos)), 4), round(float(torch.max(joint_pos)), 4), flush=True)
    print(
        "[INFO]: Max applied torque:",
        round(float(torch.max(max_abs_applied_by_joint).item()), 4),
        "Nm; max computed torque:",
        round(float(torch.max(max_abs_computed_by_joint).item()), 4),
        "Nm; max saturated joints:",
        max_saturated_count,
        flush=True,
    )
    top_k = min(8, len(joint_names))
    top_torque_values, top_torque_ids = torch.topk(max_abs_applied_by_joint.detach().cpu(), k=top_k)
    print("[INFO]: Top joints by max applied torque:", flush=True)
    for value, joint_id in zip(top_torque_values.tolist(), top_torque_ids.tolist()):
        print(
            f"  - {joint_names[joint_id]}: applied={value:.4f} Nm, "
            f"computed={float(max_abs_computed_by_joint[joint_id].item()):.4f} Nm, "
            f"max_error={float(max_abs_error_by_joint[joint_id].item()):.4f} rad",
            flush=True,
        )
    if diagnostics_file is not None:
        diagnostics_file.close()
        print(f"[INFO]: Wrote diagnostics CSV: {args_cli.diagnostics_csv}", flush=True)
    if args_cli.screenshot_dir is not None:
        capture_screenshot(args_cli.screenshot_dir / "hexapod_after_landing.png")

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
