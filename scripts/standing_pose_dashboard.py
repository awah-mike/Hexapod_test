"""Interactive Python dashboard for tuning the hexapod standing pose.

The front-right leg is the reference leg:

* Femur slider value is applied to FR/MR/BR femur joints.
* The negative of that value is applied to FL/ML/BL femur joints.
* Tibia slider value is applied to FR/MR/BR tibia joints.
* The negative of that value is applied to FL/ML/BL tibia joints.
* Coxa joints stay fixed at the current standing-pose value.

Run locally from the repo root:

    python -m pip install numpy matplotlib
    python scripts/standing_pose_dashboard.py

This script has no Isaac Sim / Isaac Lab dependency. It only needs Python,
NumPy, and Matplotlib.
"""

from __future__ import annotations

import ast
import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "assets/robots/hexapod/urdf/hexapod.urdf"
ASSET_CFG_PATH = REPO_ROOT / "source/hexapod_lab/hexapod_lab/assets/hexapod.py"

RIGHT_FEMUR = ["FR_femur_joint", "MR_femur_joint", "BR_femur_joint"]
LEFT_FEMUR = ["FL_femur_joint", "ML_femur_joint", "BL_femur_joint"]
RIGHT_TIBIA = ["FR_tibia_joint", "MR_tibia_joint", "BR_tibia_joint"]
LEFT_TIBIA = ["FL_tibia_joint", "ML_tibia_joint", "BL_tibia_joint"]
FOOT_LINKS = {"FL_tibia_1", "FR_tibia_1", "ML_tibia_1", "MR_tibia_1", "BL_tibia_1", "BR_tibia_1"}


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def rpy_to_r(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_to_r(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def load_standing_joint_pos() -> dict[str, float]:
    tree = ast.parse(ASSET_CFG_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STANDING_JOINT_POS":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find STANDING_JOINT_POS in {ASSET_CFG_PATH}")


def parse_urdf() -> tuple[dict[str, ET.Element], list[Joint]]:
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = []
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0] if origin is None else [float(x) for x in origin.get("xyz", "0 0 0").split()]
        rpy = [0.0, 0.0, 0.0] if origin is None else [float(x) for x in origin.get("rpy", "0 0 0").split()]
        axis_el = joint.find("axis")
        axis = [0.0, 0.0, 0.0] if axis_el is None else [float(x) for x in axis_el.get("xyz", "0 0 0").split()]
        joints.append(
            Joint(
                name=joint.get("name"),
                type=joint.get("type"),
                parent=parent_el.get("link"),
                child=child_el.get("link"),
                xyz=np.array(xyz, dtype=float),
                rpy=np.array(rpy, dtype=float),
                axis=np.array(axis, dtype=float),
            )
        )
    return links, joints


def mirrored_pose(reference_femur: float, reference_tibia: float, coxa: float) -> dict[str, float]:
    joint_pos = {".*_coxa_joint": coxa}
    for name in RIGHT_FEMUR:
        joint_pos[name] = reference_femur
    for name in LEFT_FEMUR:
        joint_pos[name] = -reference_femur
    for name in RIGHT_TIBIA:
        joint_pos[name] = reference_tibia
    for name in LEFT_TIBIA:
        joint_pos[name] = -reference_tibia
    return joint_pos


def joint_angle(joint_name: str, joint_pos: dict[str, float]) -> float:
    if joint_name in joint_pos:
        return float(joint_pos[joint_name])
    if joint_name.endswith("_coxa_joint"):
        return float(joint_pos.get(".*_coxa_joint", 0.0))
    return 0.0


def forward_kinematics(links: dict[str, ET.Element], joints: list[Joint], joint_pos: dict[str, float]):
    all_children = {joint.child for joint in joints}
    root_link = next(link for link in links if link not in all_children)
    world_t: dict[str, tuple[np.ndarray, np.ndarray]] = {root_link: (np.eye(3), np.zeros(3))}

    joints_by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        joints_by_parent.setdefault(joint.parent, []).append(joint)

    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        parent_r, parent_t = world_t[parent]
        for joint in joints_by_parent.get(parent, []):
            origin_r = rpy_to_r(joint.rpy)
            joint_origin_w = parent_r @ joint.xyz + parent_t
            joint_r_w = parent_r @ origin_r
            q = joint_angle(joint.name, joint_pos) if joint.type in ("revolute", "continuous") else 0.0
            child_r_w = joint_r_w @ axis_angle_to_r(joint.axis, q)
            world_t[joint.child] = (child_r_w, joint_origin_w)
            queue.append(joint.child)
    return world_t


def set_axes_equal(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.45, center[2] + radius * 1.05)


class StandingPoseDashboard:
    def __init__(self, links: dict[str, ET.Element], joints: list[Joint], initial_femur: float, initial_tibia: float):
        self.links = links
        self.joints = joints
        self.coxa = float(load_standing_joint_pos().get(".*_coxa_joint", 0.0))
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.24)

        femur_ax = self.fig.add_axes([0.18, 0.12, 0.62, 0.035])
        tibia_ax = self.fig.add_axes([0.18, 0.07, 0.62, 0.035])
        reset_ax = self.fig.add_axes([0.83, 0.07, 0.10, 0.085])

        self.femur_slider = Slider(
            femur_ax,
            "FR femur rad",
            valmin=0.0,
            valmax=1.221730,
            valinit=initial_femur,
            valstep=0.005,
        )
        self.tibia_slider = Slider(
            tibia_ax,
            "FR tibia rad",
            valmin=0.0,
            valmax=2.443461,
            valinit=initial_tibia,
            valstep=0.005,
        )
        self.reset_button = Button(reset_ax, "Reset")

        self.femur_initial = initial_femur
        self.tibia_initial = initial_tibia
        self.femur_slider.on_changed(lambda _: self.draw())
        self.tibia_slider.on_changed(lambda _: self.draw())
        self.reset_button.on_clicked(self.reset)
        self.draw()

    def reset(self, _event) -> None:
        self.femur_slider.reset()
        self.tibia_slider.reset()

    def draw(self) -> None:
        femur = float(self.femur_slider.val)
        tibia = float(self.tibia_slider.val)
        pose = mirrored_pose(femur, tibia, self.coxa)
        world_t = forward_kinematics(self.links, self.joints, pose)
        points = {name: t for name, (_, t) in world_t.items()}
        positions = np.array(list(points.values()))

        self.ax.clear()
        self.ax.set_title(
            "Hexapod standing-pose dashboard\n"
            f"Apply: right femur/tibia = +({femur:.3f}, {tibia:.3f}), "
            f"left femur/tibia = -({femur:.3f}, {tibia:.3f}), coxa = {self.coxa:.3f}"
        )
        self.ax.set_xlabel("+X right")
        self.ax.set_ylabel("+Y forward")
        self.ax.set_zlabel("+Z up")

        for joint in self.joints:
            p = points[joint.parent]
            c = points[joint.child]
            color = "#d97706" if joint.type == "revolute" else "#a3a3a3"
            width = 2.2 if joint.type == "revolute" else 1.0
            self.ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color=color, linewidth=width)

        for name, point in points.items():
            if name == "base_link":
                self.ax.scatter(point[0], point[1], point[2], color="black", s=55)
                self.ax.text(point[0], point[1], point[2], " base_link", color="black")
            elif name in FOOT_LINKS:
                self.ax.scatter(point[0], point[1], point[2], color="#2563eb", s=36)
                self.ax.text(point[0], point[1], point[2], f" {name}", color="#2563eb", fontsize=8)
            else:
                self.ax.scatter(point[0], point[1], point[2], color="#555555", s=10)

        axis_len = 0.30
        self.ax.plot([0, axis_len], [0, 0], [0, 0], color="#dc2626", linewidth=3)
        self.ax.plot([0, 0], [0, axis_len], [0, 0], color="#16a34a", linewidth=3)
        self.ax.plot([0, 0], [0, 0], [0, axis_len], color="#2563eb", linewidth=3)
        self.ax.text(axis_len, 0, 0, "+X right", color="#dc2626")
        self.ax.text(0, axis_len, 0, "+Y forward", color="#16a34a")
        self.ax.text(0, 0, axis_len, "+Z up", color="#2563eb")

        set_axes_equal(self.ax, positions)
        self.ax.view_init(elev=22, azim=-58)
        self.fig.canvas.draw_idle()

        print(
            "\r"
            f"Use these values in HEXAPOD_CFG: right femur={femur:.3f}, left femur={-femur:.3f}, "
            f"right tibia={tibia:.3f}, left tibia={-tibia:.3f}",
            end="",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive standing-pose dashboard for the custom hexapod.")
    parser.add_argument("--femur", type=float, default=None, help="Initial front-right femur reference angle in rad.")
    parser.add_argument("--tibia", type=float, default=None, help="Initial front-right tibia reference angle in rad.")
    args = parser.parse_args()

    standing = load_standing_joint_pos()
    initial_femur = float(args.femur if args.femur is not None else standing.get("FR_femur_joint", 0.55))
    initial_tibia = float(args.tibia if args.tibia is not None else standing.get("FR_tibia_joint", 1.20))
    links, joints = parse_urdf()

    StandingPoseDashboard(links, joints, initial_femur, initial_tibia)
    plt.show()


if __name__ == "__main__":
    main()
