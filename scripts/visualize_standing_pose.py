"""Render the Isaac Lab standing pose as an interactive wireframe HTML.

This script reads STANDING_JOINT_POS from:
    source/hexapod_lab/hexapod_lab/assets/hexapod.py

It then applies those revolute joint angles to the URDF kinematic tree and
draws links, joint frames, rotation axes, and foot/body labels.
"""

from __future__ import annotations

import ast
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "assets/robots/hexapod/urdf/hexapod.urdf"
ASSET_CFG_PATH = REPO_ROOT / "source/hexapod_lab/hexapod_lab/assets/hexapod.py"
DEFAULT_OUT = REPO_ROOT / "hexapod_standing_pose.html"


def rpy_to_r(rpy: list[float]) -> np.ndarray:
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


def joint_angle(joint_name: str, joint_pos: dict[str, float]) -> float:
    if joint_name in joint_pos:
        return float(joint_pos[joint_name])
    if joint_name.endswith("_coxa_joint") and ".*_coxa_joint" in joint_pos:
        return float(joint_pos[".*_coxa_joint"])
    return 0.0


def parse_urdf() -> tuple[dict[str, ET.Element], list[dict]]:
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
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent_el.get("link"),
                "child": child_el.get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
        )
    return links, joints


def render(out_html: Path) -> None:
    links, joints = parse_urdf()
    joint_pos = load_standing_joint_pos()

    all_children = {joint["child"] for joint in joints}
    root_link = next(link for link in links if link not in all_children)
    world_t: dict[str, tuple[np.ndarray, np.ndarray]] = {root_link: (np.eye(3), np.zeros(3))}
    joint_world = {}

    joints_by_parent: dict[str, list[dict]] = {}
    for joint in joints:
        joints_by_parent.setdefault(joint["parent"], []).append(joint)

    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        parent_r, parent_t = world_t[parent]
        for joint in joints_by_parent.get(parent, []):
            origin_r = rpy_to_r(joint["rpy"])
            origin_t = np.array(joint["xyz"])
            joint_origin_w = parent_r @ origin_t + parent_t
            joint_r_w = parent_r @ origin_r
            axis_local = np.array(joint["axis"], dtype=float)
            axis_world = joint_r_w @ axis_local
            q = joint_angle(joint["name"], joint_pos) if joint["type"] in ("revolute", "continuous") else 0.0
            child_r_w = joint_r_w @ axis_angle_to_r(axis_local, q)
            world_t[joint["child"]] = (child_r_w, joint_origin_w)
            joint_world[joint["name"]] = (joint_origin_w, joint_r_w, axis_world, q, joint)
            queue.append(joint["child"])

    positions = np.array([t for _, t in world_t.values()])
    foot_links = {"FL_tibia_1", "FR_tibia_1", "ML_tibia_1", "MR_tibia_1", "BL_tibia_1", "BR_tibia_1"}

    lines = []
    for joint in joints:
        parent_pos = world_t[joint["parent"]][1]
        child_pos = world_t[joint["child"]][1]
        lines.append(
            {
                "parent": joint["parent"],
                "child": joint["child"],
                "joint": joint["name"],
                "type": joint["type"],
                "p": parent_pos,
                "c": child_pos,
                "q": joint_angle(joint["name"], joint_pos),
            }
        )

    points = {name: t for name, (_, t) in world_t.items()}

    def projection_svg(title: str, axis_a: int, axis_b: int, label_a: str, label_b: str) -> str:
        width, height, pad = 760, 520, 54
        all_xy = np.array([[p[axis_a], p[axis_b]] for p in points.values()])
        min_xy = all_xy.min(axis=0)
        max_xy = all_xy.max(axis=0)
        size_xy = np.maximum(max_xy - min_xy, 0.05)
        scale = min((width - 2 * pad) / size_xy[0], (height - 2 * pad) / size_xy[1])

        def to_px(vec: np.ndarray) -> tuple[float, float]:
            x = pad + (vec[axis_a] - min_xy[0]) * scale
            y = height - pad - (vec[axis_b] - min_xy[1]) * scale
            return x, y

        parts = [
            f'<h2>{title}</h2>',
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#fafafa" stroke="#ddd"/>',
        ]
        for line in lines:
            x1, y1 = to_px(line["p"])
            x2, y2 = to_px(line["c"])
            color = "#d97706" if line["type"] == "revolute" else "#aaa"
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="5" stroke-linecap="round">'
                f'<title>{line["parent"]} -> {line["child"]}; {line["joint"]}; q={line["q"]:.3f}</title></line>'
            )
        for name, point in points.items():
            x, y = to_px(point)
            is_foot = name in foot_links
            is_base = name == "base_link"
            radius = 6 if is_base else 5 if is_foot else 2.5
            color = "#111" if is_base else "#2563eb" if is_foot else "#444"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"><title>{name}</title></circle>')
            if is_foot or is_base:
                parts.append(
                    f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="12" fill="{color}" '
                    f'font-family="monospace">{name}</text>'
                )
        parts.extend(
            [
                f'<text x="{pad}" y="{height - 16}" font-size="13" fill="#b91c1c" font-family="monospace">{label_a}</text>',
                f'<text x="{width - 160}" y="{pad}" font-size="13" fill="#15803d" font-family="monospace">{label_b}</text>',
                "</svg>",
            ]
        )
        return "\n".join(parts)

    pose_table = "\n".join(
        f"<tr><td>{name}</td><td>{value:.3f}</td></tr>" for name, value in sorted(joint_pos.items())
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hexapod Standing Pose</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    h1 {{ margin-bottom: 0; }}
    h2 {{ margin: 18px 0 8px; }}
    p {{ max-width: 900px; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 18px; }}
    td, th {{ border: 1px solid #ddd; padding: 4px 10px; font-family: monospace; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Hexapod Isaac Lab Standing Pose</h1>
  <p>This is the current spawn pose from <code>STANDING_JOINT_POS</code>. Blue labels are the six tibia/foot links.
  Black is <code>base_link</code>. Orange lines are revolute-joint link chains.</p>
  <div class="grid">
    {projection_svg("Top View: X right vs Y forward", 0, 1, "+X right", "+Y forward")}
    {projection_svg("Side View: Y forward vs Z up", 1, 2, "+Y forward", "+Z up")}
    {projection_svg("Front View: X right vs Z up", 0, 2, "+X right", "+Z up")}
  </div>
  <h2>Standing Joint Angles</h2>
  <table><tr><th>Joint pattern/name</th><th>rad</th></tr>{pose_table}</table>
</body>
</html>
"""
    out_html.write_text(html)
    print(f"Saved: {out_html}")
    print("Standing joint pose:")
    for name, value in sorted(joint_pos.items()):
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    render(output)
