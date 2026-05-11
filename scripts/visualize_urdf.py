"""Generic interactive URDF visualizer at q=0.

Usage:
    python3 visualize_any_urdf.py <input.urdf> <output.html>

Shows skeleton (parent->child lines), per-joint local RGB frame triads,
joint rotation axes (yellow for revolute, hidden for fixed), and world frame.

Hover any element for joint/link details. Drag to rotate, scroll to zoom.
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np
import plotly.graph_objects as go


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def visualize(urdf_path, out_html):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}

    joints = []
    for j in root.findall("joint"):
        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            continue
        origin = j.find("origin")
        if origin is None:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        else:
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        axis_el = j.find("axis")
        axis = [0.0, 0.0, 0.0]
        if axis_el is not None:
            axis = [float(x) for x in axis_el.get("xyz", "0 0 0").split()]
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": parent_el.get("link"), "child": child_el.get("link"),
            "xyz": xyz, "rpy": rpy, "axis": axis,
        })

    # FK at q=0
    all_children = {j["child"] for j in joints}
    roots = [n for n in links if n not in all_children]
    if not roots:
        raise RuntimeError("No root link found")
    root_link = roots[0]
    world_T = {root_link: (np.eye(3), np.zeros(3))}
    joint_world = {}
    joints_by_parent = {}
    for j in joints:
        joints_by_parent.setdefault(j["parent"], []).append(j)

    queue = [root_link]
    while queue:
        p = queue.pop(0)
        if p not in world_T:
            continue
        pR, pt = world_T[p]
        for j in joints_by_parent.get(p, []):
            jR = rpy_to_R(j["rpy"])
            jt = np.array(j["xyz"])
            origin_world = pR @ jt + pt
            R_world = pR @ jR
            axis_world = R_world @ np.array(j["axis"])
            joint_world[j["name"]] = (origin_world, R_world, axis_world, j)
            world_T[j["child"]] = (R_world, origin_world)
            queue.append(j["child"])

    # Determine model bounding box to scale axes
    positions = np.array([t for (R, t) in world_T.values()])
    span = float(np.max(positions.max(0) - positions.min(0)))
    triad_len = span * 0.03
    rot_len = span * 0.06
    world_len = span * 0.20

    traces = []

    # Skeleton lines
    for j in joints:
        if j["parent"] not in world_T or j["child"] not in world_T:
            continue
        p_pos = world_T[j["parent"]][1]
        c_pos = world_T[j["child"]][1]
        # Color by joint type
        if j["type"] in ("revolute", "continuous"):
            color = "darkorange"
            width = 6
            dash = "solid"
        else:  # fixed and others
            color = "lightgray"
            width = 2
            dash = "dot"
        traces.append(go.Scatter3d(
            x=[p_pos[0], c_pos[0]], y=[p_pos[1], c_pos[1]], z=[p_pos[2], c_pos[2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertext=f"{j['parent']} -> {j['child']}<br>joint: {j['name']} ({j['type']})",
            hoverinfo="text",
            showlegend=False,
        ))

    # Link origin markers
    for name, (R, t) in world_T.items():
        traces.append(go.Scatter3d(
            x=[t[0]], y=[t[1]], z=[t[2]],
            mode="markers",
            marker=dict(size=2, color="black"),
            hovertext=name,
            hoverinfo="text",
            showlegend=False,
        ))

    # Per-joint local frame triads (RGB) — only for revolute joints to declutter
    for name, (origin, R, axis_world, j_data) in joint_world.items():
        if j_data["type"] not in ("revolute", "continuous"):
            continue
        for axis_idx, color in enumerate(["red", "green", "blue"]):
            d = R[:, axis_idx] * triad_len
            traces.append(go.Scatter3d(
                x=[origin[0], origin[0] + d[0]],
                y=[origin[1], origin[1] + d[1]],
                z=[origin[2], origin[2] + d[2]],
                mode="lines",
                line=dict(color=color, width=2),
                hovertext=f"{name} local {'XYZ'[axis_idx]}",
                hoverinfo="text",
                showlegend=False,
            ))

    # Rotation axes (yellow) for revolute joints
    for name, (origin, R, axis_world, j_data) in joint_world.items():
        if j_data["type"] not in ("revolute", "continuous"):
            continue
        d = np.array(axis_world) * rot_len
        traces.append(go.Scatter3d(
            x=[origin[0], origin[0] + d[0]],
            y=[origin[1], origin[1] + d[1]],
            z=[origin[2], origin[2] + d[2]],
            mode="lines",
            line=dict(color="yellow", width=5),
            hovertext=f"{name} rotation axis: {tuple(round(x, 3) for x in j_data['axis'])}<br>(world: {tuple(round(float(x), 3) for x in axis_world)})",
            hoverinfo="text",
            showlegend=False,
        ))
        traces.append(go.Scatter3d(
            x=[origin[0] + d[0]], y=[origin[1] + d[1]], z=[origin[2] + d[2]],
            mode="markers",
            marker=dict(size=4, color="gold", symbol="diamond"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # World frame at origin
    world_axes = [
        ([world_len, 0, 0], "red", "X"),
        ([0, world_len, 0], "green", "Y"),
        ([0, 0, world_len], "blue", "Z"),
    ]
    for d, color, label in world_axes:
        traces.append(go.Scatter3d(
            x=[0, d[0]], y=[0, d[1]], z=[0, d[2]],
            mode="lines+text",
            line=dict(color=color, width=8),
            text=["", f"world {label}"],
            textposition="top center",
            textfont=dict(size=14, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Legend
    legend = [
        ("Revolute joint chain", "darkorange", "solid"),
        ("Fixed joint chain", "lightgray", "dot"),
        ("Joint local X (revolute only)", "red", "solid"),
        ("Joint local Y", "green", "solid"),
        ("Joint local Z", "blue", "solid"),
        ("Rotation axis", "yellow", "solid"),
    ]
    for label, color, dash in legend:
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color=color, width=6, dash=dash),
            name=label,
            showlegend=True,
        ))

    title = f"{root.get('name')} URDF at q=0  ({len(links)} links, {sum(1 for j in joints if j['type'] in ('revolute', 'continuous'))} revolute, {sum(1 for j in joints if j['type'] == 'fixed')} fixed)"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="#fafafa",
        ),
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.85)"),
        width=1200, height=800,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved: {out_html}")
    print(f"  Revolute joints: {sum(1 for j in joints if j['type'] in ('revolute', 'continuous'))}")
    print(f"  Fixed joints: {sum(1 for j in joints if j['type'] == 'fixed')}")
    print(f"  Links: {len(links)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 visualize_any_urdf.py <input.urdf> <output.html>")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2])
