"""Convert assembly.xacro -> assembly.urdf with proper axes derived from
the ACTUAL kinematic chain (parent->child segment directions at q=0), not
from radial position from body center.

Axes:
- Coxa joints: (0, 0, +1) for right legs (FR/MR/BR), (0, 0, -1) for left
  legs (FL/ML/BL). This makes a positive coxa command swing both sides'
  legs in the same body-frame direction (forward).
- Femur joints: perpendicular (CCW) to (coxa-joint-world -> femur-joint-world)
  in the horizontal plane.
- Tibia joints: perpendicular (CCW) to (femur-joint-world -> tibia-joint-world)
  in the horizontal plane.

Bilateral symmetry is preserved automatically because mirrored-position legs
get mirrored segment directions, hence mirrored axes.
"""
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = str(REPO_ROOT / "assets" / "robots" / "hexapod" / "urdf" / "assembly.xacro")
OUTPUT = str(REPO_ROOT / "assets" / "robots" / "hexapod" / "urdf" / "hexapod.urdf")

MASS_MAP = {"coxa": 0.2, "femur": 1.0, "tibia": 0.7}
INERTIA_DIAG = {"coxa": 1.0e-4, "femur": 1.0e-3, "tibia": 7.0e-4}

# User-provided femur/tibia rotation-axis angles, in degrees.
# theta measured CCW about +Z from -Y (so right-side axes point in -Y hemisphere).
# Positive theta = axis tilts toward +X (forward side of body).
# Left side is mirrored: axis_left = (-axis_right_x, axis_right_y, 0).
# axis_right = (sin theta, -cos theta, 0).
THETA_DEG = {
    "FR": +28.583,
    "MR": +1.417,
    "BR": -26.417,
    # left side filled in below as mirror of right
}

# Joint limits in radians.
# Coxa: +/- 45 degrees, symmetric.
# Femur: 0 to +70 deg on RIGHT, -70 to 0 deg on LEFT (asymmetric due to mirrored axes).
# Tibia: -140 to 0 deg on RIGHT, 0 to +140 deg on LEFT.
import math as _math
DEG = _math.pi / 180.0
LIMITS = {
    "coxa":  {"R": (-45 * DEG, +45 * DEG),  "L": (-45 * DEG, +45 * DEG)},
    "femur": {"R": (0.0,       +70 * DEG),  "L": (-70 * DEG, 0.0)},
    "tibia": {"R": (-140 * DEG, 0.0),       "L": (0.0, +140 * DEG)},
}

# Material color (gray)
MATERIAL_NAME = "gray"
MATERIAL_RGBA = "0.7 0.7 0.7 1.0"

# Read and strip xacro
with open(INPUT, "r") as f:
    text = f.read()
text = re.sub(r"<xacro:[^>]*>", "", text)
text = re.sub(r"</xacro:[^>]*>", "", text)
text = re.sub(r'\s*xmlns:xacro="[^"]+"', "", text)
# Rewrite mesh paths from ROS-style package URIs to relative paths.
# Output URDF lives at <repo>/urdf/urdf/assembly.urdf, meshes at <repo>/urdf/meshes/.
# Relative path from URDF dir to meshes dir is "../meshes/".
text = text.replace("package://assembly_description/meshes/", "../meshes/")
# Rename material references from "silver" (undefined) to "gray" (we'll define inline).
text = text.replace('name="silver"', f'name="{MATERIAL_NAME}"')
root = ET.fromstring(text)

# Rename robot
root.set("name", "hexapod")

# Add inline material definition at the top of the robot element so visualizers/Isaac
# can resolve <material name="gray"/> references later in the file.
material_el = ET.Element("material", name=MATERIAL_NAME)
ET.SubElement(material_el, "color", rgba=MATERIAL_RGBA)
root.insert(0, material_el)

# Pass 1: index joints by child link, gather origin xyz
joint_by_child = {}
for j in root.iter("joint"):
    child = j.find("child")
    if child is None:
        continue
    o = j.find("origin")
    xyz = [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
    joint_by_child[child.get("link")] = {"elem": j, "xyz": xyz,
                                          "parent": j.find("parent").get("link")}


def world_pos(link):
    """Forward-kinematic position of a link's frame origin at q=0.
    Assumes joint origin rpy=0 everywhere (verified for this URDF)."""
    if link == "base_link":
        return (0.0, 0.0, 0.0)
    info = joint_by_child[link]
    px, py, pz = world_pos(info["parent"])
    return (px + info["xyz"][0], py + info["xyz"][1], pz + info["xyz"][2])


def perp_horizontal_ccw(dx, dy):
    """Unit vector perpendicular to (dx, dy) in horizontal plane,
    rotated 90 CCW (so (1,0) -> (0,1) and (0,1) -> (-1,0))."""
    n = math.sqrt(dx * dx + dy * dy)
    if n < 1e-6:
        return (1.0, 0.0, 0.0)
    return (round(-dy / n, 4), round(dx / n, 4), 0.0)


joint_summary = []
for j in root.iter("joint"):
    if j.get("type") != "fixed":
        continue
    child = j.find("child")
    parent = j.find("parent")
    if child is None or parent is None:
        continue
    child_link = child.get("link", "")
    m = re.match(r"^([FMB][LR])_(coxa|femur|tibia)_\d+$", child_link)
    if not m:
        continue
    leg_id, link_type = m.group(1), m.group(2)
    side = leg_id[1]  # 'L' or 'R'

    if link_type == "coxa":
        # +Z for right side, -Z for left side
        ax = (0.0, 0.0, 1.0 if side == "R" else -1.0)
    else:
        # User-specified angles per leg pair (femur and tibia share same axis).
        # Lookup using leg position (front/middle/back), with left-side mirrored.
        pos = leg_id[0]  # F, M, or B
        right_leg_id = f"{pos}R"
        theta_deg = THETA_DEG.get(right_leg_id, 0.0)
        theta_rad = math.radians(theta_deg)
        sx = math.sin(theta_rad)
        cy = -math.cos(theta_rad)
        if side == "R":
            ax = (round(sx, 4), round(cy, 4), 0.0)
        else:  # mirror across YZ plane
            ax = (round(-sx, 4), round(cy, 4), 0.0)

    # Apply
    j.set("type", "revolute")
    j.set("name", f"{leg_id}_{link_type}_joint")
    # Remove existing axis/limit/dynamics if present (defensive)
    for tag in ("axis", "limit", "dynamics"):
        existing = j.find(tag)
        if existing is not None:
            j.remove(existing)
    axis_el = ET.SubElement(j, "axis")
    axis_el.set("xyz", f"{ax[0]} {ax[1]} {ax[2]}")
    lo, hi = LIMITS[link_type][side]
    limit_el = ET.SubElement(j, "limit")
    limit_el.set("lower", f"{lo:.6f}")
    limit_el.set("upper", f"{hi:.6f}")
    limit_el.set("effort", "10.0")
    limit_el.set("velocity", "10.0")
    dyn_el = ET.SubElement(j, "dynamics")
    dyn_el.set("damping", "0.1")
    dyn_el.set("friction", "0.01")
    joint_summary.append((f"{leg_id}_{link_type}_joint", parent.get("link"), child_link, ax))

# Mass and inertia (same as before)
for link in root.iter("link"):
    name = link.get("name", "")
    if name == "base_link":
        continue
    m = re.match(r"^([FMB][LR])_(coxa|femur|tibia)_\d+$", name)
    if not m:
        continue
    link_type = m.group(2)
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="0")
        ET.SubElement(inertial, "inertia", ixx="0", iyy="0", izz="0", ixy="0", iyz="0", ixz="0")
    mass_el = inertial.find("mass")
    if mass_el is not None:
        mass_el.set("value", str(MASS_MAP[link_type]))
    inertia_el = inertial.find("inertia")
    if inertia_el is not None:
        d = INERTIA_DIAG[link_type]
        inertia_el.set("ixx", f"{d:.6e}")
        inertia_el.set("iyy", f"{d:.6e}")
        inertia_el.set("izz", f"{d:.6e}")
        inertia_el.set("ixy", "0.0")
        inertia_el.set("iyz", "0.0")
        inertia_el.set("ixz", "0.0")

tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(OUTPUT, xml_declaration=True, encoding="utf-8")

print(f"Wrote: {OUTPUT}\n")
print("Joint axes after fix:")
print(f"{'Joint':25s} {'Axis (x, y, z)':30s}")
for name, parent, child, ax in joint_summary:
    print(f"  {name:25s} ({ax[0]:+.4f}, {ax[1]:+.4f}, {ax[2]:+.4f})")
