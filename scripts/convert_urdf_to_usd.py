"""Convert hexapod.urdf to hexapod.usd using Isaac Sim's URDF importer.

This script must be run inside an Isaac Sim Python environment. Typical invocation:

    # On a Brev "Isaac Launchable" box:
    cd ~/Hexapod_test
    /workspace/isaaclab/_isaac_sim/python.sh scripts/convert_urdf_to_usd.py

Or via Isaac Lab's wrapper:

    cd ~/Hexapod_test
    /workspace/isaaclab/isaaclab.sh -p scripts/convert_urdf_to_usd.py

Inputs/outputs (paths are repo-relative):
  - Input:  assets/robots/hexapod/urdf/hexapod.urdf
  - Output: assets/robots/hexapod/hexapod.usd

Run once. The resulting USD is checked into the repo and re-imported by
Isaac Lab's ArticulationCfg in downstream scripts.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets" / "robots" / "hexapod" / "urdf" / "hexapod.urdf"
USD_PATH = REPO_ROOT / "assets" / "robots" / "hexapod" / "hexapod.usd"


def main():
    # Boot a headless Isaac Sim app so URDF importer extensions load
    from isaaclab.app import AppLauncher  # type: ignore

    app = AppLauncher(headless=True).app

    # Try the new Isaac Sim 5.x API first, fall back to omni.importer.urdf for older versions.
    try:
        from isaacsim.asset.importer.urdf import _urdf  # type: ignore
        importer_iface = _urdf.acquire_urdf_interface()
        api = "isaacsim.asset.importer.urdf"
    except ImportError:
        from omni.importer.urdf import _urdf  # type: ignore
        importer_iface = _urdf.acquire_urdf_interface()
        api = "omni.importer.urdf"

    print(f"Using importer API: {api}")
    print(f"Input URDF:  {URDF_PATH}")
    print(f"Output USD:  {USD_PATH}")
    assert URDF_PATH.is_file(), f"URDF not found at {URDF_PATH}"

    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = False         # keep joint hierarchy explicit
    cfg.fix_base = False                    # mobile robot, free-floating base
    cfg.make_default_prim = True
    cfg.import_inertia_tensor = True        # use URDF inertias, don't recompute
    cfg.density = 0.0                        # 0 means "trust the URDF inertia/mass"
    cfg.distance_scale = 1.0                # URDF is in meters already
    cfg.convex_decomp = False                # use raw STL collision (may be slow; revisit)
    cfg.self_collision = False               # let Isaac Lab manage self-collision later
    cfg.create_physics_scene = False         # we'll create our own scene
    cfg.parse_mimic = True

    # Run import
    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    importer_iface.import_robot(str(URDF_PATH), str(USD_PATH), cfg)
    print(f"\nUSD written to {USD_PATH}")
    print("File size:", USD_PATH.stat().st_size, "bytes")

    app.close()


if __name__ == "__main__":
    main()
