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

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # type: ignore

    print("Using Isaac Lab UrdfConverter")
    print(f"Input URDF:  {URDF_PATH}")
    print(f"Output USD:  {USD_PATH}")
    assert URDF_PATH.is_file(), f"URDF not found at {URDF_PATH}"

    USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_PATH.parent),
        usd_file_name=USD_PATH.name,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        link_density=0.0,
        self_collision=False,
        collision_from_visuals=False,
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=40.0, damping=2.0),
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"\nUSD written to {converter.usd_path}")
    print("File size:", Path(converter.usd_path).stat().st_size, "bytes")

    app.close()


if __name__ == "__main__":
    main()
