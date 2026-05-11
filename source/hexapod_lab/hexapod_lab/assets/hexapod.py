# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the custom 18-DOF hexapod robot.

The design frame is documented in the repository README:
+X is the robot's right side, +Y is forward, and +Z is up. The URDF zero pose has
all legs nearly horizontal, so the runtime initial state below deliberately uses
a folded standing pose.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Paths and joint groups.
##

REPO_ROOT = Path(__file__).resolve().parents[4]
USD_PATH = str(REPO_ROOT / "assets" / "robots" / "hexapod" / "hexapod.usd")

RIGHT_COXA_JOINTS = ["FR_coxa_joint", "MR_coxa_joint", "BR_coxa_joint"]
LEFT_COXA_JOINTS = ["FL_coxa_joint", "ML_coxa_joint", "BL_coxa_joint"]
RIGHT_FEMUR_JOINTS = ["FR_femur_joint", "MR_femur_joint", "BR_femur_joint"]
LEFT_FEMUR_JOINTS = ["FL_femur_joint", "ML_femur_joint", "BL_femur_joint"]
RIGHT_TIBIA_JOINTS = ["FR_tibia_joint", "MR_tibia_joint", "BR_tibia_joint"]
LEFT_TIBIA_JOINTS = ["FL_tibia_joint", "ML_tibia_joint", "BL_tibia_joint"]

FOOT_BODY_NAMES = ["FL_tibia_1", "FR_tibia_1", "ML_tibia_1", "MR_tibia_1", "BL_tibia_1", "BR_tibia_1"]

##
# Initial standing pose.
##

STANDING_JOINT_POS = {
    ".*_coxa_joint": 0.0,
    "FR_femur_joint": 0.55,
    "MR_femur_joint": 0.55,
    "BR_femur_joint": 0.55,
    "FL_femur_joint": -0.55,
    "ML_femur_joint": -0.55,
    "BL_femur_joint": -0.55,
    "FR_tibia_joint": 1.20,
    "MR_tibia_joint": 1.20,
    "BR_tibia_joint": 1.20,
    "FL_tibia_joint": -1.20,
    "ML_tibia_joint": -1.20,
    "BL_tibia_joint": -1.20,
}
"""Nominal spawn pose.

The signs are mirrored because the left and right femur/tibia joint axes are
mirrored in the URDF. The values stay inside the documented limits:
right femur/tibia are positive, left femur/tibia are negative.
"""

##
# Robot configuration.
##

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos=STANDING_JOINT_POS,
    ),
    actuators={
        "coxa": ImplicitActuatorCfg(
            joint_names_expr=[".*_coxa_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "femur": ImplicitActuatorCfg(
            joint_names_expr=[".*_femur_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "tibia": ImplicitActuatorCfg(
            joint_names_expr=[".*_tibia_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Isaac Lab articulation configuration for the custom hexapod."""

NUM_LEGS = 6
NUM_JOINTS = 18
NUM_JOINTS_PER_LEG = 3
