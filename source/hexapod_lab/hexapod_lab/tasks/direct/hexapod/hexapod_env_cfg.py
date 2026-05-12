# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from hexapod_lab.assets import HEXAPOD_CFG


@configclass
class EventCfg:
    """Startup randomization for the first locomotion baseline."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.9, 0.9),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    """Flat-terrain body-frame velocity tracking for the custom hexapod."""

    episode_length_s = 10.0
    decimation = 4
    action_scale = 0.35
    action_space = 18
    observation_space = 66
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Command ranges in robot body frame. +Y is the robot's forward axis.
    forward_velocity_range = (0.15, 0.75)
    lateral_velocity_range = (0.0, 0.0)
    yaw_velocity_range = (-0.4, 0.4)

    # Reward scales.
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.35
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    action_saturation_reward_scale = -0.08
    stillness_reward_scale = -1.0
    feet_air_time_reward_scale = 0.25
    base_contact_reward_scale = -2.0
    flat_orientation_reward_scale = -4.0
    survival_reward_scale = 0.25
