# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .hexapod_env_cfg import HexapodFlatEnvCfg


class HexapodEnv(DirectRLEnv):
    """Simple flat-terrain locomotion environment for the custom hexapod."""

    cfg: HexapodFlatEnvCfg

    def __init__(self, cfg: HexapodFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_commands = torch.zeros_like(self._commands)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_body_xy_exp",
                "track_yaw_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "dof_vel_l2",
                "action_rate_l2",
                "action_saturation",
                "forward_stillness",
                "feet_air_time",
                "stuck_foot",
                "stance_foot_drag",
                "max_stance_foot_drag",
                "short_foot_air_time",
                "air_time_imbalance",
                "contact_time_imbalance",
                "episode_foot_duty_imbalance",
                "episode_low_foot_duty",
                "episode_high_foot_duty",
                "coxa_symmetry",
                "coxa_activity_imbalance",
                "coxa_inactivity",
                "coxa_action_saturation",
                "middle_leg_stuck",
                "right_rear_overlap",
                "base_contact",
                "flat_orientation_l2",
                "forward_tilt_l2",
                "survival",
                "episode_body_forward_velocity",
                "episode_world_forward_velocity",
                "episode_world_lateral_velocity_abs",
            ]
        }

        self._body_forward_distance_traveled = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._world_forward_distance_traveled = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._world_lateral_distance_abs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._command_forward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._action_saturation_accum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._joint_velocity_accum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._flat_orientation_accum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._forward_tilt_accum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._total_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._foot_contact_time_accum = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)

        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, self._feet_names = self._contact_sensor.find_bodies(".*_tibia_1")
        self._feet_body_ids, _ = self._robot.find_bodies(self._feet_names, preserve_order=True)
        self._middle_feet_ids, _ = self._contact_sensor.find_bodies("M[LR]_tibia_1")
        self._action_scale = self._make_action_scale()
        self._coxa_ids = self._find_joints(".*_coxa_joint")
        self._middle_action_ids = self._find_joint_names(
            ["ML_coxa_joint", "MR_coxa_joint", "ML_tibia_joint", "MR_tibia_joint"]
        )
        self._right_rear_overlap_ids = self._find_joint_names(["MR_coxa_joint", "BR_coxa_joint"])
        self._coxa_pair_ids = self._find_joint_pairs(
            [("FR_coxa_joint", "FL_coxa_joint"), ("MR_coxa_joint", "ML_coxa_joint"), ("BR_coxa_joint", "BL_coxa_joint")]
        )

    def _make_action_scale(self) -> torch.Tensor:
        joint_names = list(getattr(self._robot.data, "joint_names", []))
        if not joint_names:
            return torch.full_like(self._actions, float(self.cfg.action_scale))
        action_scale = torch.full((self._actions.shape[1],), float(self.cfg.action_scale), device=self.device)
        for joint_id, joint_name in enumerate(joint_names):
            if "coxa" in joint_name:
                action_scale[joint_id] = float(self.cfg.coxa_action_scale)
            elif "femur" in joint_name:
                action_scale[joint_id] = float(self.cfg.femur_action_scale)
            elif "tibia" in joint_name:
                action_scale[joint_id] = float(self.cfg.tibia_action_scale)
        return action_scale.unsqueeze(0)

    def _find_joint_pairs(self, joint_pairs: list[tuple[str, str]]) -> torch.Tensor:
        joint_names = list(getattr(self._robot.data, "joint_names", []))
        joint_name_to_id = {joint_name: joint_id for joint_id, joint_name in enumerate(joint_names)}
        pair_ids = [
            [joint_name_to_id[left_name], joint_name_to_id[right_name]]
            for left_name, right_name in joint_pairs
            if left_name in joint_name_to_id and right_name in joint_name_to_id
        ]
        if not pair_ids:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        return torch.tensor(pair_ids, dtype=torch.long, device=self.device)

    def _find_joints(self, suffix: str) -> torch.Tensor:
        joint_names = list(getattr(self._robot.data, "joint_names", []))
        if suffix.startswith(".*"):
            name_suffix = suffix[2:]
            joint_ids = [joint_id for joint_id, joint_name in enumerate(joint_names) if joint_name.endswith(name_suffix)]
        else:
            joint_ids = [joint_id for joint_id, joint_name in enumerate(joint_names) if joint_name == suffix]
        return torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    def _find_joint_names(self, names: list[str]) -> torch.Tensor:
        joint_names = list(getattr(self._robot.data, "joint_names", []))
        joint_name_to_id = {joint_name: joint_id for joint_id, joint_name in enumerate(joint_names)}
        joint_ids = [joint_name_to_id[name] for name in names if name in joint_name_to_id]
        return torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_command_ramp()
        self._actions = actions.clone()
        self._processed_actions = self._action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _update_command_ramp(self):
        ramp_duration = max(float(self.cfg.command_ramp_duration_s), self.step_dt)
        ramp = torch.clamp(self._total_time / ramp_duration, 0.0, 1.0).unsqueeze(-1)
        self._commands[:] = self._target_commands * ramp

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        desired_lin_vel_b = torch.stack(
            [self._commands[:, 0], self.cfg.forward_axis_sign * self._commands[:, 1]], dim=1
        )
        lin_vel_error = torch.sum(torch.square(desired_lin_vel_b - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        joint_vel = torch.mean(torch.square(self._robot.data.joint_vel), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_saturation = torch.mean(torch.square(torch.clamp(torch.abs(self._actions) - 0.75, min=0.0) / 0.25), dim=1)

        forward_cmd_gate = (self._commands[:, 1] > 0.10).float()
        body_forward_speed = self.cfg.forward_axis_sign * self._robot.data.root_lin_vel_b[:, 1]
        world_forward_speed = self.cfg.forward_axis_sign * self._robot.data.root_lin_vel_w[:, 1]
        world_lateral_speed = self._robot.data.root_lin_vel_w[:, 0]
        forward_stillness = forward_cmd_gate * torch.square(torch.clamp(0.05 - world_forward_speed, min=0.0) / 0.05)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        moving = torch.norm(self._commands[:, :2], dim=1) > 0.1
        air_time = torch.sum((last_air_time - 0.35) * first_contact, dim=1) * moving
        short_foot_air_time = (
            torch.sum(
                torch.square(
                    torch.clamp(self.cfg.min_foot_air_time_s - last_air_time, min=0.0)
                    / self.cfg.min_foot_air_time_s
                )
                * first_contact.float(),
                dim=1,
            )
            * forward_cmd_gate
        )
        stuck_foot = (
            torch.sum(
                torch.square(torch.clamp(current_contact_time - self.cfg.max_foot_contact_time_s, min=0.0)),
                dim=1,
            )
            * forward_cmd_gate
        )
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_contact_forces = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0]
        foot_contact_mask = foot_contact_forces > self.cfg.foot_drag_contact_force_threshold
        foot_speed_xy_w = torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2], dim=-1)
        foot_drag_excess = torch.clamp(foot_speed_xy_w - self.cfg.foot_drag_deadband_mps, min=0.0)
        foot_drag_range = max(self.cfg.foot_drag_bad_threshold_mps - self.cfg.foot_drag_deadband_mps, 1.0e-6)
        stance_foot_drag = (
            torch.mean(torch.square(foot_drag_excess / foot_drag_range) * foot_contact_mask.float(), dim=1)
            * forward_cmd_gate
        )
        max_stance_foot_drag = (
            torch.max(torch.square(foot_drag_excess / foot_drag_range) * foot_contact_mask.float(), dim=1)[0]
            * forward_cmd_gate
        )
        air_time_imbalance = torch.var(current_air_time, dim=1, unbiased=False) * forward_cmd_gate
        contact_time_imbalance = torch.var(current_contact_time, dim=1, unbiased=False) * forward_cmd_gate
        if self._coxa_pair_ids.numel() > 0:
            coxa_pair_pos = self._robot.data.joint_pos[:, self._coxa_pair_ids]
            coxa_symmetry = torch.mean(torch.square(torch.sum(coxa_pair_pos, dim=2)), dim=1) * forward_cmd_gate
        else:
            coxa_symmetry = torch.zeros(self.num_envs, device=self.device)
        if self._coxa_ids.numel() > 0:
            coxa_activity = torch.abs(self._robot.data.joint_vel[:, self._coxa_ids])
            coxa_activity_imbalance = torch.var(coxa_activity, dim=1, unbiased=False) * forward_cmd_gate
            coxa_action_saturation = (
                torch.mean(
                    torch.square(torch.clamp(torch.abs(self._actions[:, self._coxa_ids]) - 0.75, min=0.0) / 0.25),
                    dim=1,
                )
                * forward_cmd_gate
            )
            coxa_inactivity = (
                torch.mean(
                    torch.square(
                        torch.clamp(self.cfg.min_coxa_activity_rad_s - coxa_activity, min=0.0)
                        / self.cfg.min_coxa_activity_rad_s
                    ),
                    dim=1,
                )
                * forward_cmd_gate
            )
        else:
            coxa_activity_imbalance = torch.zeros(self.num_envs, device=self.device)
            coxa_action_saturation = torch.zeros(self.num_envs, device=self.device)
            coxa_inactivity = torch.zeros(self.num_envs, device=self.device)
        if len(self._middle_feet_ids) > 0:
            middle_contact_time = self._contact_sensor.data.current_contact_time[:, self._middle_feet_ids]
            middle_contact_stuck = torch.sum(
                torch.square(torch.clamp(middle_contact_time - self.cfg.max_middle_foot_contact_time_s, min=0.0)),
                dim=1,
            )
        else:
            middle_contact_stuck = torch.zeros(self.num_envs, device=self.device)
        if self._middle_action_ids.numel() > 0:
            middle_action_saturation = torch.mean(
                torch.square(torch.clamp(torch.abs(self._actions[:, self._middle_action_ids]) - 0.80, min=0.0) / 0.20),
                dim=1,
            )
        else:
            middle_action_saturation = torch.zeros(self.num_envs, device=self.device)
        middle_leg_stuck = (middle_contact_stuck + 0.5 * middle_action_saturation) * forward_cmd_gate
        if self._right_rear_overlap_ids.numel() == 2:
            right_rear_coxa_pos = self._robot.data.joint_pos[:, self._right_rear_overlap_ids]
            mr_coxa_pos = right_rear_coxa_pos[:, 0]
            br_coxa_pos = right_rear_coxa_pos[:, 1]
            mr_parked_back = torch.clamp(-mr_coxa_pos - self.cfg.mr_backward_overlap_threshold, min=0.0)
            br_swinging_forward = torch.clamp(br_coxa_pos - self.cfg.br_forward_overlap_threshold, min=0.0)
            right_rear_overlap = mr_parked_back * br_swinging_forward * forward_cmd_gate
        else:
            right_rear_overlap = torch.zeros(self.num_envs, device=self.device)

        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        ).float()
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        forward_tilt = torch.square(self._robot.data.projected_gravity_b[:, 1])
        survival = torch.ones(self.num_envs, device=self.device)

        self._body_forward_distance_traveled += body_forward_speed * self.step_dt
        self._world_forward_distance_traveled += world_forward_speed * self.step_dt
        self._world_lateral_distance_abs += torch.abs(world_lateral_speed) * self.step_dt
        self._command_forward_sum += self._commands[:, 1] * self.step_dt
        self._action_saturation_accum += action_saturation * self.step_dt
        self._joint_velocity_accum += joint_vel * self.step_dt
        self._flat_orientation_accum += flat_orientation * self.step_dt
        self._forward_tilt_accum += forward_tilt * self.step_dt
        self._total_time += self.step_dt
        total_time = torch.clamp(self._total_time, min=self.step_dt)
        episode_body_forward_velocity = self._body_forward_distance_traveled / total_time
        episode_world_forward_velocity = self._world_forward_distance_traveled / total_time
        episode_world_lateral_velocity_abs = self._world_lateral_distance_abs / total_time
        if self._foot_contact_time_accum.shape[1] == foot_contact_mask.shape[1]:
            self._foot_contact_time_accum += foot_contact_mask.float() * self.step_dt
            episode_foot_duty = self._foot_contact_time_accum / total_time.unsqueeze(-1)
            episode_foot_duty_imbalance = torch.var(episode_foot_duty, dim=1, unbiased=False) * forward_cmd_gate
            episode_low_foot_duty = (
                torch.mean(
                    torch.square(
                        torch.clamp(self.cfg.min_episode_foot_duty - episode_foot_duty, min=0.0)
                        / self.cfg.min_episode_foot_duty
                    ),
                    dim=1,
                )
                * forward_cmd_gate
            )
            high_duty_range = max(1.0 - self.cfg.max_episode_foot_duty, 1.0e-6)
            episode_high_foot_duty = (
                torch.mean(
                    torch.square(
                        torch.clamp(episode_foot_duty - self.cfg.max_episode_foot_duty, min=0.0)
                        / high_duty_range
                    ),
                    dim=1,
                )
                * forward_cmd_gate
            )
        else:
            episode_foot_duty_imbalance = torch.zeros(self.num_envs, device=self.device)
            episode_low_foot_duty = torch.zeros(self.num_envs, device=self.device)
            episode_high_foot_duty = torch.zeros(self.num_envs, device=self.device)

        rewards = {
            "track_body_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_yaw_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "dof_vel_l2": joint_vel * self.cfg.joint_velocity_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_saturation": action_saturation * self.cfg.action_saturation_reward_scale * self.step_dt,
            "forward_stillness": forward_stillness * self.cfg.stillness_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "stuck_foot": stuck_foot * self.cfg.stuck_foot_reward_scale * self.step_dt,
            "stance_foot_drag": stance_foot_drag * self.cfg.foot_drag_reward_scale * self.step_dt,
            "max_stance_foot_drag": max_stance_foot_drag * self.cfg.max_foot_drag_reward_scale * self.step_dt,
            "short_foot_air_time": short_foot_air_time * self.cfg.short_foot_air_time_reward_scale * self.step_dt,
            "air_time_imbalance": air_time_imbalance * self.cfg.air_time_imbalance_reward_scale * self.step_dt,
            "contact_time_imbalance": contact_time_imbalance
            * self.cfg.contact_time_imbalance_reward_scale
            * self.step_dt,
            "episode_foot_duty_imbalance": episode_foot_duty_imbalance
            * self.cfg.episode_foot_duty_imbalance_reward_scale
            * self.step_dt,
            "episode_low_foot_duty": episode_low_foot_duty * self.cfg.episode_low_foot_duty_reward_scale * self.step_dt,
            "episode_high_foot_duty": episode_high_foot_duty
            * self.cfg.episode_high_foot_duty_reward_scale
            * self.step_dt,
            "coxa_symmetry": coxa_symmetry * self.cfg.coxa_symmetry_reward_scale * self.step_dt,
            "coxa_activity_imbalance": coxa_activity_imbalance
            * self.cfg.coxa_activity_imbalance_reward_scale
            * self.step_dt,
            "coxa_inactivity": coxa_inactivity * self.cfg.coxa_inactivity_reward_scale * self.step_dt,
            "coxa_action_saturation": coxa_action_saturation
            * self.cfg.coxa_action_saturation_reward_scale
            * self.step_dt,
            "middle_leg_stuck": middle_leg_stuck * self.cfg.middle_leg_stuck_reward_scale * self.step_dt,
            "right_rear_overlap": right_rear_overlap * self.cfg.right_rear_overlap_reward_scale * self.step_dt,
            "base_contact": base_contact * self.cfg.base_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "forward_tilt_l2": forward_tilt * self.cfg.forward_tilt_reward_scale * self.step_dt,
            "survival": survival * self.cfg.survival_reward_scale * self.step_dt,
            "episode_body_forward_velocity": episode_body_forward_velocity * self.step_dt,
            "episode_world_forward_velocity": episode_world_forward_velocity * self.step_dt,
            "episode_world_lateral_velocity_abs": episode_world_lateral_velocity_abs
            * self.cfg.world_lateral_velocity_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._target_commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.lateral_velocity_range
        )
        self._target_commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.forward_velocity_range
        )
        self._target_commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.yaw_velocity_range
        )
        self._commands[env_ids] = 0.0
        self._body_forward_distance_traveled[env_ids] = 0.0
        self._world_forward_distance_traveled[env_ids] = 0.0
        self._world_lateral_distance_abs[env_ids] = 0.0
        self._command_forward_sum[env_ids] = 0.0
        self._action_saturation_accum[env_ids] = 0.0
        self._joint_velocity_accum[env_ids] = 0.0
        self._flat_orientation_accum[env_ids] = 0.0
        self._forward_tilt_accum[env_ids] = 0.0
        self._total_time[env_ids] = 0.0
        self._foot_contact_time_accum[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras
