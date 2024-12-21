# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from . import observations as obs
from .commands import TargetPosCommand
from omni.isaac.core.utils.torch.rotations import normalize_angle

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()

def off_track(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """penalty for going off track."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_vel_w[:, 1])

def heading_forward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for heading forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    heading_vec = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    return heading_vec[:,0]

def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)

class joint_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)

def pole_target_tracking(
    env: ManagerBasedRLEnv, object_name: str, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for tracking pole's target position."""
    obj: RigidObject = env.scene[object_name]
    return torch.exp(-(obj.data.root_pos_w - env.scene.env_origins - torch.tensor(target_pos, device=env.device)).norm(dim=-1))

def pole_moving(
    env: ManagerBasedRLEnv, object_name: str, target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for tracking pole's target position."""
    obj: RigidObject = env.scene[object_name]
    return -torch.abs(obj.data.root_vel_w[:, 0]/target_vel - 1.0) + 1.0

def forward_speed_obj(
    env: ManagerBasedRLEnv, object_name: str, target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for going forward."""
    obj: RigidObject = env.scene[object_name]
    result = obj.data.root_vel_w[:, 0] / target_vel
    result[result>1.0] = 1.0
    return result

def object_off_track(
    env: ManagerBasedRLEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """penalty for going off track."""
    obj: RigidObject = env.scene[object_name]
    return torch.abs(obj.data.root_vel_w[:, 1])

#####################

def keep_orientation_xy(
    env: ManagerBasedRLEnv, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for keeping close to target orientation on roll and pitch with less weight on pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    eulers_diff[:,1] = eulers_diff[:,1] / 4 # do not need to keep pitch exactly
    return torch.exp(-torch.norm(eulers_diff[:, :2], dim=-1))

class reach_box(ManagerTermBase):
    """reward for getting body part close to the corresponding holding point"""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.dist_left = torch.zeros(env.num_envs, device=env.device)
        self.dist_right = torch.zeros(env.num_envs, device=env.device)
        self.prev_dist_left = torch.zeros_like(self.dist_left)
        self.prev_dist_right = torch.zeros_like(self.dist_right)
        self.BASIS_VEC_Y = torch.tensor((0.0, 1.0, 0.0), device=self.device).repeat(env.num_envs, 1)
        self.dist_torso_xy = torch.zeros(env.num_envs, device=env.device)
        self.prev_dist_torso_xy = torch.zeros(env.num_envs, device=env.device)
        # outside this range consider xy_distance torso rather than hands
        self.dist_range_xy = 1.0

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        obj: RigidObject = self._env.scene["box"]
        pos_left_contact = obj.data.root_pos_w[env_ids] + math_utils.quat_rotate(obj.data.root_quat_w[env_ids], self.BASIS_VEC_Y[env_ids])
        pos_right_contact = obj.data.root_pos_w[env_ids] + math_utils.quat_rotate(obj.data.root_quat_w[env_ids], -self.BASIS_VEC_Y[env_ids])
        to_left_hand = pos_left_contact - asset.data.body_pos_w[env_ids, asset.data.body_names.index("left_hand")]
        to_right_hand = pos_right_contact - asset.data.body_pos_w[env_ids, asset.data.body_names.index("right_hand")]
        to_torso = obj.data.root_pos_w[env_ids] - asset.data.body_pos_w[env_ids, asset.data.body_names.index("torso")]
        self.dist_left[env_ids] = torch.norm(to_left_hand, p=2, dim=-1)
        self.dist_right[env_ids] = torch.norm(to_right_hand, p=2, dim=-1)
        self.dist_torso_xy[env_ids] = torch.norm(to_torso[:, :2], p=2, dim=-1)
        self.prev_dist_left[env_ids] = self.dist_left[env_ids]
        self.prev_dist_right[env_ids] = self.dist_right[env_ids]
        self.prev_dist_torso_xy[env_ids] = self.dist_torso_xy[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        box_size_y: float, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        obj: RigidObject = self._env.scene["box"]
        pos_left_contact = obj.data.root_pos_w + math_utils.quat_rotate(obj.data.root_quat_w, box_size_y/2*self.BASIS_VEC_Y)
        pos_right_contact = obj.data.root_pos_w + math_utils.quat_rotate(obj.data.root_quat_w, -box_size_y/2*self.BASIS_VEC_Y)
        to_left_hand = pos_left_contact - asset.data.body_pos_w[:, asset.data.body_names.index("left_hand")]
        to_right_hand = pos_right_contact - asset.data.body_pos_w[:, asset.data.body_names.index("right_hand")]
        to_torso = obj.data.root_pos_w - asset.data.body_pos_w[:, asset.data.body_names.index("torso")]
        
        self.prev_dist_left = self.dist_left
        self.prev_dist_right = self.dist_right
        self.prev_dist_torso_xy = self.dist_torso_xy
        self.dist_left = torch.norm(to_left_hand, p=2, dim=-1)
        self.dist_right = torch.norm(to_right_hand, p=2, dim=-1)
        self.dist_torso_xy = torch.norm(to_torso[:, :2], p=2, dim=-1)
        # not reward speed higher than 1 m/s
        return torch.where(self.dist_torso_xy>self.dist_range_xy, 
                           torch.clamp((self.prev_dist_torso_xy - self.dist_torso_xy)/env.step_dt, max=1.0),
                           torch.clamp((self.prev_dist_left - self.dist_left) / env.step_dt, max=1.0)*0.5 \
                            + torch.clamp((self.prev_dist_right - self.dist_right) / env.step_dt, max=1.0)*0.5)
    
def hold_box(
    env: ManagerBasedRLEnv, box_size_y: float, dist_range: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward positioning of hands to box holding points proximity"""
    asset: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene["box"]
    pos_lefthand_rel_obj = math_utils.quat_rotate_inverse(obj.data.root_quat_w, 
                                                          asset.data.body_pos_w[:, asset.data.body_names.index("left_hand")] 
                                                          - obj.data.root_pos_w)
    pos_righthand_rel_obj = math_utils.quat_rotate_inverse(obj.data.root_quat_w, 
                                                          asset.data.body_pos_w[:, asset.data.body_names.index("right_hand")] 
                                                          - obj.data.root_pos_w)
    dist_left2boxY = torch.norm(pos_lefthand_rel_obj[:,[0,2]], dim=-1)
    dist_right2boxY = torch.norm(pos_righthand_rel_obj[:, [0,2]], dim=-1)
    rew_left2y = torch.exp(-dist_left2boxY * 4.0)
    rew_right2y = torch.exp(-dist_right2boxY * 4.0)
    dist_left2surface = pos_lefthand_rel_obj[:, 1] - box_size_y/2 - 0.03
    dist_right2surface = -box_size_y/2 - pos_righthand_rel_obj[:, 1]  - 0.03
    rew_left2surface = torch.exp(-dist_left2surface * 2.0)
    rew_left2surface[dist_left2surface<0] = 0
    rew_right2surface = torch.exp(-dist_right2surface * 2.0)
    rew_right2surface[dist_right2surface<0] = 0
    # print(f"dist_left2boxY: {dist_left2boxY}")
    # print(f"dist_right2boxY: {dist_right2boxY}")
    # print(f"dist_left2surface: {dist_left2surface}")
    # print(f"dist_right2surface: {dist_right2surface}")
    # print(f"rew_left2y: {rew_left2y}")
    # print(f"rew_right2y: {rew_right2y}")
    # print(f"rew_left2surface: {rew_left2surface}")
    # print(f"rew_right2surface: {rew_right2surface}")
    mask_left = torch.ones_like(rew_left2y)
    mask_left[torch.norm(pos_lefthand_rel_obj, dim=-1)>dist_range] = 0
    mask_right = torch.ones_like(rew_right2y)
    mask_right[torch.norm(pos_righthand_rel_obj, dim=-1)>dist_range] = 0

    return mask_left * rew_left2y * rew_left2surface / 2 + mask_right * rew_right2y * rew_right2surface / 2

class box_to_target(ManagerTermBase):
    """reward for getting box close to the target"""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.dist = torch.zeros(env.num_envs, device=env.device)
        self.dist_z = torch.zeros(env.num_envs, device=env.device)
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)
        self.prev_dist_z = torch.zeros(env.num_envs, device=env.device)
        self.command_term: TargetPosCommand = env.command_manager.get_term("box_position")

    def reset(self, env_ids: torch.Tensor):
        obj: RigidObject = self._env.scene["box"]
        goal_position = self.command_term.command
        to_target = goal_position[env_ids] + self._env.scene.env_origins[env_ids] - obj.data.root_pos_w[env_ids]
        self.dist[env_ids] = torch.norm(to_target, p=2, dim=-1)
        self.prev_dist[env_ids] = self.dist[env_ids]
        self.dist_z[env_ids] = torch.abs(obj.data.root_pos_w[env_ids, 2] - goal_position[env_ids, 2])
        self.prev_dist_z[env_ids] = self.dist_z[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        obj: RigidObject = self._env.scene["box"]
        goal_position = self.command_term.command
        to_target = goal_position + self._env.scene.env_origins - obj.data.root_pos_w

        self.prev_dist = self.dist
        self.dist = torch.norm(to_target, p=2, dim=-1)

        # prioritize box positioning in z direction to avoid pushing box
        self.prev_dist_z = self.dist_z
        self.dist_z = torch.abs(obj.data.root_pos_w[:, 2] - goal_position[:, 2])
        rew_z = torch.clamp((self.prev_dist_z - self.dist_z) / env.step_dt, min=-1.0, max=1.0)

        # not reward speed higher than 1 m/s
        # only reward when close in the z axis to avoid pushing box
        # penalty also when far in z axis to avoid robot drawing circles with box to collect reward
        rew_dist = torch.clamp((self.prev_dist - self.dist) / env.step_dt, max=1.0) 
        rew_dist = rew_dist * torch.logical_or(self.dist_z<0.2, rew_dist<0)

        return rew_z + rew_dist

def box_on_target(
    env: ManagerBasedRLEnv, position_success_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward positioning of hands to box holding points proximity"""
    obj: RigidObject = env.scene["box"]
    command_term: TargetPosCommand = env.command_manager.get_term("box_position")
    goal_position = command_term.command
    to_target = goal_position + env.scene.env_origins - obj.data.root_pos_w
    dist = torch.norm(to_target, p=2, dim=-1)
    # result = torch.exp(-dist * 6.0)
    # result[dist>dist_range] = 0
    result = torch.zeros_like(dist)
    result[dist<position_success_threshold] = 60.0
    return result

def feet_contact_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensors: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # sensor history correspond to simulation frequency, not task control frequency
    feet_force = 0.5 * (torch.sum(torch.norm(contact_sensors.data.net_forces_w_history[:,0], dim=-1), dim=-1)
                        + torch.sum(torch.norm(contact_sensors.data.net_forces_w_history[:,1], dim=-1), dim=-1))
    feet_force_change = 0.5 * (torch.sum(torch.norm(contact_sensors.data.net_forces_w_history[:,0] - contact_sensors.data.net_forces_w_history[:,1], dim=-1), dim=-1)
                         + torch.sum(torch.norm(contact_sensors.data.net_forces_w_history[:,1] - contact_sensors.data.net_forces_w_history[:,2], dim=-1), dim=-1))
    # breakpoint()
    # print(f"contact force: {feet_force}")
    # print(f"contact force change: {feet_force_change}")
    return (feet_force + 2*feet_force_change) / 4000.0

def center_support(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ Reward for torso-pelvis center close to feet center in xy plane"""
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_xy = asset.data.body_pos_w[:,asset.data.body_names.index("left_foot")]
    left_foot_xy[:,2] = 0
    right_foot_xy = asset.data.body_pos_w[:,asset.data.body_names.index("right_foot")]
    right_foot_xy[:,2] = 0
    torso_xy = asset.data.body_pos_w[:,asset.data.body_names.index("torso")]
    torso_xy[:,2] = 0
    pelvis_xy = asset.data.body_pos_w[:,asset.data.body_names.index("pelvis")]
    pelvis_xy[:,2] = 0
    pseudo_grav_center_xy = (torso_xy+pelvis_xy) / 2.0
    feet_center_xy = (left_foot_xy + right_foot_xy) / 2.0
    pos_rel_xy = pseudo_grav_center_xy - feet_center_xy
    pos_rel_xy[:,0] = pos_rel_xy[:,0] / 2 # less sensitive in x direction
    dist = torch.norm(pos_rel_xy, dim=-1)
    # dist = torch.norm(torch.linalg.cross(left_foot_xy-right_foot_xy, left_foot_xy-pseudo_grav_center_xy), dim=-1) \
    #         / torch.norm(left_foot_xy-right_foot_xy, dim=-1)
    return torch.exp(-dist * 6)