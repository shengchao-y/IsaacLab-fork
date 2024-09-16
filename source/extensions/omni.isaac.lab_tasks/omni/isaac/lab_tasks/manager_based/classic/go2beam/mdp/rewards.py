# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs
import math

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.core.utils.torch.rotations import normalize_angle


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


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

def forward_speed(
    env: ManagerBasedRLEnv, slope: float,  target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for going forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_along_beam = asset.data.root_vel_w[:,2]*math.sin(slope) + asset.data.root_vel_w[:,0]*math.cos(slope)
    result = vel_along_beam / target_vel
    result[result>1.0] = 1.0
    return result

def forward_speed_feet(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for going forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    FL_foot_x = asset.data.body_vel_w[:,asset.data.body_names.index("FL_foot"),0]
    FR_foot_x = asset.data.body_vel_w[:,asset.data.body_names.index("FR_foot"),0]
    RL_foot_x = asset.data.body_vel_w[:,asset.data.body_names.index("RL_foot"),0]
    RR_foot_x = asset.data.body_vel_w[:,asset.data.body_names.index("RR_foot"),0]
    result = (FL_foot_x+FR_foot_x+RL_foot_x+RR_foot_x) / 2.0
    result[result>1.0] = 1.0
    return result

def noside_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    side_vec = torch.tensor((0.0, 1.0, 0.0), device=asset.data.device).repeat(asset.data.GRAVITY_VEC_W.shape[0], 1)
    side_proj = math_utils.quat_rotate(asset.data.root_quat_w, side_vec)[:,1]
    return (side_proj > threshold).float()

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

def jump_up(
    env: ManagerBasedRLEnv, slope: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """penalty for jumping up."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_perp_beam = asset.data.root_vel_w[:,2]*math.cos(slope) - asset.data.root_vel_w[:,0]*math.sin(slope)
    vel_perp_beam[vel_perp_beam<0] = 0
    return vel_perp_beam

def keep_orientation(
    env: ManagerBasedRLEnv, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for keeping close to target orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    return torch.exp(-torch.norm(eulers_diff, dim=-1)*2)


def feet_cross(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty when the asset's left and right feet cross.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    FL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),1]
    FR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),1]
    RL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),1]
    RR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),1]
    cross_front = FR_foot_y - FL_foot_y + 0.05
    cross_rear = RR_foot_y - RL_foot_y + 0.05
    # cross_front[cross_front<0] = 0
    # cross_rear[cross_rear<0] = 0
    # print(f"feet_cross: {(cross_front>0) | (cross_rear>0)}")
    # breakpoint()
    return ((cross_front>0) | (cross_rear>0)).float()

def feet_under_hip(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward when the asset's feet are under the corresponding hips.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    FL_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),0]
    FR_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),0]
    RL_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),0]
    RR_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),0]
    return torch.exp(-torch.abs(FL_foot_x-asset.data.root_pos_w[:,0]-0.22)) \
        + torch.exp(-torch.abs(FR_foot_x-asset.data.root_pos_w[:,0]-0.22)) \
        + torch.exp(-torch.abs(asset.data.root_pos_w[:,0]-RL_foot_x-0.22)) \
        + torch.exp(-torch.abs(asset.data.root_pos_w[:,0]-RR_foot_x-0.22))
