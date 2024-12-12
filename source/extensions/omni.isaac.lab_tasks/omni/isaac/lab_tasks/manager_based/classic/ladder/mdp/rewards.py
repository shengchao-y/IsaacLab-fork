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
from omni.isaac.core.utils.torch.rotations import normalize_angle

from . import observations as obs
import numpy as np

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
        # # compute vector to target
        # target_pos = torch.tensor(target_pos, device=env.device)
        # to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        # to_target_pos[:, 2] = 0.0
        # # update history buffer and compute new potential
        # self.prev_potentials[:] = self.potentials[:]
        # self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        # return self.potentials - self.prev_potentials
        return asset.data.root_vel_w[:, 0]


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
    env: ManagerBasedRLEnv, target_pos, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for tracking pole's target position."""
    obj: Articulation = env.scene["pole"]
    assert False, "this is wrong, the env position env.scene.env_origins should be removed from every env."
    return torch.exp(-(obj.data.root_pos_w - torch.tensor(target_pos, device=obj.data.root_pos_w.device)).norm(dim=-1))

def body_part_position_x(
     env: ManagerBasedRLEnv, ladder_slope: func, body_part: str, distance_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for stepping around ladder x position."""
    asset: Articulation = env.scene[asset_cfg.name]
    ind_body = asset.data.body_names.index(body_part)
    ladder_x = ladder_slope(asset.data.body_pos_w[:,ind_body,2])
    result = torch.abs(asset.data.body_pos_w[:,ind_body,0] - env.scene.env_origins[:,0] - ladder_x)
    result = -2.0 * result + 1.0 + 2.0*distance_limit
    result[result>1.0] = 1.0
    return result
    
def move_up_vel(
    env: ManagerBasedRLEnv, target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving up at defined velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    # vel_diff = asset.data.root_vel_w[:, 2] - target_vel
    # robot learns to shake at very high frequency to give every sample step a positive v value
    # result = asset.data.root_vel_w[:, 2] / target_vel
    vel_z_substitute = (asset.data.root_pos_w[:, 2] - env.root_pos_w_z) / env.step_dt
    # robot learns to do pull-up to gather the reward difference between positive and negative v
    # result = torch.exp(vel_z_substitute * np.log(2.0) / target_vel) - 1.0
    result = vel_z_substitute  / target_vel
    result[result>1.0] = 1.0
    # if(torch.any(env.reset_buf==True)):
    #     breakpoint()
    # print(f"v_z: {asset.data.root_vel_w[:, 2].item()}")
    # print(f"z: {asset.data.root_pos_w[:, 2].item()}")
    return result

def limbs_up_vel(
    env: ManagerBasedRLEnv, target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving hands and feet up at defined velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    # limbs_end = {"left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None}
    hands_inds = [asset.data.body_names.index(body_part) for body_part in ["left_hand", "right_hand"]]
    feet_inds = [asset.data.body_names.index(body_part) for body_part in ["left_foot", "right_foot"]]
    torso_ind = asset.data.body_names.index("torso")
    # ind_left_hand = asset.data.body_names.index("left_hand")
    # ind_right_hand = asset.data.body_names.index("right_hand")
    # ind_left_foot = asset.data.body_names.index("left_foot")
    # ind_right_foot = asset.data.body_names.index("right_foot")

    hands_vel_z_max = torch.cat([asset.data.body_lin_vel_w[:,ind_body_part,2:3] 
                for ind_body_part in hands_inds], dim=-1).max(dim=-1).values
    feet_vel_z_max = torch.cat([asset.data.body_lin_vel_w[:,ind_body_part,2:3] 
            for ind_body_part in feet_inds], dim=-1).max(dim=-1).values
    torso_vel_z = asset.data.body_lin_vel_w[:,torso_ind,2]
    hands_vel_z_min = torch.cat([asset.data.body_lin_vel_w[:,ind_body_part,2:3] 
                for ind_body_part in hands_inds], dim=-1).min(dim=-1).values
    feet_vel_z_min = torch.cat([asset.data.body_lin_vel_w[:,ind_body_part,2:3] 
            for ind_body_part in feet_inds], dim=-1).min(dim=-1).values
    penalty = (hands_vel_z_min + feet_vel_z_min) / 2.0  / target_vel
    penalty[penalty>0.0] = 0
    rew = (hands_vel_z_max + feet_vel_z_max + torso_vel_z) / 3.0  / target_vel
    rew[rew>1.0] = 1.0
    return rew + penalty

def keep_orientation(
    env: ManagerBasedRLEnv, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for keeping close to target orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    eulers_diff[:,1] = eulers_diff[:,1] * 2 # do not need to keep pitch exactly
    return torch.exp(-torch.norm(eulers_diff, dim=-1))
