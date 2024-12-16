# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv as BaseEnv


def base_yaw_roll(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)

def base_eulers(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw, pitch and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), pitch.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = math_utils.quat_rotate(asset.data.root_quat_w, -asset.data.GRAVITY_VEC_W)

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: BaseEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: BaseEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)

def object_pose_rel(env: BaseEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of object to robot in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[object_name]
    asset: Articulation = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, :3] - asset.data.root_pos_w[:, :3]

def object_pose_rel_x(env: BaseEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of object to robot in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    return object_pose_rel(env, object_name)[:, 0].unsqueeze(-1)

def object_lin_vel(env: BaseEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Object linear velocity in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[object_name]
    return obj.data.root_lin_vel_w

def object_ang_vel(env: BaseEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Object angular velocity in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[object_name]
    return obj.data.root_ang_vel_w

def object_quat(env: BaseEnv, object_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Object quaternion in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[object_name]
    return obj.data.root_quat_w

def step_above(env: BaseEnv, height_steps: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Object quaternion in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_height = asset.data.root_pos_w[:, 2:3] # [num_envs, 1]
    step_height_diff = height_steps[-1] - height_steps[-2]
    height_steps_cand = torch.tensor(height_steps, device=env.device).unsqueeze(0).repeat([env.num_envs, 1]) # [num_envs, num_steps]
    height_steps_cand[height_steps_cand<torso_height] = torch.inf
    result = height_steps_cand.min(dim=-1, keepdim=True).values
    if torch.any(result==torch.inf):
        breakpoint()
        # raise ValueError("A robot has already climbed to the top of the ladder.")
    pseudo_angle = (step_height_diff - (result - torso_height)) *2 * torch.pi / step_height_diff
    return torch.cat((torch.sin(pseudo_angle), torch.cos(pseudo_angle)), dim=-1)

def base_pos_x(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0:1]-env.scene.env_origins[:,0:1]

def base_dist_x_ladder(env: BaseEnv, ladder_slope: func, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative root x to ladder in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    ladder_x = ladder_slope(asset.data.root_pos_w[:, 2])
    result = torch.abs(asset.data.root_pos_w[:, 0]-env.scene.env_origins[:,0] - ladder_x)
    return result.unsqueeze(-1)