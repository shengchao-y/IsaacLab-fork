# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.utils.torch.rotations import normalize_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def bad_object_pose(
    env: ManagerBasedEnv, object_name: str, minimum_height: float=0, maximum_height: float=100, min_z_proj: float=0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the object pose is out of proper range.

    Note:
        This is currently only supported for poleonhuman task
    """
    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[object_name]
    obj_up_vec = math_utils.quat_rotate(obj.data.root_quat_w, -obj.data.GRAVITY_VEC_W)
    return torch.logical_or(torch.logical_or(obj.data.root_pos_w[:,2] < minimum_height, obj.data.root_pos_w[:,2] > maximum_height),
                            obj_up_vec[:, 2] < min_z_proj)

def pole0_off_hand(
    env: ManagerBasedEnv, dist_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when right hand of away from pole0.

    Note:
        This is currently only supported for poleonhuman task
    """
    asset: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene["pole0"]
    pole0_buttom_pos = obj.data.root_pos_w + math_utils.quat_rotate(obj.data.root_quat_w, obj.data.GRAVITY_VEC_W)
    pole0_bottom2hand = pole0_buttom_pos - asset.data.body_pos_w[:, asset.data.body_names.index("right_hand"), :]
    return (torch.norm(pole0_bottom2hand[:,:2], dim=-1)>dist_threshold) | (pole0_bottom2hand[:,2]<-dist_threshold)


###################

def bad_orientation_xy(
    env: ManagerBasedEnv, limit_angles_diff: tuple[float, float], target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when robot deviates too much from target orientation on roll and pitch.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    return torch.logical_or(eulers_diff[:,0]>limit_angles_diff[0], eulers_diff[:,1]>limit_angles_diff[1])

def bad_orientation_quat_feet(
    env: ManagerBasedEnv, limit_angle_diff: float, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.
    This is computed by checking the normalized angle eulers' difference.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    quat_left_foot = asset.data.body_quat_w[:,asset.data.body_names.index('left_foot')]
    quat_right_foot = asset.data.body_quat_w[:,asset.data.body_names.index('right_foot')]
    quat_diff_left = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), quat_left_foot)
    quat_diff_right = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), quat_right_foot)
    eulers_diff_left = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff_left), dim=1))
    eulers_diff_right = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff_right), dim=1))
    return (torch.norm(eulers_diff_left, dim=-1) > limit_angle_diff) | (torch.norm(eulers_diff_right, dim=-1) > limit_angle_diff)

def unstable_support(
    env: ManagerBasedEnv, dist_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ Terminate when torso-pelvis center far from feet line"""
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
    dist_center = torch.norm(pseudo_grav_center_xy - feet_center_xy, dim=-1)
    # dist_line = torch.norm(torch.linalg.cross(left_foot_xy-right_foot_xy, left_foot_xy-pseudo_grav_center_xy), dim=-1) \
    #         / torch.norm(left_foot_xy-right_foot_xy, dim=-1)
    # print(f"dist: {dist_center}")
    return dist_center > dist_limit

def box_near_body(
    env: ManagerBasedEnv, dist_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene["box"]
    result = torch.zeros_like(asset.data.root_pos_w[:,0])
    for body_part in set(asset.data.body_names) - set(("left_hand", "right_hand", "left_foot", "right_foot", "left_lower_arm", "right_lower_arm", "left_shin", "right_shin")):
        dist = torch.norm(asset.data.body_pos_w[:,asset.data.body_names.index(body_part)]-obj.data.root_pos_w, p=2, dim=-1)
        result = torch.logical_or(result, dist<dist_limit)
    return result

def thigh_diff(
    env: ManagerBasedEnv, angle_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # Terminate when thighs' angle differ too much
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(math_utils.quat_inv(asset.data.body_quat_w[:,asset.data.body_names.index("left_thigh")]),
                                    asset.data.body_quat_w[:,asset.data.body_names.index("right_thigh")])
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    return torch.norm(eulers_diff, dim=-1) > angle_limit