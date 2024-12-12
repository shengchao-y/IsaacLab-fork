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


def body_part_off_ladder(
     env: ManagerBasedRLEnv, ladder_slope: func, body_part: str, distance_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for stepping around ladder x position."""
    asset: Articulation = env.scene[asset_cfg.name]
    ind_body = asset.data.body_names.index(body_part)
    ladder_x = ladder_slope(asset.data.body_pos_w[:,ind_body,2])
    distance = torch.abs(asset.data.body_pos_w[:,ind_body,0] - env.scene.env_origins[:,0] - ladder_x)
    return distance > distance_limit

def bad_orientation_quat_feet(
    env: ManagerBasedRLEnv, limit_angle_diff: float, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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

def bad_orientation_quat(
    env: ManagerBasedRLEnv, limit_angle_diff: float, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.
    This is computed by checking the normalized angle eulers' difference.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    return torch.norm(eulers_diff, dim=-1) > limit_angle_diff
