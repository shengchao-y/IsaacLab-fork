# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.core.utils.torch.rotations import normalize_angle
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height

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

def feet_off(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's any foot is off the rope.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("left_foot"),2]
    right_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("right_foot"),2]
    return (left_foot_z<minimum_height) | (right_foot_z<minimum_height)