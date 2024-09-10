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
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height

def fall_off_beam(
    env: ManagerBasedRLEnv, minimum_height: float, slope_func: func, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root beam-relative height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    beam_z = slope_func(asset.data.root_pos_w[:,0] - env.scene.env_origins[:,0])
    return asset.data.root_pos_w[:, 2]-beam_z < minimum_height

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

def feet_cross(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's left and right feet cross.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    FL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),1]
    FR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),1]
    RL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),1]
    RR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),1]
    return (FL_foot_y-FR_foot_y < -0.05) | (RL_foot_y-RR_foot_y < -0.05)

def feet_off(
    env: ManagerBasedRLEnv, slope_func: func, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's any foot is off the beam.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    FL_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),0]
    FR_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),0]
    RL_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),0]
    RR_foot_x = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),0]
    FL_foot_beam_z = slope_func(FL_foot_x - env.scene.env_origins[:,0])
    FR_foot_beam_z = slope_func(FR_foot_x - env.scene.env_origins[:,0])
    RL_foot_beam_z = slope_func(RL_foot_x - env.scene.env_origins[:,0])
    RR_foot_beam_z = slope_func(RR_foot_x - env.scene.env_origins[:,0])
    FL_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),2]
    FR_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),2]
    RL_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),2]
    RR_foot_z = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),2]
    
    # result = (FL_foot_z < FL_foot_beam_z+0.05) | (FR_foot_z < FR_foot_beam_z+0.05) \
    #     | (RL_foot_z < RL_foot_beam_z+0.05) | (RR_foot_z < RR_foot_beam_z+0.05)
    
    FL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FL_foot"),1] - env.scene.env_origins[:,1]
    result_FL = (FL_foot_y < -(FL_foot_z-FL_foot_beam_z)) | (FL_foot_y > FL_foot_z-FL_foot_beam_z)
    FR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("FR_foot"),1] - env.scene.env_origins[:,1]
    result_FR =(FR_foot_y < -(FR_foot_z-FR_foot_beam_z)) | (FR_foot_y > FR_foot_z-FR_foot_beam_z)

    RL_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RL_foot"),1] - env.scene.env_origins[:,1]
    result_RL = (RL_foot_y < -(RL_foot_z-RL_foot_beam_z)) | (RL_foot_y > RL_foot_z-RL_foot_beam_z)
    RR_foot_y = asset.data.body_pos_w[:,asset.data.body_names.index("RR_foot"),1] - env.scene.env_origins[:,1]
    result_RR =(RR_foot_y < -(RR_foot_z-RR_foot_beam_z)) | (RR_foot_y > RR_foot_z-RR_foot_beam_z)

    return  result_FL | result_FR | result_RL | result_RR

