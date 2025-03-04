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
from omni.isaac.core.utils.torch.rotations import normalize_angle

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def bad_heading(
    env: ManagerBasedEnv, minimum_heading_proj: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when deviating too much from heading forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    heading_vec = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    heading_vec_pelvis = math_utils.quat_rotate(asset.data.body_quat_w[:,5], asset.data.FORWARD_VEC_B)
    return torch.logical_or(heading_vec[:,0] < minimum_heading_proj, heading_vec_pelvis[:,0] < minimum_heading_proj)

def bad_pelvis_height(
    env: ManagerBasedEnv, minimum_height: float=0, maximum_height: float=100, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's humanoid pelvis is out of proper range.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # print(f"pelvis height: {asset.data.body_state_w[:, 5, 2]}")
    return torch.logical_or(asset.data.body_state_w[:, 5, 2] < minimum_height, asset.data.body_state_w[:, 5, 2] > maximum_height)

def hands_feet_align(
    env: ManagerBasedEnv, maximum_dist: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward having similar x values for hands and feet positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    # limbs_end = {"left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None}
    bodypart_inds = [asset.data.body_names.index(body_part) for body_part in ["left_hand", "right_hand", "left_foot", "right_foot"]]

    bodypart_x = torch.cat([asset.data.body_pos_w[:,ind_body_part,:1] for ind_body_part in bodypart_inds], dim=-1)
    diff_x = bodypart_x.max(dim=-1).values - bodypart_x.min(dim=-1).values
    return diff_x > maximum_dist

def bad_feet_heading(
    env: ManagerBasedEnv, minimum_heading_proj: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when feet deviating too much from heading forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    ind_left_foot = asset.data.body_names.index("left_foot")
    ind_right_foot = asset.data.body_names.index("right_foot")
    heading_vec_left_foot = math_utils.quat_rotate(asset.data.body_quat_w[:,ind_left_foot], asset.data.FORWARD_VEC_B)
    heading_vec_right_foot = math_utils.quat_rotate(asset.data.body_quat_w[:,ind_right_foot], asset.data.FORWARD_VEC_B)
    return torch.logical_or(heading_vec_left_foot[:,0] < minimum_heading_proj, heading_vec_right_foot[:,0] < minimum_heading_proj)
