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
    from omni.isaac.lab.envs import ManagerBasedEnv


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