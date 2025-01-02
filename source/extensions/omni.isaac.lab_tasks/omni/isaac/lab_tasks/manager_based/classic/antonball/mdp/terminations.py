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

def bad_orientation_quat(
    env: ManagerBasedEnv, limit_angle_diff: float, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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