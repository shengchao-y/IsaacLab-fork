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
    return heading_vec[:,0] < minimum_heading_proj

def bad_pelvis_height(
    env: ManagerBasedEnv, minimum_height: float=0, maximum_height: float=100, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's humanoid pelvis is out of proper range.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.logical_or(asset.data.body_state_w[:, 5, 2] < minimum_height, asset.data.body_state_w[:, 5, 2] > maximum_height)
