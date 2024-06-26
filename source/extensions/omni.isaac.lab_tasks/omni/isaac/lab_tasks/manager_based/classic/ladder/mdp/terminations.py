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

from . import observations as obs
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv as RLTaskEnv


def body_part_off_ladder(
     env: RLTaskEnv, ladder_slope: func, body_part: str, distance_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """reward for stepping around ladder x position."""
    asset: Articulation = env.scene[asset_cfg.name]
    ind_body = asset.data.body_names.index(body_part)
    ladder_x = ladder_slope(asset.data.body_pos_w[:,ind_body,2])
    distance = torch.abs(asset.data.body_pos_w[:,ind_body,0] - env.scene.env_origins[:,0] - ladder_x)
    return distance > distance_limit