# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaacsim.core.utils.torch.rotations import normalize_angle
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def bad_orientation_quat(
    env: ManagerBasedRLEnv, limit_angle_diff: float, target_quat: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.
    This is computed by checking the normalized angle eulers' difference.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_diff = math_utils.quat_mul(target_quat.to(env.device).repeat(env.num_envs, 1), 
                                    asset.data.root_quat_w)
    eulers_diff = normalize_angle(torch.stack(math_utils.euler_xyz_from_quat(quat_diff), dim=1))
    return torch.norm(eulers_diff, dim=-1) > limit_angle_diff