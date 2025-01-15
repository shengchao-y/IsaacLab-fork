# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from .commands import TargetDirCommand


@configclass
class TargetDirCommandCfg(CommandTermCfg):
    """Configuration for the uniform target direction command term.

    Please refer to the :class:`TargetDirCommand` class for more details.
    """

    class_type: type = TargetDirCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    range_heading: tuple[float, float] = MISSING
    """Ranges for the heading command (in rad)."""

    # goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/goal_marker",
    #     markers={
    #         "goal": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #             scale=(1.0, 1.0, 1.0),
    #         ),
    #     },
    # )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 1.0)
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 1.0)
    """The configuration for the goal pose visualization marker. DexCube marker for pos+orientation
    and sphere for pos."""
