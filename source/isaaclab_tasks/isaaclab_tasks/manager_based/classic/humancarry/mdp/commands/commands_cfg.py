# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .commands import TargetPosCommand


@configclass
class TargetPosCommandCfg(CommandTermCfg):
    """Configuration for the uniform object 3D target pos command term.

    Please refer to the :class:`TargetPosCommand` class for more details.
    """

    class_type: type = TargetPosCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    position_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    # goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/goal_marker",
    #     markers={
    #         "goal": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #             scale=(1.0, 1.0, 1.0),
    #         ),
    #     },
    # )
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    """The configuration for the goal pose visualization marker. DexCube marker for pos+orientation
    and sphere for pos."""
