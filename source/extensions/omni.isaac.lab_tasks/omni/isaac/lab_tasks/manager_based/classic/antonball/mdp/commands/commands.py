# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals for objects."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .commands_cfg import TargetDirCommandCfg


class TargetDirCommand(CommandTerm):
    """Command term that generates target direction command term for antonball.

    This command term generates target velocity direction commands for the object. The direction commands
    are initialized uniformly from [-pi, pi). 

    This command term resamples the goals based on time.
    """

    cfg: TargetDirCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TargetDirCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: heading
        self.heading_w = torch.zeros(self.num_envs, device=self.device)
        zeros = torch.zeros_like(self.heading_w)
        self.goal_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_w)

    def __str__(self) -> str:
        msg = "AntonballTargetDirCommandGenerator:\n"
        msg += f"\tCommand dimension: {(1,)}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired velocity direction in the world frame. Shape is (num_envs,)."""
        return self.heading_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the position error
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets
        r = torch.empty(len(env_ids), device=self.device)
        self.heading_w[env_ids] = r.uniform_(*self.cfg.range_heading)
        zeros = torch.zeros_like(self.heading_w)
        self.goal_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_w)

    def _update_command(self):
        # no need
        pass

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set visibility
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.object.data.root_pos_w
        # visualize the goal marker
        self.goal_vel_visualizer.visualize(translations=marker_pos, orientations=self.goal_quat)

        # current velocity marker
        current_marker_pos = marker_pos.clone()
        current_marker_pos[:,-1] += 1.1
        heading_angle = torch.atan2(self.object.data.root_lin_vel_w[:,1], self.object.data.root_lin_vel_w[:,0])
        zeros = torch.zeros_like(heading_angle)
        current_heading_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        current_arrow_scale = torch.ones_like(current_marker_pos)
        current_arrow_scale[:,0] *= torch.linalg.norm(self.object.data.root_lin_vel_w[:,:2], dim=1)
        # breakpoint()
        self.current_vel_visualizer.visualize(translations=current_marker_pos, orientations=current_heading_quat, scales=current_arrow_scale)
