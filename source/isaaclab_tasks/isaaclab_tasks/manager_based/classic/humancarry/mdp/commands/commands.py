# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals for objects."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import TargetPosCommandCfg


class TargetPosCommand(CommandTerm):
    """Command term that generates 3D pose commands for humancarry task.

    This command term generates 3D position commands for the object. The position commands
    are sampled uniformly from the 3D space. 

    The orientation is not included, so that the task is easier.

    Unlike typical command terms, where the goals are resampled based on time, this command term
    does not resample the goals based on time. Instead, the goals are resampled when the object
    reaches the goal orientation. The goal orientation is considered to be reached when the
    orientation error is below a certain threshold.
    """

    cfg: TargetPosCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TargetPosCommandCfg, env: ManagerBasedRLEnv):
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
        # -- command: (x, y, z)
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        # self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        # -- goal reset ids for reaching target
        self.goal_reset_ids = (self.metrics["position_error"]>1.0).nonzero(as_tuple=False).squeeze(-1)

    def __str__(self) -> str:
        msg = "HumancarryTargetPosCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 3)."""
        return self.pos_command_e

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self._env.scene.env_origins - self.pos_command_e, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_e[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_e[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_e[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = self.metrics["position_error"] < self.cfg.position_success_threshold
            self.goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(self.goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_e + self._env.scene.env_origins + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)
