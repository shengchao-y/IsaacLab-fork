# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab_assets import HUMANOID_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

import torch
import omni.isaac.lab.utils.math as math_utils
from collections.abc import Sequence
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn

ball_init_pose = (0.5, 0, 0.15)

@configclass
class HumandribbleEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    num_actions = 21
    num_observations = 81
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    # ball
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.4,),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ball_init_pose,),
        
    )

    energy_cost_scale: float = 0.01
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 0.4
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.6

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class HumandribbleEnv(LocomotionEnv):
    cfg: HumandribbleEnvCfg

    def __init__(self, cfg: HumandribbleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.direction_angles = torch.zeros_like(self.basis_vec0[:,0], device=self.device)
        self.last_pos_turn = torch.zeros_like(self.basis_vec0, device=self.device)
        self.direction_quats = quat_from_euler_xyz(torch.zeros_like(self.basis_vec0[:,0]), torch.zeros_like(self.basis_vec0[:,0]), 
                                                   self.direction_angles)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "rew_ball_vel",
                "rew_alive",
                "rew_ball_dist",
                "cost_action",
                "cost_energy",
                "cost_joint_limit",
                "cost_ball_off",
                "cost_orient",
            ]
        }
    
    def _setup_scene(self):
        super()._setup_scene()
        self.ball = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["ball"] = self.ball

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids=env_ids)
        # reset ball
        self.ball.reset(env_ids)
        default_ball_state = self.ball.data.default_root_state[env_ids]
        default_ball_state[:, :3] += self.scene.env_origins[env_ids]

        self.ball.write_root_pose_to_sim(default_ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(default_ball_state[:, 7:], env_ids)

        # reset turning date
        self.direction_angles[env_ids] = 0
        self.direction_quats[env_ids] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.last_pos_turn[env_ids] = self.ball.data.root_pos_w[env_ids]
        self.last_pos_turn[:,-1] = 0
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        self.ball_dist = torch.norm(self.ball.data.root_pos_w[:,:2]-self.torso_position[:,:2], dim=-1)
        self.torso_rot_local = quat_mul(quat_conjugate(self.direction_quats), self.torso_rotation)

    def _change_direction(self, env_ids):
        """ change goal direction of envs_ids
        """
        num_resets = len(env_ids)
        direction_angles_delta = (torch.rand(num_resets, dtype=torch.float32, device=self.device)*2-1) * torch.pi / 6 # [-pi/6, pi/6]
        probs = torch.ones_like(direction_angles_delta) * 0.3
        direction_angles_delta[torch.bernoulli(probs).bool()] = 0.
        self.direction_angles[env_ids] = normalize_angle(self.direction_angles[env_ids]+direction_angles_delta)
        self.direction_quats = quat_from_euler_xyz(torch.zeros_like(self.basis_vec0[:,0]), torch.zeros_like(self.basis_vec0[:,0]), 
                                                   self.direction_angles)
        self.last_pos_turn[env_ids,:2] = self.ball.data.root_pos_w[env_ids,:2]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # breakpoint()
        died, time_out = super()._get_dones()
        change_buf = (self.episode_length_buf+1) % 180 == 0
        change_dir_env_ids = change_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(change_dir_env_ids)>0:
            self._change_direction(change_dir_env_ids)

        left_foot_quat = self.robot.data.body_quat_w[:,-1,:]
        right_foot_quat = self.robot.data.body_quat_w[:,-2,:]
        left_foot_up = quat_rotate(left_foot_quat, self.basis_vec1)
        right_foot_up = quat_rotate(right_foot_quat, self.basis_vec1)
        died = died | (left_foot_up[:,2]<0.35) | (right_foot_up[:,2]<0.35) | (self.ball_dist>1)
        return died, time_out

    def _get_observations(self) -> dict:
        """ get the observation for humanoid dribbling
        output: 
            {"policy": obs}
        """
        # breakpoint()
        # get info from simulator
        torso_pos_local = math_utils.quat_rotate_inverse(self.direction_quats, self.torso_position-self.last_pos_turn)
        ball_position = self.ball.data.root_pos_w
        ball_vel_lin = self.ball.data.root_lin_vel_w

        # calculate observation values
        ball_pos_b = quat_rotate_inverse(self.torso_rotation, ball_position-self.torso_position)
        ball_vel_b = quat_rotate_inverse(self.torso_rotation, ball_vel_lin)

        # the roll angle will change from -pi to pi (angle wrapping) in poleonhuman, not good for training
        # roll, pitch, yaw = get_euler_xyz(self.torso_rot_local)

        obs = torch.cat(
            (
                torso_pos_local[:,1:],                                              # 2
                self.robot.data.root_lin_vel_b,                                     # 3
                self.robot.data.root_ang_vel_b * self.cfg.angular_velocity_scale,   # 3
                # normalize_angle(roll).unsqueeze(-1),
                # normalize_angle(pitch).unsqueeze(-1),
                # normalize_angle(yaw).unsqueeze(-1),
                # the roll angle will change from -pi to pi (angle wrapping) in poleonhuman, not good for training
                self.torso_rot_local,                                               # 4
                ball_pos_b,                                                         # 3
                ball_vel_b,                                                         # 3
                self.dof_pos_scaled,                                                # 21
                self.dof_vel * self.cfg.dof_vel_scale,                              # 21
                self.actions,                                                       # 21
            ), dim=-1
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # breakpoint()
        
        # reward for duration of staying alive
        rew_alive = torch.ones_like(self.actions[:,0]) * self.cfg.alive_reward_scale

        # reward for keeping ball velocity at 1.5m/s
        ball_vel_lin_local = quat_rotate_inverse(self.direction_quats, self.ball.data.root_lin_vel_w)
        rew_ball_vel = torch.exp(-torch.abs(ball_vel_lin_local[:,0] - 1.5)/0.25) * 2.0

        # reward for keeping close to the ball
        rew_ball_dist = torch.exp(-self.ball_dist) * 0.2

        # penalty for ball off track
        ball_pos_local = quat_rotate_inverse(self.direction_quats, self.ball.data.root_pos_w-self.last_pos_turn)
        cost_ball_off_track = torch.square(ball_pos_local[:,1]) * 0.5

        # penalty for torso not heading target direction
        # ************************************ self.torso_rot_local=[1,0,0,0]
        torso_heading_vec = get_basis_vector(self.torso_rot_local, self.basis_vec0)
        cost_orient = torch.abs(torso_heading_vec[:,1]) * 1.0

        # energy penalty for movement
        actions_cost = torch.sum(self.actions ** 2, dim=-1)
        electricity_cost = torch.sum(torch.abs(self.actions * self.dof_vel) * self.motor_effort_ratio.unsqueeze(0), dim=-1)

        # dof at limit cost
        dof_at_limit_cost = torch.sum(self.dof_pos_scaled > 0.98, dim=-1)

        rewards = {
            "rew_ball_vel":     rew_ball_vel * self.step_dt,
            "rew_alive":        rew_alive * self.step_dt,
            "rew_ball_dist":    rew_ball_dist * self.step_dt,
            "cost_action":      - self.cfg.actions_cost_scale * actions_cost * self.step_dt,
            "cost_energy":      - self.cfg.energy_cost_scale * electricity_cost * self.step_dt,
            "cost_joint_limit": - dof_at_limit_cost * self.step_dt,
            "cost_ball_off":    - cost_ball_off_track * self.step_dt,
            "cost_orient":      - cost_orient * self.step_dt,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # adjust reward for fallen agents
        # total_reward = torch.where(self.reset_terminated, torch.ones_like(total_reward) * self.cfg.death_cost, total_reward)
        return total_reward
