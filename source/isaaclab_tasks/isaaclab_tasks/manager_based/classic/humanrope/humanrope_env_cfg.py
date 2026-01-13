# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanrope.mdp as mdp
from isaaclab.assets import RigidObjectCfg
import isaaclab.utils.math as math_utils
import torch
import math

from isaaclab_assets.robots.humanoid import HUMANOID_CFG

##
# Scene definition
##
_robot_orientation = (0.7071068, 0, 0, 0.7071068)
# _robot_orientation = (0.9914449, 0, 0.1305262, 0)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot walking on tight rope."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = HUMANOID_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=HUMANOID_CFG.init_state.replace(
        pos=(0.0, 0.0, 2.34),   
        rot=_robot_orientation, 
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    rope = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/rope",
                    spawn=sim_utils.CylinderCfg(
                        radius=0.05,
                        height=100.,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(45,0,0.8),rot=(0.7071068, 0, 0.7071068, 0)),
                )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_y_env = ObsTerm(func=mdp.base_pos_y_env)
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel_b = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_b = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping) in poleonhuman, not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )

        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    rew_progress = RewTerm(func=mdp.forward_speed, weight=1.0, params={"target_vel": 2.0})
    # (2) Stay alive bonus
    rew_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (3) Reward for maintaining desired orientation with less weight on pitch than roll and yaw
    rew_orientation = RewTerm(func=mdp.keep_orientation, weight=1.0, 
                              params={"target_quat": _robot_orientation})
    # (4) Reward for maintaining desired feet orientation with less weight on pitch than roll and yaw
    # rew_orientation_feet = RewTerm(func=mdp.keep_orientation_feet, weight=0.5, 
    #                           params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)})
    # Reward for keeping feet aligned in y direction
    # rew_feet_align = RewTerm(func=mdp.align_feet, weight=0.1)

    # (5) Penalty for large action commands
    cost_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    cost_energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.05,
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )
    # (7) Penalty for reaching close to joint limits
    cost_joint_limits = RewTerm(
        func=mdp.joint_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )

    # penalty for moving in y direction
    cost_off_track = RewTerm(func=mdp.off_track, weight=-1.0)

    # penalty for moving in z direction (avoid jumping)
    cost_jump_up = RewTerm(func=mdp.jump_up, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.85+0.9})
    # (3) Terminate if the robot deviates too much from target orientation
    torso_orientation = DoneTerm(func=mdp.bad_orientation_quat, params={"limit_angle_diff": math.pi/6,
                                                                        "target_quat": _robot_orientation} )
    # (4) Terminate if the feet deviate too much from target orientation
    feet_orientation = DoneTerm(func=mdp.bad_orientation_quat_feet, params={"limit_angle_diff": math.pi/2,
                                                                        "target_quat": _robot_orientation} )
    # (5) Terminate if the feet are off
    feet_off = DoneTerm(func=mdp.feet_off, params={"minimum_height": 0.86})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class HumanropeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking on tight rope environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0