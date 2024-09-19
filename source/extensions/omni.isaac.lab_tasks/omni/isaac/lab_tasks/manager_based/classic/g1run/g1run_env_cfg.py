# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.classic.g1run.mdp as mdp
from omni.isaac.lab_assets.unitree import G1_CFG  # isort: skip
import math
import omni.isaac.lab.utils.math as math_utils
import torch

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a Unitree G1 robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/robot",
                           spawn=G1_CFG.spawn.replace(articulation_props=G1_CFG.spawn.articulation_props.replace(enabled_self_collisions=True)),)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           joint_names=[".*_hip_yaw_joint",
                                                        ".*_hip_roll_joint",
                                                        ".*_hip_pitch_joint",
                                                        ".*_knee_joint",
                                                        "torso_joint",
                                                        ".*_ankle_pitch_joint", 
                                                        ".*_ankle_roll_joint",
                                                        ".*_shoulder_pitch_joint",
                                                        ".*_shoulder_roll_joint",
                                                        ".*_shoulder_yaw_joint",
                                                        ".*_elbow_pitch_joint",
                                                        ".*_elbow_roll_joint",], 
                                           scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping) in poleonhuman, not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
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
    rew_progress = RewTerm(func=mdp.forward_speed, weight=0.3)
    # (2) Stay alive bonus
    rew_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (3) Reward for maintaining desired orientation with less weight on pitch than roll and yaw
    rew_orientation = RewTerm(func=mdp.keep_orientation, weight=1.0, 
                              params={"target_quat": math_utils.quat_inv(torch.tensor((0.9659258, 0, 0.258819, 0))).unsqueeze(0)})

    # (5) Penalty for large action commands
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Penalty for large joint_torque
    cost_dof_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    # (6) Penalty for large joint_acc
    cost_dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)

    cost_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

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
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.6})
    # torso_orientation = DoneTerm(func=mdp.bad_orientation_quat, params={"limit_angle_diff": math.pi/2,
    #                                                                     "target_quat": math_utils.quat_inv(torch.tensor((1.0,0.0,0.0,0.0))).unsqueeze(0)} )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class G1runEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree G1 running environment."""

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
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0