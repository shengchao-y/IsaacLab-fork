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
from omni.isaac.lab.sensors import ContactSensorCfg

import numpy as np
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
import omni.isaac.lab_tasks.manager_based.classic.humancarry.mdp as mdp
import omni.isaac.lab.utils.math as math_utils
import torch
import math

##
# Scene definition
##

_box_size = (0.3, 0.3, 0.3)
_box_init_pose = (3.5, 0, 0.155)
_robot_orientation = (1.0, 0, 0, 0)
_position_success_threshold = 0.1

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.34),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_waist.*": 20.0,
                    ".*_upper_arm.*": 10.0,
                    "pelvis": 10.0,
                    ".*_lower_arm": 2.0,
                    ".*_thigh:0": 10.0,
                    ".*_thigh:1": 20.0,
                    ".*_thigh:2": 10.0,
                    ".*_shin": 5.0,
                    ".*_foot.*": 2.0,
                },
                damping={
                    ".*_waist.*": 5.0,
                    ".*_upper_arm.*": 5.0,
                    "pelvis": 5.0,
                    ".*_lower_arm": 1.0,
                    ".*_thigh:0": 5.0,
                    ".*_thigh:1": 5.0,
                    ".*_thigh:2": 5.0,
                    ".*_shin": 0.1,
                    ".*_foot.*": 1.0,
                },
            ),
        },
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box",
        spawn=sim_utils.CuboidCfg(
            size=_box_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # TODO: test box with different masses
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_box_init_pose,),
    )

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=False)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # box target position
    box_position = mdp.TargetPosCommandCfg(
        asset_name="box",
        update_goal_on_success=True,
        position_success_threshold=_position_success_threshold,
        make_quat_unique=False,
        marker_pos_offset=(-0.0, -0.0, 0.0),
        debug_vis=True,
        ranges=mdp.TargetPosCommandCfg.Ranges(
            pos_x=(3.5, 5), pos_y=(-2, 2), pos_z=(0.6, 1.5)
        )
    )


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
        # target
        target_pos_rel = ObsTerm(func=mdp.target_pos_rel, params={"command_name": "box_position"})

        # box
        box_position_rel = ObsTerm(func=mdp.object_pose_rel_b, params={"object_name": "box"})
        box_lin_vel_rel = ObsTerm(func=mdp.object_lin_vel_rel_b, params={"object_name": "box"})
        box_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"object_name": "box"})
        box_quat = ObsTerm(func=mdp.object_quat, params={"object_name": "box"})

        # robot non-joints
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the euler angles will change from -pi to pi (angle wrapping) in humancarry, not good for training
        root_quat = ObsTerm(func=mdp.root_quat_w)

        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot", "left_hand", "right_hand"])},
        )

        # joints
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

    reset_box = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        params={"object_name": "box", "pose_range": {}, "velocity_range": {}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for both hands reaching the holding points
    rew_hand2box = RewTerm(func=mdp.reach_box, weight=1.0, params={"box_size_y": _box_size[1]})
    # Reward for positioning of hands to box holding points proximity
    rew_handonbox = RewTerm(func=mdp.hold_box, weight=2.0, params={"box_size_y": _box_size[1], "dist_range": 1.0})
    # (2) Stay alive bonus
    rew_alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (3) Reward for maintaining roll and pitch angle close to 0 with less weight on pitch
    rew_orientation = RewTerm(func=mdp.keep_orientation_xy, weight=1.0, 
                              params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)})
    # (4) Reward for box reaching target
    rew_box2target = RewTerm(func=mdp.box_to_target, weight=1.0, params={"box_size_z": _box_size[2]})
    rew_boxontarget = RewTerm(func=mdp.box_on_target, weight=1.0,params={"position_success_threshold": _position_success_threshold})
    # Reward for keeping upper body gravity center in middle of two feet
    rew_center_support = RewTerm(func=mdp.center_support, weight=0.1)
    # Reward for maintaining desired orientation for body part with less weight on pitch than roll and yaw
    rew_left_foot_orientation = RewTerm(func=mdp.keep_orientation_body, weight=0.1, 
                                        params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0), 
                                                    "body_part": "left_foot"})
    rew_right_foot_orientation = RewTerm(func=mdp.keep_orientation_body, weight=0.1, 
                                        params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0), 
                                                    "body_part": "right_foot"})
    # (5) Penalty for large action commands
    cost_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    cost_energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.05*0.1,
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
        weight=-0.25*0.5,
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

    # cost for feet's contact forces
    cost_feet_contact = RewTerm(func=mdp.feet_contact_force, weight=-0.4, 
                                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot",]),})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls, TODO: maybe add maximum height to avoid pure jumping
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})    
    # (3) Terminate if the robot deviates too much from target orientation on roll and pitch
    torso_orientation = DoneTerm(func=mdp.bad_orientation_xy, params={"limit_angles_diff": (math.pi/4, math.pi/6),
                                                                        "target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)} )
    # (4) Terminate if the feet deviate too much from target orientation
    feet_orientation = DoneTerm(func=mdp.bad_orientation_quat_feet, params={"limit_angle_diff": math.pi/2,
                                                                        "target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)} )
    # Terminate if thighs' angle differ too much
    thigh_diff = DoneTerm(func=mdp.thigh_diff, params={"angle_limit": math.pi*0.7})
    # Terminate if support center deviates too much from feet center
    unstable_support = DoneTerm(func=mdp.unstable_support, params={"dist_limit": 0.4})
    # Terminate if box too near to any body part other than hands and feet
    box_near_body = DoneTerm(func=mdp.box_near_body, params={"dist_limit": min(_box_size)/2+0.15})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class HumancarryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid humancarry environment."""

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
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
