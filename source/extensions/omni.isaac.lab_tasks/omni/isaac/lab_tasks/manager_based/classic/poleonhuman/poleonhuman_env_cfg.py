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

import numpy as np
from omni.isaac.lab.assets import RigidObjectCfg
import omni.isaac.lab_tasks.manager_based.classic.poleonhuman.mdp as mdp

##
# Scene definition
##

_pole0_init_pose = (0.37, -0.17, 2.5)
# _pole1_init_pose = (0.37, 0.17, 2.5)

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

    pole0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pole0",
        spawn=sim_utils.CylinderCfg(
            radius=0.08,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_pole0_init_pose,),
    )
    # pole1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Pole1",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.08,
    #         height=2.0,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=_pole1_init_pose,),
    # )


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
        # pole
        pole0_position_rel_b = ObsTerm(func=mdp.object_pose_rel_b, params={"object_name": "pole0"})
        pole0_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_rel_b, params={"object_name": "pole0"})
        pole0_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"object_name": "pole0"})
        pole0_quat = ObsTerm(func=mdp.object_quat, params={"object_name": "pole0"})
        # pole1_position_rel = ObsTerm(func=mdp.object_pose_rel, params={"object_name": "pole1"})
        # pole1_lin_vel = ObsTerm(func=mdp.object_lin_vel, params={"object_name": "pole1"})
        # pole1_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"object_name": "pole1"})
        # pole1_quat = ObsTerm(func=mdp.object_quat, params={"object_name": "pole1"})

        # robot non-joints
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel_b = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_b = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping) in poleonhuman, not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)

        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot", "right_hand"])}, #, "left_hand"
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

    reset_pole0 = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        params={"object_name": "pole0", "pose_range": {}, "velocity_range": {}},
    )

    # reset_pole1 = EventTerm(
    #     func=mdp.reset_object_state_uniform,
    #     mode="reset",
    #     params={"object_name": "pole1", "pose_range": {}, "velocity_range": {}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving poles in x direction
    rew_pole0_moving = RewTerm(func=mdp.forward_speed_obj, weight=1.0, params={"object_name": "pole0", "target_vel": 0.5})
    # (2) Stay alive bonus
    rew_alive = RewTerm(func=mdp.is_alive, weight=1.0)

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
    cost_pole0_off_track = RewTerm(func=mdp.object_off_track, weight=-1.0, params={"object_name": "pole0"})
    # pole1_off_track = RewTerm(func=mdp.object_off_track, weight=-1.0, params={"object_name": "pole1"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the pole is too tilting or low
    pole0_pose = DoneTerm(func=mdp.bad_object_pose, params={"object_name": "pole0",
                                                           "minimum_height": 2.0,
                                                           "maximum_height": 2.6,
                                                           "min_z_proj": 0.98})
    # pole1_pose = DoneTerm(func=mdp.bad_object_pose, params={"object_name": "pole1",
    #                                                        "minimum_height": 2.0,
    #                                                        "maximum_height": 2.6,
    #                                                        "min_z_proj": 0.98})
    # (3) Terminate if the right hand is away from pole0
    pole0_off_hand = DoneTerm(func=mdp.pole0_off_hand, params={"dist_threshold": 0.1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class PoleonhumanEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid poleonhuman environment."""

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
