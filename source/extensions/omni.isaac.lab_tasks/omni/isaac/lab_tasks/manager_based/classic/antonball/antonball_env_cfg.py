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
import omni.isaac.lab_tasks.manager_based.classic.antonball.mdp as mdp
from omni.isaac.lab_assets.ant import ANT_CFG  # isort: skip

##
# Scene definition
##

_ball_init_pos = (0.,0.,1.05)
_pole_init_pos = (0.0, 0.0, 3.8)
_ant_init_pos = (0, 0, 2.5)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
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

    # robot
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot",
                            init_state=ANT_CFG.init_state.replace(pos=_ant_init_pos),)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    pole = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pole",
        spawn=sim_utils.CylinderCfg(
            radius=0.08,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_pole_init_pos,),
    )
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_ball_init_pos,),
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

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        # pole
        pole_pos_rel = ObsTerm(func=mdp.object_pose_rel_b, params={"object_name": "pole"})
        pole_lin_vel = ObsTerm(func=mdp.object_lin_vel_rel_b, params={"object_name": "pole"})
        pole_ang_vel = ObsTerm(func=mdp.object_ang_vel, params={"object_name": "pole"})
        pole_quat = ObsTerm(func=mdp.object_quat, params={"object_name": "pole"})
        # ball
        ball_pos_rel = ObsTerm(func=mdp.object_pose_rel_b, params={"object_name": "ball"})
        ball_lin_vel = ObsTerm(func=mdp.object_lin_vel_rel_b, params={"object_name": "ball"})

        # robot non-joints
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping), not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)

        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso", "front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )

        # joints
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
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

    reset_pole = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        params={"object_name": "pole", "pose_range": {}, "velocity_range": {}},
    )

    reset_ball = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        params={"object_name": "ball", "pose_range": {}, "velocity_range": {}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) Reward for moving poles in x direction
    pole_moving = RewTerm(func=mdp.pole_moving, weight=1.0, params={"object_name": "pole", "target_vel": 1.0})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    # (7) Penalty for reaching close to joint limits
    joint_limits = RewTerm(
        func=mdp.joint_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    )
    # penalty for moving in y direction
    pole_off_track = RewTerm(func=mdp.object_off_track, weight=-1.0, params={"object_name": "pole"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the pole is too tilting or low
    pole_pose = DoneTerm(func=mdp.bad_object_pose, params={"object_name": "pole",
                                                           "minimum_height": 3.3,
                                                           "maximum_height": 3.8,
                                                           "min_z_proj": 0.98})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class AntonballEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style antonball acrobatics environment."""

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
