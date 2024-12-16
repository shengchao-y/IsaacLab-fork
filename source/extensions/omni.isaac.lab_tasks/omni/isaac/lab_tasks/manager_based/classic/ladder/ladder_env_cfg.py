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
import omni.isaac.lab_tasks.manager_based.classic.ladder.mdp as mdp
import omni.isaac.lab.utils.math as math_utils
import torch
import math

##
# Scene definition
##
_robot_orientation = (1.0, 0.0, 0.0, 0.0)

_step_poses = [(0.05*i+0.25, 0, 0.3*i) for i in range(1,30)]
def ladder_slope(z: float)-> float:
    return (z+1.5)/6.0

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
            pos=(0.0, 0.0, 1.3),
            joint_pos={
                    "lower_waist:0*": 0.0,
                    "lower_waist:1": 0.5,
                    ".*_upper_arm.*": 0.0,
                    "pelvis": 0.0,
                    ".*_lower_arm": 0.0,
                    ".*_thigh:0": 0.0,
                    ".*_thigh:1": -1.8,
                    ".*_thigh:2": 0.0,
                    ".*_shin": -1.6,
                    ".*_foot.*": 0.0,
                },
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

for i in range(len(_step_poses)):
    setattr(MySceneCfg,
            f"step{i}",
            RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Step"+str(i),
                    spawn=sim_utils.CuboidCfg(
                        size=(0.04, 10, 0.04),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=_step_poses[i]),
                )
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
        # Maybe the relative pos of steps are not needed, since they are fixed in world
        # step0_position_rel_x = ObsTerm(func=mdp.object_pose_rel_x, params={"object_name": "step0"})
        # relative height of next step above the torso
        step_above = ObsTerm(func=mdp.step_above, params={"height_steps": [step_pos[2] for step_pos in _step_poses]})
        base_x_rel = ObsTerm(func=mdp.base_dist_x_ladder, params={"ladder_slope": ladder_slope})

        # robot non-joints
        # base_x = ObsTerm(func=mdp.base_pos_x)
        # base_z = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping) in wall, not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)

        # TODO: maybe add forces for lower arms
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
        params={"pose_range": {}, "velocity_range": {"x": (0.5, 0.5), "z": (0.6, 0.6)}},
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

    # (1) Reward for moving upward at defined velocity
    progress = RewTerm(func=mdp.move_up_vel, weight=2.0, params={"target_vel": 0.2})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (3) Reward for maintaining desired orientation with less weight on pitch than roll and yaw
    rew_orientation = RewTerm(func=mdp.keep_orientation, weight=0.1, 
                              params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)})
    # (4) Reward for stepping around the ladder x position
    left_foot_near_ladder = RewTerm(func=mdp.body_part_near_x, weight=0.05, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "left_foot",
                                                                         "distance_limit": 0.2})
    right_foot_near_ladder = RewTerm(func=mdp.body_part_near_x, weight=0.05, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "right_foot",
                                                                         "distance_limit": 0.2})
    # Reward for holding around the ladder x position
    left_hand_near_ladder = RewTerm(func=mdp.body_part_near_x, weight=0.05, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "left_hand",
                                                                         "distance_limit": 0.1})
    right_hand_near_ladder = RewTerm(func=mdp.body_part_near_x, weight=0.05, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "right_hand",
                                                                         "distance_limit": 0.1})
    # Reward for keeping away from the ladder x position
    torso_away_ladder = RewTerm(func=mdp.body_part_away_x, weight=0.2, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "torso",
                                                                         "distance_limit": 0.4})
    pelvis_away_ladder = RewTerm(func=mdp.body_part_away_x, weight=0.4, params={"ladder_slope": ladder_slope, 
                                                                         "body_part": "pelvis",
                                                                         "distance_limit": 0.4})
    # (3) Reward for maintaining desired orientation for body part with less weight on pitch than roll and yaw
    rew_left_foot_orientation = RewTerm(func=mdp.keep_orientation_body, weight=0.1, 
                                        params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0), 
                                                    "body_part": "left_foot"})
    rew_right_foot_orientation = RewTerm(func=mdp.keep_orientation_body, weight=0.1, 
                                        params={"target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0), 
                                                    "body_part": "right_foot"})
    # move_to_target = RewTerm(
    #     func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    # )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.02,
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
    joint_limits = RewTerm(
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
    off_track = RewTerm(func=mdp.off_track, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls, TODO: maybe add maximum height to avoid pure jumping
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})
    # (3) Terminate if the robot deviates too much from target orientation
    torso_orientation = DoneTerm(func=mdp.bad_orientation_quat, params={"limit_angle_diff": math.pi/2,
                                                                        "target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)} )
    # terminate if hands too far away from ladder
    left_hand_off = DoneTerm(func=mdp.body_part_off_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "left_hand",
                                                                "distance_limit": 0.25})
    right_hand_off = DoneTerm(func=mdp.body_part_off_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "right_hand",
                                                                "distance_limit": 0.25})
    # terminate if body too near to ladder
    torso_near = DoneTerm(func=mdp.body_part_near_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "torso",
                                                                "distance_limit": 0.15})
    pelvis_near = DoneTerm(func=mdp.body_part_near_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "pelvis",
                                                                "distance_limit": 0.25})
    # terminate if feet too far away from ladder
    left_foot_off = DoneTerm(func=mdp.body_part_off_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "left_foot",
                                                                "distance_limit": 0.35})
    right_foot_off = DoneTerm(func=mdp.body_part_off_ladder, params={"ladder_slope": ladder_slope, 
                                                                "body_part": "right_foot",
                                                                "distance_limit": 0.35})
    # (4) Terminate if the feet deviate too much from target orientation
    # feet_orientation = DoneTerm(func=mdp.bad_orientation_quat_feet, params={"limit_angle_diff": math.pi/2,
    #                                                                     "target_quat": math_utils.quat_inv(torch.tensor(_robot_orientation)).unsqueeze(0)} )
    


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class LadderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid ladder environment."""

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
