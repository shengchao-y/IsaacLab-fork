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

import omni.isaac.lab_tasks.manager_based.classic.go2beam.mdp as mdp
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
import math

##
# Scene definition
##

_go2_init_pos = (0.0, 0.0, 0.4+0.85+0.15)
_beam_length = 50
## Change together
_beam_angle = math.pi/9
_beam_quat = (0.9848078, 0, -0.1736482, 0)
##
_beam_x = _beam_length*math.cos(_beam_angle)/2
_beam_z = 0.8+_beam_length*math.sin(_beam_angle)/2

def slope_func(x: float)-> float:
    """ Given x calculate the z value of the balance beam"""
    return math.tan(_beam_angle)*(x-_beam_x) + _beam_z

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a Go2 robot walking on tight rope."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/robot",
                            init_state=UNITREE_GO2_CFG.init_state.replace(pos=_go2_init_pos,rot=_beam_quat),)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # balance beam
    beam = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/beam",
                    spawn=sim_utils.CuboidCfg(
                        size=(_beam_length+2, 0.1, 0.1),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.6), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(_beam_x,0,_beam_z),
                                                              rot=_beam_quat),
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
        scale=23.5,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_y_env = ObsTerm(func=mdp.base_pos_y_env)
        base_height_rel = ObsTerm(func=mdp.base_pos_z_beam, params={"slope_func": slope_func})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # the roll angle will change from -pi to pi (angle wrapping), not good for training
        # base_yaw_pitch_roll = ObsTerm(func=mdp.base_eulers)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])},
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
    rew_progress = RewTerm(func=mdp.forward_speed, weight=1.0)
    # (2) Stay alive bonus
    rew_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # (3) Reward for not turning aside
    rew_no_side_turn = RewTerm(func=mdp.noside_posture_bonus, weight=0.1, params={"threshold": 0.93})

    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.05*0.5,
        params={
            "gear_ratio": {".*": 23.5}
        },
    )
    # (7) Penalty for reaching close to joint limits
    joint_limits = RewTerm(
        func=mdp.joint_limits_penalty_ratio,
        weight=-0.25*0.5,
        params={
            "threshold": 0.98,
            "gear_ratio": {".*": 23.5},
        },
    )

    # penalty for moving in y direction
    off_track = RewTerm(func=mdp.off_track, weight=-1.0)

    # penalty for moving away from beam (avoid jumping)
    jump_up = RewTerm(func=mdp.jump_up, weight=-1.0, params={"slope": _beam_angle})

    # reward for heading forward
    # heading_forward = RewTerm(func=mdp.heading_forward, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.fall_off_beam, params={"minimum_height": 0.05+0.26, "slope_func": slope_func})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class Go2beamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree Go2 walking on balance beam environment."""

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