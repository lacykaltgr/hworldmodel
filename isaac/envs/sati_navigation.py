# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp_navigation

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from isaac.assets.wheeled_actionterm import WheeledRobotActionTermCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns 
from omni.isaac.lab.terrains import TerrainImporterCfg

from isaac.assets.turtlebot import SATIDOG_CFG
from isaac.assets.camera_observationterm import camera_depth
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg
from omni.isaac.lab.managers import SceneEntityCfg

from ..assets.curriculum import task_order, node_based_termiantions
from ..assets.commands import UniformPose2dCommandCfg
from ..assets.curriculum import task_order
from ..assets.navigation import generated_commands, position_command_error_tanh, heading_command_error_abs, height_scan

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()

@configclass
class NavigationSceneCfg(InteractiveSceneCfg):

    #'''
    room = AssetBaseCfg(
        prim_path="/World/office", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Office/office.usd",
        )  
    )
    '''
    # lights for outdoor environments
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    '''

    robot: ArticulationCfg = SATIDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    #LIDAR
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(channels=16, vertical_fov_range = [-30.0, 30.0], horizontal_fov_range = [-180.0, 180.0], horizontal_res = 0.2),
        debug_vis=False,
        mesh_prim_paths=["/World/office"],
    )
    # GRID 
    '''
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.6, 4.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/office"],
    )
    '''

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
   
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/Camera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, -1.0, 1.0, -1.0), convention="ros"),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light1", 
        spawn=sim_utils.DomeLightCfg(intensity=8000.0, color=(0.75, 0.75, 0.75)),
        init_state = AssetBaseCfg.InitialStateCfg(
            pos=(-11.0, 0.0, 1.0)
        )
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light2", 
        spawn=sim_utils.DomeLightCfg(intensity=8000.0, color=(0.75, 0.75, 0.75)),
        init_state = AssetBaseCfg.InitialStateCfg(
            pos=(-12.0, 0.0, 1.0)
        )
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, -0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp_navigation.PreTrainedPolicyActionCfg = mdp_navigation.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DepthObsCfg(ObsGroup):
        """Observations for depth group."""

        camera_depth = ObsTerm(func=camera_depth)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityObsCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CommandObsCfg(ObsGroup):
        """Observations for policy group."""

        pose_command = ObsTerm(func=generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    """Observation specifications for the MDP."""

    @configclass
    class GravityCfg(ObsGroup):
        """Observations for projected gravity group."""

        
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    
    # observation groups
    velocity = VelocityObsCfg()
    command = CommandObsCfg()
    gravity = GravityCfg()
    depth = DepthObsCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    #'''
    pose_command = UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=UniformPose2dCommandCfg.Ranges(pos_x=(-0.3, 0.3), pos_y=(6.7, 7.3), heading=(-math.pi, math.pi)),

    )
    #'''
    
    '''
    pose_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base",
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-10.691331, -10.691331),
            pos_y=(-0.0, 0.0),
            pos_z=(1.6467236, 1.6467236),
            roll=(-0.0, 0.0), 
            pitch=(-0.0, 0.0),      
            yaw=(-0.0, 0.0), 
        ),
    )
    #'''


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    task_difficulty = CurrTerm(func=task_order)
    node_resets = CurrTerm(func=node_based_termiantions)
    pass



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 10000000.0},
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneEntityCfg = NavigationSceneCfg(num_envs=2, env_spacing=0.0)
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
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        '''
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        '''

@configclass
class NavigationEnvCfg_GRAPH(NavigationEnvCfg):
    scene: SceneEntityCfg = NavigationSceneCfg(num_envs=1, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    #graph_builder : None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        '''
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        '''