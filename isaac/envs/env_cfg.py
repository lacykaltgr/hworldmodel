import torch
import math
import gymnasium as gym

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from isaac.assets.wheeled_actionterm import WheeledRobotActionTermCfg

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaac.assets.turtlebot import TURTLEBOT4_CFG
from isaac.assets.camera_observationterm import camera_rgb
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg 
from omni.isaac.lab.envs import ManagerBasedRLEnv


##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""
    
    """
    # add terrain
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
        debug_vis=False,
    )
    """
    #terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    room = AssetBaseCfg(
        prim_path="/World/room", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/simple_room.usd",
        )    
    )

    # add robot
    robot: ArticulationCfg = TURTLEBOT4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
        update_period=1/30,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=4.81, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.3, 100)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, -1.0, 1.0, -1.0), convention="ros"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=6000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control = WheeledRobotActionTermCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint*"],
        wheel_radius = 0.072,
        wheel_base = 0.3,
        max_linear_speed = 0.31,
        max_angular_speed = 1.90,
        max_wheel_speed = 0.31 / 0.072
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        #joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        #joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        camera_rgb = ObsTerm(func=camera_rgb)

        def __post_init__(self):
            self.enable_corruption = True
            #self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


##
# Environment configuration
##


@configclass
class TurtlebotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1  # env decimation -> 50 Hz control
        # simulation settings
        #self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        #self.sim.physics_material = self.scene.terrain.physics_material
