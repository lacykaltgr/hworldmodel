import torch
import math
import gymnasium as gym

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg
from omni.isaac.lab.managers import SceneEntityCfg

from ..assets.wheeled_actionterm import WheeledRobotActionTermCfg
from ..assets.turtlebot import TURTLEBOT4_CFG
from ..assets.camera_observationterm import camera_depth



@configclass
class MySceneCfg(InteractiveSceneCfg):

    #room = AssetBaseCfg(
    #    prim_path="/World/room", 
    #    spawn=sim_utils.UsdFileCfg(
    #        usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/simple_room.usd",
    #    )    
    #)

    room = AssetBaseCfg(
        prim_path="/World/office", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Office/office.usd",
      )    
    )

    robot: ArticulationCfg = TURTLEBOT4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
        update_period=1/30,
        height=240,
        width=320,
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=4.81, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.3, 100)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, -1.0, 1.0, -1.0), convention="ros"),
    )

    contact_forces = ContactSensorCfg(
         prim_path="{ENV_REGEX_NS}/Robot/.*", 
         update_period=0.0, 
         history_length=6, 
         debug_vis=True
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=6000.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )




@configclass
class ActionsCfg:

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

    @configclass
    class PolicyCfg(ObsGroup):

        #joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        #joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        camera_depth = ObsTerm(func=camera_depth)

        def __post_init__(self):
            self.enable_corruption = True
            #self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    acc = RewTerm(
        func=mdp.body_lin_acc_l2, 
        weight=1.0
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="bumper.*"), "threshold": 1.0},
    )



@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    pass


@configclass
class CommandsCfg:
    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class TurtlebotEnvCfg(ManagerBasedRLEnvCfg):

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=0.01)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        #self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.dt = 0.01  # simulation timestep -> 100 Hz physics
        self.episode_length_s = 50.0

    
