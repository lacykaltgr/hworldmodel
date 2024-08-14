# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp_navigation

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
from isaac.assets.wheeled_actionterm import WheeledRobotActionTermCfg

from isaac.assets.turtlebot import TURTLEBOT3_CFG
from isaac.assets.camera_observationterm import camera_depth
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg
from omni.isaac.lab.managers import SceneEntityCfg

from assets.navigation import generated_commands, position_command_error_tanh, heading_command_error_abs


@configclass
class NavigationSceneCfg(InteractiveSceneCfg):

    room = AssetBaseCfg(
        prim_path="/World/office", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
      )    
    )

    robot: ArticulationCfg = TURTLEBOT3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
        update_period=1/30,
        height=240,
        width=320,
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=7, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.3, 100)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, -1.0, 1.0, -1.0), convention="ros"),
    )

    contact_forces = ContactSensorCfg(
         prim_path="{ENV_REGEX_NS}/Robot/base_link", 
         update_period=0.0, 
         history_length=6, 
         debug_vis=True
    )


@configclass
class EventCfg:
    """
    Configuration for events.

    - randomly sample the position, velocity, pose of the robot upon reset
    
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
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

    control = WheeledRobotActionTermCfg(
        asset_name="robot",
        joint_names=["wheel_.*_joint"],
        wheel_radius = 0.025,
        wheel_base = 0.16,
        max_linear_speed = 0.22,
        max_angular_speed = 1.90,
        max_wheel_speed = 0.22 / 0.025
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

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


    @configclass
    class DepthObsCfg(ObsGroup):
        """Observations for policy group."""

        camera_depth = ObsTerm(func=camera_depth)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    velocity = VelocityObsCfg()
    command = CommandObsCfg()
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
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneEntityCfg = NavigationSceneCfg(num_envs=2, env_spacing=1)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 8.0

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False