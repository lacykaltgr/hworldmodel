# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claus

"""
This script demonstrates the environment for a quadruped robot with height-scan sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The height-scan sensor is used to detect the height of
the terrain.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p source/standalone/tutorials/04_envs/quadruped_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument("--cameras", default=False, help="Enable cameras.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.cameras:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.wheeled_robots")
simulation_app.update()

"""Rest everything follows."""

import torch
import math

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
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
from isaac.assets.wheeled_actionterm import WheeledRobotActionTermCfg

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaac.assets.turtlebot import TURTLEBOT4_CFG
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg 


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
    terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    #room = AssetBaseCfg(
    #    prim_path="/World/room", 
    #    spawn=sim_utils.UsdFileCfg(
    #        usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/simple_room.usd",
    #    )    
    #)

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
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, -0.25, 0.0, -1), convention="ros"),
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
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class TurtlebotEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1  # env decimation -> 50 Hz control
        # simulation settings
        #self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        #self.sim.physics_material = self.scene.terrain.physics_material

def main():
    """Main function."""
    # setup base environment
    env_cfg = TurtlebotEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    print(f"[INFO]: Environment observation space: {env_cfg.observations}")
    print(f"[INFO]: Environment action space: {env_cfg.actions}")

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = torch.tensor([10, 0])
            # take action n_envs times
            action = action.repeat(env.num_envs, 1)
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()