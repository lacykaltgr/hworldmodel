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
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.wheeled_robots")
enable_extension("omni.anim.people")
simulation_app.update()

import torch
import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
import isaac.envs.navigation.NavigationEnvCfg_GRAPH as NavigationEnvCfg_GRAPH
from isaac.assets.graph_build import GraphBuilder
from isaac.wrapper import IsaacEnv


def main():
    """Main function."""
    ALL_EDGES_FILE_PATH = "/path/to/edges.txt"
    POLICY_FILE_PATH = "/path/to/policy.pth"

    edges = torch.load(ALL_EDGES_FILE_PATH)

    graph_builder = GraphBuilder(edges)
    env_cfg = NavigationEnvCfg_GRAPH(graph_builder=graph_builder)

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = IsaacEnv(env=env, num_envs=1)

    polciy = torch.load(POLICY_FILE_PATH)

    while graph_builder.not_ready():
        td = env.reset()
        env.rollout(
            policy=polciy,
            max_steps=1,
            render_mode="rgb_array",
            render_kwargs={"mode": "human"},
            tensordict=td,
        )
        # check if it succeeded
        # update graph

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()