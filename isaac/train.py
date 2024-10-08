# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")

parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Sati-Navigation-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=420, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10e6, help="RL Policy training iterations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hworldmodel.train import main as run_training
import isaac.envs.env_cfg


if __name__ == "__main__":
    # run the main function
    run_training()
    # close sim app
    simulation_app.close()
