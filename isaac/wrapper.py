import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from torchrl.envs import GymWrapper

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg


class IsaacEnv(GymWrapper):

    def __init__(
        self, 
        env_name, 
        num_envs=1, 
        use_fabric=True,
        seed=420,
        **kwargs
    ):
            
        env_cfg = parse_env_cfg(env_name, use_gpu=True, num_envs=num_envs, use_fabric=use_fabric)
        env = gym.make(env_name, cfg=env_cfg)
        #env = Sb3VecEnvWrapper(env)
        env.seed(seed=seed)
        super().__init__(env, **kwargs)

