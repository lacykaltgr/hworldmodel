import gymnasium as gym
from .env_cfg import TurtlebotEnvCfg


gym.register(
    id="Turtlebot-AccCollision-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtlebotEnvCfg,
    },
)