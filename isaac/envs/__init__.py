import gymnasium as gym
from .env_cfg import TurtlebotEnvCfg
from.navigation import NavigationEnvCfg
from .sati_walking import WalkingSceneCfg as sati_walking
from .sati_navigation import NavigationEnvCfg as sati_navigate

gym.register(
    id="Turtlebot-AccCollision-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtlebotEnvCfg,
    },
)

gym.register(
    id="Turtlebot-Navigation-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NavigationEnvCfg,
    },
)

gym.register(
    id="Sati-Walking-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sati_walking,
    },
)

gym.register(
    id="Sati-Navigation-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sati_navigate,
    },
)