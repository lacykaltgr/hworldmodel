from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.vector import VectorEnv
import numpy as np
import os
import torch
from datetime import datetime
from torchrl.envs import GymWrapper

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv


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
        env = GymIsaacWrapper(env)
        env.seed(seed=seed)
        super().__init__(
            env, 
            device="cuda",
            **kwargs
        )




"""
Vectorized environment wrapper.
"""


class GymIsaacWrapper(VectorEnv):
    """Wraps around Isaac Lab environment for gymnasium.VectorEnv.

    The following changes are made to the environment to make it compatible with Isaac Lab:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # Check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # Initialize the wrapper
        self.env = env
        # Collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # Obtain gym spaces
        observation_space = self.unwrapped.single_observation_space["policy"]
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, spaces.Box) and not action_space.is_bounded("both"):
            action_space = spaces.Box(low=-100, high=100, shape=action_space.shape)

        # Initialize vec-env
        VectorEnv.__init__(self, self.num_envs, observation_space, action_space)
        # Add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self, seed: int | None = None, options: dict = None) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        # Convert data types to numpy depending on backend
        return self._process_obs(obs_dict)

    def step_async(self, actions):  # noqa: D102
        # Convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # Store the actions
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        # Record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
        # Update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        # Compute reset ids
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # Convert data types to numpy depending on backend
        obs = self._process_obs(obs_dict)
        rew = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        # Convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)

        # Reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, rew, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # Resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # Obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # Return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            return self.env.render()
        else:
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError("Checking if environment is wrapped is not supported.")

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Gymnasium doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # ManagerBasedRLEnv uses torch backend by default.
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_extras(
        self, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # Create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]
        # Fill-in information for each sub-environment
        for idx in range(self.num_envs):
            # Fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # Fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # Fill-in information from extras
            for key, value in extras.items():
                if key == "log":
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # Add information about terminal observation separately
            if idx in reset_ids:
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        return infos

