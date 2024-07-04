# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import re
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.utils import DEVICE_TYPING
from tensordict.base import NO_DEFAULT

from torchrl.envs.common import _EnvWrapper

from torchrl.envs.gym_like import BaseInfoDictReader, default_info_dict_reader



class IsaacLikeEnv(_EnvWrapper):
    """An Isaac-like env is an environment.

    Its behaviour is similar to Isaac Lab environments in what common methods (specifically reset and step) are expected to do.

    A :obj:`IsaacLikeEnv` has a :obj:`.step()` method with the following signature:

        ``env.step(action: torch.Tensor) -> Tuple[Union[torch.Tensor, dict], torch.Tensor, torch.Tensor, *info]``

    where the outputs are the observation, reward and done state respectively.
    In this implementation, the info output is discarded (but specific keys can be read
    by updating info_dict_reader, see :meth:`~.set_info_dict_reader` method).

    By default, the first output is written at the "observation" key-value pair in the output tensordict, unless
    the first output is a dictionary. In that case, each observation output will be put at the corresponding
    :obj:`f"{key}"` location for each :obj:`f"{key}"` of the dictionary.

    It is also expected that env.reset() returns an observation similar to the one observed after a step is completed.
    """

    _info_dict_reader: List[BaseInfoDictReader]

    def __init__(
        self,
        *args,
        device: DEVICE_TYPING = NO_DEFAULT,
        batch_size: Optional[torch.Size] = None,
        allow_done_after_reset: bool = False,
        detach: bool = True,
        **kwargs,
    ):
        super().__init__(*args, device=device, batch_size=batch_size, 
                         allow_done_after_reset=allow_done_after_reset **kwargs)
        self.detach = detach


    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._info_dict_reader = []
        return super().__new__(cls, *args, _batch_locked=True, **kwargs)

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        extracted_action = self.action_spec(action, safe=False)

        if self.detach:
            return extracted_action.detach()
        else:
            return extracted_action

    def read_done(
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Done state reader.

        In torchrl, a `"done"` signal means that a trajectory has reached its end,
        either because it has been interrupted or because it is terminated.
        Truncated means the episode has been interrupted early.
        Terminated means the task is finished, the episode is completed.

        Args:
            terminated (torch.Tensor or None): completion state
                obtained from the environment.
                ``"terminated"`` equates to ``"termination"`` in gymnasium:
                the signal that the environment has reached the end of the
                episode, any data coming after this should be considered as nonsensical.
                Defaults to ``None``.
            truncated (torch.Tensor or None): early truncation signal.
                Defaults to ``None``.
            done (torch.Tensor or None): end-of-trajectory signal.
                This should be the fallback value of envs which do not specify
                if the ``"done"`` entry points to a ``"terminated"`` or
                ``"truncated"``.
                Defaults to ``None``.

        Returns: a tuple with 4 tensor values,
            - a terminated state,
            - a truncated state,
            - a done state,
            - a boolean value indicating whether the frame_skip loop should be broken.

        """
        if truncated is not None and done is None:
            done = truncated | terminated
        elif truncated is None and done is None:
            done = terminated
        
        do_break = done.any().item() if not isinstance(done, bool) else done
        
        return (
            terminated,
            truncated,
            done,
            bool(do_break)  # Ensure do_break is a Python boolean
        )

    def read_reward(self, reward):
        """Reads the reward and maps it to the reward space.

        Args:
            reward (torch.Tensor or TensorDict): reward to be mapped.

        """
        if isinstance(reward, int) and reward == 0:
            return self.reward_spec.zero()
        reward = self.reward_spec.encode(reward, ignore_device=True)

        if reward is None:
            reward = torch.tensor(np.nan).expand(self.reward_spec.shape)

        return reward

    def read_obs(
        self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, Mapping):
            for key, spec in self.observation_spec.items(True, True):
                observations_dict = {}
                observations_dict[key] = spec.encode(observations, ignore_device=True)
                # we don't check that there is only one spec because obs spec also
                # contains the data spec of the info dict.
                break
            else:
                raise RuntimeError("Could not find any element in observation_spec.")
            observations = observations_dict
        else:
            for key, val in observations.items():
                observations[key] = self.observation_spec[key].encode(
                    val, ignore_device=True
                )
        return observations

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        action_ex = self.read_action(action)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(self._env.step(action_ex))

            if _reward is not None:
                reward = reward + _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward

        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done
        if truncated is not None:
            obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated
        validated = self.validated
        if not validated:
            tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)
            if validated is None:
                # check if any value has to be recast to something else. If not, we can safely
                # build the tensordict without running checks
                self.validated = all(
                    val is tensordict_out.get(key)
                    for key, val in TensorDict(obs_dict, []).items(True, True)
                )
        else:
            tensordict_out = TensorDict(
                obs_dict, batch_size=tensordict.batch_size, _run_checks=False
            )
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device, non_blocking=True)
            self._sync_device()

        if self.info_dict_reader and (info_dict is not None):
            if not isinstance(info_dict, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info_dict)} with values {str(info_dict)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info_dict, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        return tensordict_out

    @property
    def validated(self):
        return self.__dict__.get("_validated", None)

    @validated.setter
    def validated(self, value):
        self.__dict__["_validated"] = value

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        obs, info = self._reset_output_transform(self._env.reset(**kwargs))

        source = self.read_obs(obs)

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            _run_checks=not self.validated,
        )
        if self.info_dict_reader and info is not None:
            for info_dict_reader in self.info_dict_reader:
                out = info_dict_reader(info, tensordict_out)
                if out is not None:
                    tensordict_out = out
        elif info is None and self.info_dict_reader:
            # populate the reset with the items we have not seen from info
            for key, item in self.observation_spec.items(True, True):
                if key not in tensordict_out.keys(True, True):
                    tensordict_out[key] = item.zero()
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device, non_blocking=True)
            self._sync_device()
        return tensordict_out

    @abc.abstractmethod
    def _output_transform(
        self, step_outputs_tuple: Tuple
    ) -> Tuple[
        Any,
        float | np.ndarray,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        dict,
    ]:
        """A method to read the output of the env step.

        Must return a tuple: (obs, reward, terminated, truncated, done, info).
        If only one end-of-trajectory is passed, it is interpreted as ``"truncated"``.
        An attempt to retrieve ``"truncated"`` from the info dict is also undertaken.
        If 2 are passed (like in gymnasium), we interpret them as ``"terminated",
        "truncated"`` (``"truncated"`` meaning that the trajectory has been
        interrupted early), and ``"done"`` is the union of the two,
        ie. the unspecified end-of-trajectory signal.

        These three concepts have different usage:

          - ``"terminated"`` indicated the final stage of a Markov Decision
            Process. It means that one should not pay attention to the
            upcoming observations (eg., in value functions) as they should be
            regarded as not valid.
          - ``"truncated"`` means that the environment has reached a stage where
            we decided to stop the collection for some reason but the next
            observation should not be discarded. If it were not for this
            arbitrary decision, the collection could have proceeded further.
          - ``"done"`` is either one or the other. It is to be interpreted as
            "a reset should be called before the next step is undertaken".

        """
        ...

    @abc.abstractmethod
    def _reset_output_transform(self, reset_outputs_tuple: Tuple) -> Tuple:
        ...

    def set_info_dict_reader(
        self, info_dict_reader: BaseInfoDictReader | None = None
    ) -> IsaacLikeEnv:
        """Sets an info_dict_reader function.

        This function should take as input an
        info_dict dictionary and the tensordict returned by the step function, and
        write values in an ad-hoc manner from one to the other.

        Args:
            info_dict_reader (Callable[[Dict], TensorDict], optional): a callable
                taking a input dictionary and output tensordict as arguments.
                This function should modify the tensordict in-place. If none is
                provided, :class:`~torchrl.envs.gym_like.default_info_dict_reader`
                will be used.

        Returns: the same environment with the dict_reader registered.

        .. note::
          Automatically registering an info_dict reader should be done via
          :meth:`~.auto_register_info_dict`, which will ensure that the env
          specs are properly constructed.

        Examples:
            >>> from torchrl.envs import default_info_dict_reader
            >>> from torchrl.envs.libs.gym import GymWrapper
            >>> reader = default_info_dict_reader(["my_info_key"])
            >>> # assuming "some_env-v0" returns a dict with a key "my_info_key"
            >>> env = GymWrapper(gym.make("some_env-v0")).set_info_dict_reader(info_dict_reader=reader)
            >>> tensordict = env.reset()
            >>> tensordict = env.rand_step(tensordict)
            >>> assert "my_info_key" in tensordict.keys()

        """
        if info_dict_reader is None:
            info_dict_reader = default_info_dict_reader()
        self.info_dict_reader.append(info_dict_reader)
        if isinstance(info_dict_reader, BaseInfoDictReader):
            # if we have a BaseInfoDictReader, we know what the specs will be
            # In other cases (eg, RoboHive) we will need to figure it out empirically.
            if (
                isinstance(info_dict_reader, default_info_dict_reader)
                and info_dict_reader.info_spec is None
            ):
                torchrl_logger.info(
                    "The info_dict_reader does not have specs. The only way to palliate to this issue automatically "
                    "is to run a dummy rollout and gather the specs automatically. "
                    "To silence this message, provide the specs directly to your spec reader."
                )
                # Gym does not guarantee that reset passes all info
                self.reset()
                info_dict_reader.reset()
                self.rand_step()
                self.reset()

            for info_key, spec in info_dict_reader.info_spec.items():
                self.observation_spec[info_key] = spec.to(self.device)

        return self

    def auto_register_info_dict(self):
        """Automatically registers the info dict.

        It is assumed that all the information contained in the info dict can be registered as numerical values
        within the tensordict.

        This method returns a (possibly transformed) environment where we make sure that
        the :func:`torchrl.envs.utils.check_env_specs` succeeds, whether
        the info is filled at reset time.

        This method requires running a few iterations in the environment to
        manually check that the behaviour matches expectations.

        Examples:
            >>> from torchrl.envs import GymEnv
            >>> env = GymEnv("HalfCheetah-v4")
            >>> env.register_info_dict()
            >>> env.rollout(3)
        """
        from torchrl.envs import check_env_specs, TensorDictPrimer, TransformedEnv

        if self.info_dict_reader:
            raise RuntimeError("The environment already has an info-dict reader.")
        self.set_info_dict_reader()
        try:
            check_env_specs(self)
            return self
        except (AssertionError, RuntimeError) as err:
            patterns = [
                "The keys of the specs and data do not match",
                "The sets of keys in the tensordicts to stack are exclusive",
            ]
            for pattern in patterns:
                if re.search(pattern, str(err)):
                    result = TransformedEnv(
                        self, TensorDictPrimer(self.info_dict_reader[0].info_spec)
                    )
                    check_env_specs(result)
                    return result
            raise err

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    @property
    def info_dict_reader(self):
        return self._info_dict_reader
    

    """
    Helper functions.
    """

    def _process_isaac_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract observations from Isaac Lab env response."""
        # torchrl doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            if self.detach:
                for key, value in obs.items():
                    obs[key] = value.detach()
        elif isinstance(obs, torch.Tensor):
            if self.detach:
                obs = obs.detach()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_isaac_extras(
        self, obs: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, extras: dict, reset_ids: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            if idx in reset_ids:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos
