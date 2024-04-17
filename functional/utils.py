# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from contextlib import nullcontext

import torch

import torch.nn as nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv

from torchrl.envs.env_creator import EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    # ExcludeTransform,
    FrameSkipTransform,
    GrayScale,
    ObservationNorm,
    RandomCropTensorDict,
    Resize,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.transforms.transforms import TensorDictPrimer

def make_env(cfg, device):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=cfg.env.from_pixels
            )
    elif lib == "dm_control":
        env = DMControlEnv(cfg.env.name, cfg.env.task, from_pixels=cfg.env.from_pixels)
        return env
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def transform_env(cfg, env, parallel_envs, dummy=False):
    env = TransformedEnv(env)
    if cfg.env.from_pixels:
        # transforms pixel from 0-255 to 0-1 (uint8 to float32)
        env.append_transform(ToTensorImage(from_int=True))
        if cfg.env.grayscale:
            env.append_transform(GrayScale())

        img_size = cfg.env.image_size
        env.append_transform(Resize(img_size, img_size))

    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())
    env.append_transform(FrameSkipTransform(cfg.env.frame_skip))
    
    if dummy:
        default_dict = {
            "state": UnboundedContinuousTensorSpec(shape=(cfg.networks.state_vars * cfg.networks.state_dim)),
            "belief": UnboundedContinuousTensorSpec(shape=(cfg.networks.rssm_hidden_dim)),
        }
    else:
        default_dict = {
            "state": UnboundedContinuousTensorSpec(shape=(parallel_envs, cfg.networks.state_vars * cfg.networks.state_dim)),
            "belief": UnboundedContinuousTensorSpec(shape=(parallel_envs, cfg.networks.rssm_hidden_dim)),
        }
    env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )

    return env


def make_environments(cfg, device, parallel_envs=1):
    """Make environments for training and evaluation."""
    
    train_env = ParallelEnv(
        parallel_envs,
        EnvCreator(lambda cfg=cfg: make_env(cfg, device=device)),
    )
    train_env = transform_env(cfg, train_env, parallel_envs)
    train_env.set_seed(cfg.env.seed)
    eval_env = ParallelEnv(
        parallel_envs,
        EnvCreator(lambda cfg=cfg: make_env(cfg, device=device)),
    )
    eval_env = transform_env(cfg, eval_env, parallel_envs)
    eval_env.set_seed(cfg.env.seed + 1)

    return train_env, eval_env


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
        reset_at_each_iter=True,
        # postproc=ExcludeTransform(
        #     "belief", "state", ("next", "belief"), ("next", "state"), "encoded_latents"
        # ),
    )
    collector.set_seed(cfg.env.seed)

    return collector


def make_replay_buffer(
    batch_size,
    batch_seq_len,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device="cpu",
    prefetch=3,
    pixel_obs=True,
    cast_to_uint8=True,
):
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        transforms = []
        crop_seq = RandomCropTensorDict(sub_seq_len=batch_seq_len, sample_dim=-1)
        transforms.append(crop_seq)

        if pixel_obs and cast_to_uint8:
            # from 0-255 to 0-1
            norm_obs = ObservationNorm(
                loc=0,
                scale=255,
                standard_normal=True,
                in_keys=["pixels", ("next", "pixels")],
            )
            transforms.append(norm_obs)

        transforms = Compose(*transforms)

        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            transform=transforms,
            batch_size=batch_size,
        )
        return replay_buffer


def cast_to_uint8(tensordict):
    tensordict["pixels"] = (tensordict["pixels"] * 255).to(torch.uint8)
    tensordict["next", "pixels"] = (tensordict["next", "pixels"] * 255).to(torch.uint8)
    return tensordict


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(name):
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "leaky_relu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    else:
        raise NotImplementedError
    
