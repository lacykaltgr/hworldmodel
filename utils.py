# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import tempfile
from contextlib import nullcontext

import torch
import torch.nn as nn
from hydra.utils import instantiate


from torchrl.collectors import SyncDataCollector

from torchrl.data import (
    LazyMemmapStorage,
    SliceSampler,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
)

from torchrl.envs import (
    Compose,
    DeviceCastTransform,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ExcludeTransform,
    FrameSkipTransform,
    GrayScale,
    GymEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    set_gym_backend,
    StepCounter,
    TensorDictPrimer,
    ToTensorImage,
    TransformedEnv,
    ObservationNorm,
    ClipTransform
)
from torchrl.envs.utils import check_env_specs
from torchrl.record import VideoRecorder
from isaac.wrapper import IsaacEnv



def _make_env(cfg, device, from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            env = GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=cfg.env.from_pixels or from_pixels,
                pixels_only=cfg.env.from_pixels,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            from_pixels=cfg.env.from_pixels or from_pixels,
            pixels_only=cfg.env.from_pixels,
        )
    elif lib == "isaac_lab":
        env = IsaacEnv(
            cfg.env.name,
            num_envs=cfg.env.n_parallel_envs,
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")
    default_dict = {
        "state": UnboundedContinuousTensorSpec(shape=(cfg.networks.state_dim * cfg.networks.state_vars,)),
        "belief": UnboundedContinuousTensorSpec(shape=(cfg.networks.rssm_hidden_dim,)),
    }
    env = env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )
    assert env is not None
    return env


def transform_env(cfg, env):
    if not isinstance(env, TransformedEnv):
        env = TransformedEnv(env)
    if cfg.env.from_pixels:
        # transforms pixel from 0-255 to 0-1 (uint8 to float32)
        env.append_transform(
            RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"])
        )
        env.append_transform(
            ToTensorImage(from_int=True, in_keys=["pixels_int"], out_keys=["pixels"])
        )
        if cfg.env.grayscale:
            env.append_transform(GrayScale())

        image_size = cfg.env.image_size
        env.append_transform(Resize(image_size, image_size))

    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())
    env.append_transform(FrameSkipTransform(cfg.env.frame_skip))
    #env.append_transform(StepCounter(cfg.env.horizon))

    return env


def make_environments(cfg, parallel_envs=1, logger=None):
    """Make environments for training and evaluation."""
    func = functools.partial(_make_env, cfg=cfg, device=cfg.env.device)
    train_env = ParallelEnv(
        parallel_envs,
        EnvCreator(func),
        serial_for_single=True,
    )
    train_env = transform_env(cfg, train_env)
    train_env.set_seed(cfg.env.seed)
    func = functools.partial(
        _make_env, cfg=cfg, device=cfg.env.device, from_pixels=cfg.logger.video
    )
    eval_env = ParallelEnv(
        1,
        EnvCreator(func),
        serial_for_single=True,
    )
    eval_env = transform_env(cfg, eval_env)
    eval_env.set_seed(cfg.env.seed + 1)
    if cfg.logger.video:
        eval_env.insert_transform(0, VideoRecorder(logger, tag="eval/video"))
    check_env_specs(train_env)
    check_env_specs(eval_env)
    return train_env, eval_env

def is_isaac_env(cfg):
    return cfg.env.backend == "isaac_lab"

def make_isaac_environments(cfg):
    #func = functools.partial(_make_env, cfg=cfg, device=cfg.env.device)
    train_env = _make_env(cfg=cfg, device=cfg.env.device)
    train_env = transform_isaac_env(cfg, train_env)
    train_env.set_seed(cfg.env.seed)
    check_env_specs(train_env)
    return train_env


def transform_isaac_env(cfg, env, logger=None):
    if not isinstance(env, TransformedEnv):
        env = TransformedEnv(env)
    if cfg.env.from_pixels:
        env.append_transform(
            RenameTransform(in_keys=["observation"], out_keys=["pixels_int"])
        )
        # transforms pixel from 0-255 to 0-1 (uint8 to float32)
        env.append_transform(
            ToTensorImage(in_keys=["pixels_int"], out_keys=["pixels"], shape_tolerant=True)
        )
        if cfg.env.grayscale:
            env.append_transform(GrayScale())

        image_size = cfg.env.image_size
        env.append_transform(Resize(image_size, image_size))
    else:

        env.append_transform(
            Resize(cfg.env.image_size, cfg.env.image_size, in_keys=["depth"], out_keys=["depth"])
        )
        env.append_transform(
            ObservationNorm(loc=0, scale=10, in_keys=["depth"], out_keys=["depth"], standard_normal=True)
        )
        env.append_transform(
            ClipTransform(low=0, high=1, in_keys=["depth"], out_keys=["depth"])
        )

    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())
    return env

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        policy_device=instantiate(cfg.collector.device),
        env_device=train_env.device,
        storing_device="cpu",
    )
    collector.set_seed(cfg.env.seed)

    return collector


def make_replay_buffer(
    *,
    batch_size,
    batch_seq_len,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device=None,
    prefetch=2,
    pixel_obs=True,
    grayscale=True,
    image_size,
    use_autocast,
):
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        transforms = Compose()
        if pixel_obs:

            def check_no_pixels(data):
                assert "pixels" not in data.keys()
                return data

            #transforms = Compose(
                #ExcludeTransform("pixels", ("next", "pixels"), inverse=True),
                #check_no_pixels,  # will be called only during forward
                #ToTensorImage(
                #    in_keys=["pixels_int", ("next", "pixels_int")],
                #    out_keys=["pixels", ("next", "pixels")],
                #),
            #)
            #if grayscale:
            #    transforms.append(GrayScale(in_keys=["pixels", ("next", "pixels")]))
            #transforms.append(
            #    Resize(image_size, image_size, in_keys=["pixels", ("next", "pixels")])
            #)
        transforms.append(DeviceCastTransform(device=device))

        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device="cpu",
                ndim=2,
            ),
            sampler=SliceSampler(
                slice_len=batch_seq_len,
                strict_length=True,
                traj_key=("collector", "traj_ids"),
                cache_values=True,
                compile=False, # TODO: set it to False for isaac_lab experiments
            ),
            transform=transforms,
            batch_size=batch_size,
        )
        return replay_buffer



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
    

def _default_device(device=None):
    if device in ("", None):
        if torch.cuda.is_available():
            return torch.device("cuda:1")
        return torch.device("cpu")
    return torch.device(device)


def get_profiler(conf):
    from torch.profiler import ProfilerActivity
    
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
        
    class NoProfiler:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_value, exc_traceback):
            pass
        def step(self):
            pass
    
    if conf.logger.enable_profiler:
        return torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
            ],
            #profile_memory=True,
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=2, repeat=3),
            on_trace_ready=trace_handler,
        )
    else:
        return NoProfiler()

