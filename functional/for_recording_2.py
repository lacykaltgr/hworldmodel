import dataclasses
from pathlib import Path
from importlib import import_module

import hydra
import torch
import torch.cuda
import tqdm
from hworldmodel_torch.dreamer.for_recording import (
    cast_to_uint8,
    log_metrics,
    make_collector,
    make_dreamer,
    make_environments,
    make_replay_buffer,
    retrieve_stats_from_state_dict,
    ArchitectureConfig,
)
from hydra.core.config_store import ConfigStore

# float16
from torch.cuda.amp import autocast
from torchrl._utils import logger as torchrl_logger

from torchrl.envs import EnvBase
from torchrl.modules.tensordict_module.exploration import (
    AdditiveGaussianWrapper,
    OrnsteinUhlenbeckProcessWrapper,
)

from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    initialize_observation_norm_transforms,
    retrieve_observation_norms_state_dict,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.models import DreamerConfig, make_dreamer
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import TrainerConfig
from torchrl.trainers.trainers import Recorder, RewardNormalizer

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        LoggerConfig,
        ReplayArgsConfig,
        DreamerConfig,
        TrainerConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)



@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    if torch.cuda.is_available() and cfg.model_device == "":
        device = torch.device("cuda:0")
    elif cfg.model_device:
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    torchrl_logger.info(f"Using device {device}")

    exp_name = generate_exp_name(cfg.architecture, cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name=cfg.architecture.lower(),
        experiment_name=exp_name,
        wandb_kwargs={
            "project": cfg.project_name,
            "group": f"{cfg.architecture}_{cfg.env_name}",
            "offline": cfg.offline_logging,
        },
    )
    video_tag = f"{cfg.architecture}_{cfg.env_name}_policy_test" if cfg.record_video else ""

    key, init_env_steps, stats = None, None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = ("next", "pixels") if cfg.from_pixels else ("next", "observation_vector")
        init_env_steps = cfg.init_env_steps
        stats = {"loc": None, "scale": None}
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    proof_env = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats
    )()
    initialize_observation_norm_transforms(
        proof_environment=proof_env, num_iter=init_env_steps, key=key
    )
    _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]
    proof_env.close()

    # Create the different components of the model
    architecture_module = import_module(cfg.architecture_module)
    architecture: ArchitectureConfig = getattr(architecture_module, cfg.architecture)()
    
    losses, modules = architecture.initialize(
        cfg=cfg, 
        proof_environment=transformed_env_constructor(cfg, stats={"loc": 0.0, "scale": 1.0})(),
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="state_value",
    )

    # reward normalization
    if cfg.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(
            scale=cfg.normalize_rewards_online_scale,
            decay=cfg.normalize_rewards_online_decay,
        )
    else:
        reward_normalizer = None


    # Exploration noise to be added to the actions
    if cfg.exploration == "additive_gaussian":
        exploration_policy = AdditiveGaussianWrapper(
            modules.policy,
            sigma_init=0.3,
            sigma_end=0.3,
        ).to(device)
    elif cfg.exploration == "ou_exploration":
        exploration_policy = OrnsteinUhlenbeckProcessWrapper(
            modules.policy,
            annealing_num_steps=cfg.total_frames,
        ).to(device)
    elif cfg.exploration == "":
        exploration_policy = modules.policy.to(device)

    action_dim_gsde, state_dim_gsde = None, None
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        obs_norm_state_dict=obs_norm_state_dict,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )
    if isinstance(create_env_fn, EnvBase):
        create_env_fn.rollout(2)
    else:
        create_env_fn().rollout(2)

    # Create the replay buffer

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=exploration_policy,
        cfg=cfg,
    )
    torchrl_logger.info(f"collector: {collector}")

    replay_buffer = make_replay_buffer("cpu", cfg)

    record = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=modules.policy,
        environment=make_recorder_env(
            cfg=cfg,
            video_tag=video_tag,
            obs_norm_state_dict=obs_norm_state_dict,
            logger=logger,
            create_env_fn=create_env_fn,
        ),
        record_interval=cfg.record_interval,
        log_keys=cfg.recorder_log_keys,
    )

    final_seed = collector.set_seed(cfg.seed)
    torchrl_logger.info(f"init seed: {cfg.seed}, final seed: {final_seed}")
    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    path = Path("./log")
    path.mkdir(exist_ok=True)


    for i, tensordict in enumerate(collector):
        cmpt = 0
        if reward_normalizer is not None:
            reward_normalizer.update_reward_stats(tensordict)
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames

        # Compared to the original paper, the replay buffer is not temporally
        # sampled. We fill it with trajectories of length batch_length.
        # To be closer to the paper, we would need to fill it with trajectories
        # of length 1000 and then sample subsequences of length batch_length.

        tensordict = tensordict.reshape(-1, cfg.batch_length)
        replay_buffer.extend(tensordict.cpu())
        logger.log_scalar(
            "r_training",
            tensordict["next", "reward"].mean().detach().item(),
            step=collected_frames,
        )

        do_log = (i % cfg.record_interval) == 0
        if collected_frames >= cfg.init_random_frames:
            if i % cfg.record_interval == 0:
                logger.log_scalar("cmpt", cmpt)
            for j in range(cfg.optim_steps_per_batch):
                cmpt += 1
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(
                    device, non_blocking=True
                )
                if reward_normalizer is not None:
                    sampled_tensordict = reward_normalizer.normalize_reward(
                        sampled_tensordict
                    )
                    
                for name, loss in losses.items():
                    with autocast(dtype=torch.float16):
                        loss_td, sampled_tensordict = loss(sampled_tensordict)
                        LOSS = loss.calculate_loss(loss_td)
                        loss.scaler.scale(LOSS).backward()
                        loss.scaler.unscale_(loss.optimizer)
                        loss.clip_grads(cfg.grad_clip)
                        loss.scaler.step(loss.optimizer)
                        
                        if j == cfg.optim_steps_per_batch - 1 and do_log: 
                            logger.log_scalar(
                                loss.get_loss_name(),
                                LOSS.detach().item(),
                                step=collected_frames,
                            )
                            logger.log_scalar(
                                f"grad_{name}",
                                grad_norm(loss.optimizer),
                                step=collected_frames,
                            )
                            
                            for key in loss.get_loss_keys():
                                logger.log_scalar(
                                    key,
                                    loss_td[key].detach().item(),
                                    step=collected_frames,
                                )
                                
                        loss.optimizer.zero_grad()
                        loss.scaler.update()

                if j == cfg.optim_steps_per_batch - 1:
                    do_log = False

            stats = retrieve_stats_from_state_dict(obs_norm_state_dict)
            call_record(
                logger,
                record,
                collected_frames,
                sampled_tensordict_save,
                stats,
                modules.model_based_env,
                modules.actor_model,
                cfg,
            )
        if cfg.exploration != "":
            exploration_policy.step(current_frames)
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()
