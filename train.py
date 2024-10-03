# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import time

import hydra
import torch
import torch.cuda
import tqdm
import models
from .utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environments,
    make_replay_buffer,
    get_profiler,
    is_isaac_env,
    make_isaac_environments
)
from hydra.utils import instantiate

from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

import os
os.environ['BATCHED_PIPE_TIMEOUT'] = str(999999)


@hydra.main(version_base="1.1", config_path="models/IsaacNavigation/configs", config_name="isaac_config")
def main(cfg: "DictConfig"):  # noqa: F821
    # cfg = correct_for_frame_skip(cfg)

    device = torch.device(instantiate(cfg.networks.device))

    # Create logger
    exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name=f"{cfg.logger.model_name.lower()}_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "name":"isaac_cimbi"},  # "config": cfg},
        )

    is_isaaclab_env = is_isaac_env(cfg)
    if not is_isaaclab_env:
        train_env, test_env = make_environments(
            cfg=cfg,
            parallel_envs=cfg.env.n_parallel_envs,
            logger=logger,
        )
    else:
        train_env = make_isaac_environments(
            cfg=cfg
        )
        test_env = train_env
    
    model_module = getattr(models, cfg.logger.model_name)
    losses, policy, mb_env, callback = model_module.make_model(
        cfg=cfg,
        device=device,
        logger=logger,
        test_env=test_env if not is_isaaclab_env else train_env,
    )

    # save policy
    policy_save_dir = "/workspace/policy"
    os.makedirs(policy_save_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(policy_save_dir, "policy.pth"))
    # save config
    from omegaconf import OmegaConf
    config_path = os.path.join(policy_save_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Make collector
    collector = make_collector(cfg, train_env, policy)

    # Make replay buffer
    batch_size = cfg.replay_buffer.batch_size
    batch_length = cfg.replay_buffer.batch_length
    buffer_size = cfg.replay_buffer.buffer_size
    scratch_dir = cfg.replay_buffer.scratch_dir
    replay_buffer = make_replay_buffer(
        batch_size=batch_size,
        batch_seq_len=batch_length,
        buffer_size=buffer_size,
        buffer_scratch_dir=scratch_dir,
        device=device,
        pixel_obs=cfg.env.from_pixels,
        grayscale=cfg.env.grayscale,
        image_size=cfg.env.image_size,
        use_autocast=cfg.optimization.use_autocast,
    )

    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    # Grad scaler for mixed precision training https://pytorch.org/docs/stable/amp.html
    use_autocast = cfg.optimization.use_autocast
    init_random_frames = cfg.collector.init_random_frames
    optim_steps_per_batch = cfg.optimization.optim_steps_per_batch
    grad_clip = cfg.optimization.grad_clip
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.logger.eval_rollout_steps
    pretrain = cfg.collector.pretrain
    first = True

    with get_profiler(cfg) as profiler:
        t_collect_init = time.time()
        for i, tensordict in enumerate(collector):
            metrics = {}
            metrics.update({"timer/t_collect_init": time.time() - t_collect_init})

            t_preproc_init = time.time()
            pbar.update(tensordict.numel())
            current_frames = tensordict.numel()
            collected_frames += current_frames

            ep_reward = tensordict.get("episode_reward")[..., -1, 0]
            replay_buffer.extend(tensordict if tensordict.device == torch.device("cpu") else tensordict.cpu())
            metrics.update({"timer/t_prepro": time.time() - t_preproc_init})

            if collected_frames >= init_random_frames:
                metrics.update({
                    f"timer/t_loss_{name}": 0.0 for name in losses.keys()
                })
                
                if first: 
                    optim_step = pretrain
                    first = False
                else:
                    optim_step = optim_steps_per_batch
                
                for o in range(optim_step):
                    # sample from replay buffer
                    t_sample_init = time.time()
                    sampled_tensordict = replay_buffer.sample().reshape(-1, batch_length)
                    metrics.update({"timer/t_sample": time.time() - t_sample_init})


                    for name, loss in losses.items():
                        
                        t_loss_init = time.time()
                        
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                        ) if use_autocast else contextlib.nullcontext():
                            
                            loss_td, sampled_tensordict = loss(sampled_tensordict)
                            LOSS = loss.calculate_loss(loss_td)
                        
                        if o == optim_step - 1:
                            metrics.update({f"loss/{key}" : loss_td[key].item() 
                                            for key in loss_td.keys()})

                        loss.optimizer.zero_grad()
                        if use_autocast:
                            loss.grad_scaler.scale(LOSS).backward()
                            loss.grad_scaler.unscale_(loss.optimizer)
                        else:
                            LOSS.backward()
                        grad = loss.clip_grads(grad_clip)
                        metrics.update({f"grad/{name}": grad})
                        if use_autocast:
                            loss.grad_scaler.step(loss.optimizer)
                            loss.grad_scaler.update()
                        else:
                            loss.optimizer.step()
                            
                        metrics[f"timer/t_loss_{name}"] += time.time() - t_loss_init
                
                callback(i)

            metrics.update({"reward": ep_reward.mean().item()})
            if collected_frames >= init_random_frames:
                metrics.update(timeit.todict(percall=False))
                timeit.erase()

            if logger is not None:
                log_metrics(logger, metrics, collected_frames)

            if hasattr(policy, "step"):
                policy.step(current_frames)
            collector.update_policy_weights_()
            
            
            # Evaluation
            if (i % eval_iter) == 0:
                    # Real env
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                    eval_rollout = test_env.rollout(
                        eval_rollout_steps,
                        policy,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    test_env.apply(dump_video)
                    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                    eval_metrics = {f"eval/reward": eval_reward}
                    if logger is not None:
                        log_metrics(logger, eval_metrics, collected_frames)

                        
                # Simulated env
                if mb_env is not None:
                    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                        eval_rollout = mb_env.rollout(
                            eval_rollout_steps,
                            policy,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                            auto_reset=False,
                            tensordict=eval_rollout[..., 0]
                            .exclude("next", "action")
                            .to(device),
                        )
                        mb_env.apply(dump_video)
                        eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                        eval_metrics = {f"eval/simulated_reward": eval_reward}
                        if logger is not None:
                            log_metrics(logger, eval_metrics, collected_frames)
                        
            elif (i % eval_iter) == 0:
                pass


            profiler.step()
            t_collect_init = time.time()
            


if __name__ == "__main__":
    main()
