import hydra
import torch
import torch.cuda
import tqdm
from importlib import import_module
from utils import *

# mixed precision training
from torch.cuda.amp import autocast
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from architectures import ArchitectureConfig


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    # cfg = correct_for_frame_skip(cfg)

    if torch.cuda.is_available() and cfg.networks.device == "":
        device = torch.device("cuda:0")
    elif cfg.networks.device:
        device = torch.device(cfg.networks.device)
    else:
        device = torch.device("cpu")

    # Create logger
    exp_name = generate_exp_name(cfg.architecture.name, cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name=f"{cfg.architecture.name.lower()}_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode},  # "config": cfg},
            
        )

    train_env, test_env = make_environments(cfg=cfg, device=device)

    
    # Create the model
    architecture_module = import_module(cfg.architecture.module)
    architecture = getattr(architecture_module, cfg.architecture.name)
    model: ArchitectureConfig = architecture(cfg, device)

    # Make collector
    collector = make_collector(cfg, train_env, model.policy)

    # Make replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.replay_buffer.batch_size,
        batch_seq_len=cfg.optimization.batch_length,
        buffer_size=cfg.replay_buffer.buffer_size,
        buffer_scratch_dir=cfg.replay_buffer.scratch_dir,
        device=cfg.networks.device,
        pixel_obs=cfg.env.from_pixels,
        cast_to_uint8=cfg.replay_buffer.uint8_casting,
    )

    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    batch_size = cfg.optimization.batch_size
    optim_steps_per_batch = cfg.optimization.optim_steps_per_batch
    grad_clip = cfg.optimization.grad_clip
    uint8_casting = cfg.replay_buffer.uint8_casting
    pixel_obs = cfg.env.from_pixels
    frames_per_batch = cfg.collector.frames_per_batch
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.logger.eval_rollout_steps

    for _, tensordict in enumerate(collector):
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames

        if uint8_casting and pixel_obs:
            tensordict = cast_to_uint8(tensordict)

        ep_reward = tensordict.get("episode_reward")[:, -1]
        replay_buffer.extend(tensordict.cpu())

        loss_tds = dict()
        if collected_frames >= init_random_frames:
            
            pixels = None
            reco_pixels = None
            
            for i in range(optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(batch_size).to(
                    device, non_blocking=True
                )
                
                
                for name, loss in model.losses.items():
                    with autocast(dtype=torch.float16):
                        loss_td, sampled_tensordict = loss(sampled_tensordict)
                        LOSS = loss.calculate_loss(loss_td)
                        loss_tds[name] = (loss_td, LOSS)
                        
                    if i == optim_steps_per_batch - 1 and name == "world_model":
                        pixels = sampled_tensordict["next", "pixels"]
                        reco_pixels = sampled_tensordict["next", "reco_pixels"]
                        
                    loss.optimizer.zero_grad()
                    loss.grad_scaler.scale(LOSS).backward()
                    loss.grad_scaler.unscale_(loss.optimizer)
                    loss.clip_grads(grad_clip)
                    loss.grad_scaler.step(loss.optimizer)
                    loss.grad_scaler.update()
            
            metrics_to_log = dict()
            logger.log_video(
                "rollout/target",
                (pixels[0].unsqueeze(0) * 255).detach().cpu().to(torch.uint8),
            )
            logger.log_video(
                "rollout/reconstruction",
                (reco_pixels[0].unsqueeze(0) * 255).detach().cpu().to(torch.uint8),
            )
            
            for name, loss in model.losses.items():
                loss_td, LOSS = loss_tds[name]
                metrics_to_log.update(
                    {loss.get_loss_name(): LOSS.item()}, 
                    **{key: loss_td[key].item() for key in loss.get_loss_keys()}
                )
                
            if collected_frames % 100_000 == 0:
                torch.save(model.state_dict(), f"model_{collected_frames}.pt")

        metrics_to_log.update({"reward": ep_reward.item()})
        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)

        model.policy.step(current_frames)
        collector.update_policy_weights_()
        
        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_rollout = test_env.rollout(
                    max_steps = eval_rollout_steps,
                    policy = model.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                eval_metrics = {"eval/reward": eval_reward}
                if logger is not None:
                    log_metrics(logger, eval_metrics, collected_frames)
                    logger.log_video(
                        "eval/rollout",
                        (eval_rollout["pixels"] * 255).detach().cpu().to(torch.uint8),
                    )
                    
                with autocast(dtype=torch.float16):
                    imgn_td = eval_rollout.select("state", "belief", ("next", "reward")).to(device)
                    imgn_td = model.parts["model_based_env"].to(device).rollout(
                        max_steps=100,
                        policy=model.parts["actor_simulator"],
                        auto_reset=False,
                        tensordict=imgn_td[:, 0],
                    )
                    reco_pixels = model.parts["model_based_env"].decode_obs(imgn_td)["next", "reco_pixels"]
                    logger.log_video(
                        "eval/rollout/simulated",
                        (reco_pixels * 255).detach().cpu().to(torch.uint8),
                    )

if __name__ == "__main__":
    main()