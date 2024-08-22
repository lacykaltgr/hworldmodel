import torch
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import DreamerDecoder
from torchrl.record import VideoRecorder

from torchrl.modules.models.model_based import RSSMRollout
from .objectives import DreamerModelLoss, DreamerActorLoss, DreamerValueLoss
from .dreamer_v2 import DreamerV2
from .modules import IsaacVideoRecorder

def make_model(
    cfg,
    device,
    logger,
    test_env
):
    # Model
    model = DreamerV2(cfg, test_env, device)

    # Losses
    world_model_loss = DreamerModelLoss(
        model.modules["world_model"]
    ).with_optimizer(
        opt_module=model.modules["world_model"],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.optimization.world_model_lr, "weight_decay": 10e-6, "eps": 10e-5},
        use_grad_scaler=cfg.optimization.use_autocast
    )
    
    # Adapt loss keys to gym backend
    #if cfg.env.backend == "gym":
    #    world_model_loss.set_keys(pixels="observation", reco_pixels="reco_observation")

    actor_loss = DreamerActorLoss(
        model.modules["actor_simulator"],
        model.modules["value_target"],
        model.modules["model_based_env"],
        imagination_horizon=cfg.optimization.imagination_horizon,
        discount_loss=True,
    ).with_optimizer(
        opt_module=model.modules["actor_simulator"],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.optimization.actor_lr, "weight_decay": 10e-6, "eps": 10e-5},
        use_grad_scaler=cfg.optimization.use_autocast
    )

    actor_loss.loss_module.make_value_estimator(
        gamma=cfg.optimization.gamma, lmbda=cfg.optimization.lmbda
    )
    
    value_loss = DreamerValueLoss(
        model.modules["value_model"], 
        discount_loss=True, 
        gamma=cfg.optimization.gamma
    ).with_optimizer(
        opt_module=model.modules["value_model"],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.optimization.value_lr, "weight_decay": 10e-6, "eps": 10e-5},
        use_grad_scaler=cfg.optimization.use_autocast
    )
    
    if cfg.optimization.compile:
        torch._dynamo.config.capture_scalar_outputs = True

        torchrl_logger.info("Compiling")
        backend = cfg.optimization.compile_backend

        def compile_rssms(module):
            if isinstance(module, RSSMRollout) and not getattr(
                module, "_compiled", False
            ):
                module._compiled = True
                module.rssm_prior.module = torch.compile(
                    module.rssm_prior.module, backend=backend
                )
                module.rssm_posterior.module = torch.compile(
                    module.rssm_posterior.module, backend=backend
                )

        world_model_loss.loss_module.apply(compile_rssms)
    
    losses = nn.ModuleDict(
        {
            "world_model": world_model_loss,
            "actor": actor_loss,
            "value": value_loss,
        }
    )
    policy = model.modules["actor_realworld"]
    mb_env = model.modules["model_based_env"]
    
    if cfg.logger.video:
        model_based_env_eval = mb_env.append_transform(DreamerDecoder())

        def depth_to_rgb(data):
            # depth is in range [-1, 1]
            reco_depth = data.get("reco_depth")
            reco_depth = (reco_depth * 255).floor()
            reco_depth = reco_depth.to(torch.uint8)
            reco_depth = torch.cat([reco_depth, reco_depth, reco_depth], dim=1)
            
            depth = data.get("depth")
            depth = (depth * 255).floor()
            depth = depth.to(torch.uint8)
            depth = torch.cat([depth, depth, depth], dim=1)

            data.set("reco_depth_rgb", reco_depth)
            data.set("depth_rgb", depth)

            return data

        model_based_env_eval.append_transform(depth_to_rgb)
        model_based_env_eval.append_transform(
            IsaacVideoRecorder(
                logger=logger, tag="eval/simulated_rendering", in_keys=["reco_depth_rgb"]
            )
        )
        model_based_env_eval.append_transform(
            IsaacVideoRecorder(
                logger=logger, tag="eval/rendering", in_keys=["depth_rgb"]
            )
        )

    else:
        model_based_env_eval = None
    
    return losses, policy, model_based_env_eval, model.update
    
    