import torch
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import DreamerDecoder
from torchrl.record import VideoRecorder

from torchrl.modules.models.model_based import RSSMRollout
from .objectives import DreamerModelLoss, DreamerActorLoss
from .dreamer_v2_value import DreamerV2value

def make_model(
    cfg,
    device,
    logger,
):
    # Model
    model = DreamerV2value(cfg, device)

    # Losses
    world_model_loss = DreamerModelLoss(
        model.modules["world_model"]
    ).with_optimizer(
        opt_module=model.modules["world_model"],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.optimization.world_model_lr, "weight_decay": 10e-6, "eps": 10e-5},
        use_grad_scaler=cfg.optimization.use_autocast
    )
    
    world_model_loss.loss_module.make_value_estimator(
        gamma=cfg.optimization.gamma, lmbda=cfg.optimization.lmbda
    )

    actor_loss = DreamerActorLoss(
        model.modules["actor_simulator"],
        model.modules["model_based_env"],
        imagination_horizon=cfg.optimization.imagination_horizon,
        discount_loss=True,
    ).with_optimizer(
        opt_module=model.modules["actor_simulator"],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.optimization.actor_lr, "weight_decay": 10e-6, "eps": 10e-5},
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
        }
    )
    
    policy = model.modules["actor_realworld"]
    mb_env = model.modules["model_based_env"]
    
    if cfg.logger.video:
        model_based_env_eval = mb_env.append_transform(DreamerDecoder())

        def float_to_int(data):
            reco_pixels_float = data.get("reco_pixels")
            reco_pixels = (reco_pixels_float * 255).floor()
            # assert (reco_pixels < 256).all() and (reco_pixels > 0).all(), (reco_pixels.min(), reco_pixels.max())
            reco_pixels = reco_pixels.to(torch.uint8)
            data.set("reco_pixels_float", reco_pixels_float)
            return data.set("reco_pixels", reco_pixels)

        model_based_env_eval.append_transform(float_to_int)
        model_based_env_eval.append_transform(
            VideoRecorder(
                logger=logger, tag="eval/simulated_rendering", in_keys=["reco_pixels"]
            )
        )
    else:
        model_based_env_eval = None
    
    return losses, policy, model_based_env_eval, model.update
    
    