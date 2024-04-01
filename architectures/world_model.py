# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torch.nn as nn
from tensordict.nn import InteractionType

from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
)
from torchrl.modules.distributions import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper

from functional.utils import *
from losses.dreamer_losses import *
from . import ArchitectureConfig
from envs.dreamer_env import DreamerEnv


class WorldModel(ArchitectureConfig):
    
    def __init__(self, config, device):
        test_env = make_env(config, device="cpu")
        test_env = transform_env(config, test_env, parallel_envs=1, dummy=True)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = config.architecture.action_key,
            value_key = config.architecture.value_key,
        )
        
        self.policy = None
        
        self.networks = self._init_networks(config, test_env)
        self.parts = self._init_modules(config, test_env, device)
        self.losses = self._init_losses(config)
        
    
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_dim = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        return dict(
            encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=64, depth=2, num_cells=hidden_dim, activation_class=activation)),
            decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
            rssm_prior = RSSMPrior(hidden_dim=hidden_dim,  rnn_hidden_dim=rssm_dim, state_dim=state_dim, action_spec=action_spec),
            rssm_posterior = RSSMPosterior(hidden_dim=rssm_dim, state_dim=state_dim),
            reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model()
            
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
        
        return dict(
            world_model = world_model.to(device)
        )
        
    def _init_losses(self, config):
        
        losses = dict(
            world_model = DreamerModelLoss(
                self.parts["world_model"],
                lambda_kl=config.optimization.kl_scale,
                free_nats = config.optimization.free_nats,
            ).with_optimizer(params=self.parts["world_model"].parameters(), 
                             lr=config.optimization.world_model_lr, weight_decay=1e-5),
        )
        #if config.env.backend == "gym" or config.env.backend == "gymnasium":
        #    losses["world_model"].set_keys(pixels="observation", reco_pixels="reco_observation")
        return losses



    def _dreamer_make_world_model(self):
        nets = self.networks
        obs_in_key = self.keys["observation_in_key"]
        obs_out_key = self.keys["observation_out_key"]
        
        rssm_rollout = RSSMRollout(
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=[("next", "prior_mean"), ("next", "prior_std"), "_", ("next", "belief")],
            ),
            SafeModule(
                nets["rssm_posterior"],
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_mean"), ("next", "posterior_std"), ("next", "state")],
            ),
        )
        event_dim = 3 if obs_out_key == "reco_pixels" else 1  # 3 for RGB
        decoder = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=["loc"],
            ),
            SafeProbabilisticModule(
                in_keys=["loc"],
                out_keys=[("next", obs_out_key)],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": event_dim},
            ),
        )

        transition_model = SafeSequential(
            SafeModule(
                nets["encoder"],
                in_keys=[("next", obs_in_key)],
                out_keys=[("next", "encoded_latents")],
            ),
            rssm_rollout,
            decoder,
        )

        reward_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["reward_model"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "loc")],
            ),
            SafeProbabilisticModule(
                in_keys=[("next", "loc")],
                out_keys=[("next", "reward")],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": 1},
            ),
        )

        world_model = WorldModelWrapper(
            transition_model,
            reward_model,
        )
        
        return world_model
