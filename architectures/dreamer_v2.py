# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from tensordict.nn import InteractionType

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeSequential,
)
from torchrl.modules.distributions import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import (
    ObsDecoder,
    ObsEncoder,
    RSSMRollout,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper

from functional.utils import *
from losses.dreamer_losses_v2 import *
from . import ArchitectureConfig
from envs.dreamer_env import DreamerEnv
from modules.rssm_v2 import RSSMPriorV2, RSSMPosteriorV2
from modules.actor import DreamerActorV2
from modules.blocks import LocScaleDist
import copy


class DreamerV2(ArchitectureConfig):
    
    def __init__(self, config, device):
        super(DreamerV2, self).__init__()
        print("DreamerV2")
        test_env = make_env(config, device="cpu")
        test_env = transform_env(config, test_env, parallel_envs=1, dummy=True)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = config.architecture.action_key,
            value_key = config.architecture.value_key,
        )
        self.slow_critic_update = config.optimization.slow_critic_update
        
        self.networks = self._init_networks(config, test_env)
        self.parts = self._init_modules(config, test_env, device)
        self.losses = self._init_losses(config)
        
        self.policy = self.parts["actor_realworld"]
        
    def update(self, step):
        with torch.no_grad():
            if step % self.slow_critic_update == 0:
                self.networks["value_target"].load_state_dict(self.networks["value_model"].state_dict())
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_vars = config.networks.state_vars
        state_classes = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        mlp_dims = config.networks.mlp_dims
        mlp_depth = config.networks.mlp_depth
        
        value_model = MLP(out_features=1, depth=mlp_depth, num_cells=mlp_dims, activation_class=activation)
        value_target = copy.deepcopy(value_model)
        return nn.ModuleDict(modules = dict(
                encoder = ObsEncoder(channels=32, num_layers=4), 
                decoder = ObsDecoder(channels=32, num_layers=4),
                rssm_prior = RSSMPriorV2(hidden_dim=hidden_dim, rnn_hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes, action_spec=action_spec),
                rssm_posterior = RSSMPosteriorV2(hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes),
                reward_model = MLP(out_features=1, depth=mlp_depth, num_cells=mlp_dims, activation_class=activation),
                actor_model = DreamerActorV2(out_features=action_spec.shape[-1], depth=mlp_depth, num_cells=mlp_dims, activation_class=activation, std_min_val=0.1),
                value_model = value_model,
                value_target = value_target
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model()
        actor_simulator = self._dreamer_make_actor_sim(proof_environment=proof_env)
        actor_realworld = self._dreamer_make_actor_real(proof_environment=proof_env)
        value_model = self._dreamer_make_value_model()
        value_target = self._dreamer_make_value_model_target()
        model_based_env = self._dreamer_make_mbenv(
            proof_env, state_dim=config.networks.state_dim, rssm_hidden_dim=config.networks.rssm_hidden_dim
        )
        
        with torch.no_grad():
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            tensordict = actor_simulator(tensordict)
            value_model(tensordict)
        
        with torch.no_grad():
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_target.device)
            tensordict = actor_simulator(tensordict)
            value_target(tensordict)
            
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
        
        return nn.ModuleDict(modules=dict(
            world_model = world_model.to(device),
            model_based_env = model_based_env.to(device),
            actor_simulator = actor_simulator.to(device),
            value_model = value_model.to(device),
            value_target = value_target.to(device),
            actor_realworld = actor_realworld.to(device)
        ))
        
    def _init_losses(self, config):
        losses = nn.ModuleDict(dict(
            world_model = DreamerModelLoss(
                self.parts["world_model"],
                lambda_kl=2.0,
                free_nats=1.0
            ).with_optimizer(params=self.parts["world_model"].parameters(), 
                             lr=config.optimization.world_model_lr, weight_decay=1e-6),
            
            actor = DreamerActorLoss(
                self.parts["actor_simulator"],
                self.parts["value_target"],
                self.parts["model_based_env"],
                imagination_horizon=config.optimization.imagination_horizon,
                discount_loss=False,
                entropy_regularization = 0,
            ).with_optimizer(params=self.parts["actor_simulator"].parameters(), 
                             lr=config.optimization.actor_lr, weight_decay=1e-6),
            
            value = DreamerValueLoss(
                self.parts["value_model"],
                discount_loss=False
            ).with_optimizer(params=self.parts["value_model"].parameters(), 
                             lr=config.optimization.value_lr, weight_decay=1e-6),
        ))
        #if config.env.backend == "gym" or config.env.backend == "gymnasium":
        #    losses["world_model"].set_keys(pixels="observation", reco_pixels="reco_observation")
        return losses


    def _dreamer_make_value_model(self):

        value_model = LocScaleDist(
            in_keys=["state", "belief"],
            out_key=self.keys["value_key"], 
            net=self.networks["value_model"],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
            loc_only=True
        )

        return value_model
    
    def _dreamer_make_value_model_target(self):

        value_model = LocScaleDist(
            in_keys=["state", "belief"],
            out_key=self.keys["value_key"], 
            net=self.networks["value_target"],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
            loc_only=True
        )
        
        return value_model



    def _dreamer_make_actor_sim(self, proof_environment):
        
        actor_simulator = LocScaleDist(
            in_keys=["state", "belief"],
            out_key=self.keys["action_key"],
            net=self.networks["actor_model"],
            distribution_class=TanhNormal,
            distribution_kwargs={"tanh_loc": True},
            spec = proof_environment.action_spec,
            default_interaction_type=InteractionType.RANDOM
        )
        
        return actor_simulator


    def _dreamer_make_actor_real(self, proof_environment):
        observation_in_key = self.keys["observation_in_key"]
        action_key = self.keys["action_key"]
        nets = self.networks
        # actor for real world: interacts with states ~ posterior
        # Out actor differs from the original paper where first they compute prior and posterior and then act on it
        # but we found that this approach worked better.
        actor_realworld = SafeSequential(
            SafeModule(
                nets["encoder"],
                in_keys=[observation_in_key],
                out_keys=["encoded_latents"],
            ),
            SafeModule(
                nets["rssm_posterior"],
                in_keys=["belief", "encoded_latents"],
                out_keys=[ "_", "state",],
            ),
            LocScaleDist(
                in_keys=["state", "belief"],
                out_key=action_key,
                net=nets["actor_model"],
                distribution_class=TanhNormal,
                distribution_kwargs={"tanh_loc": True},
                spec=proof_environment.action_spec, 
            ),
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", action_key],
                out_keys=["_", "_", "belief"], # we don't need the prior state
            ),
        )
        
        actor_realworld = AdditiveGaussianWrapper(
            actor_realworld,
            sigma_init=1.0,
            sigma_end=1.0,
            annealing_num_steps=1,
            mean=0.0,
            std=0.3,
        )
        
        
        return actor_realworld


    def _dreamer_make_mbenv(self, test_env, state_dim: int = 30, rssm_hidden_dim: int = 200):
        nets = self.networks
        
        # MB environment
        transition_model = SafeSequential(
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=["_", "state", "belief"],
            ),
        )

        reward_model = SafeModule(
                nets["reward_model"],
                in_keys=["state", "belief"],
                out_keys=["reward"],
        )
        
        mb_env_obs_decoder = SafeModule(
            nets["decoder"],
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", self.keys["observation_out_key"])],
        )

        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(
                transition_model,
                reward_model,
            ),
            prior_shape=torch.Size([state_dim]),
            belief_shape=torch.Size([rssm_hidden_dim]),
            obs_decoder=mb_env_obs_decoder,
        )

        model_based_env.set_specs_from_env(test_env)
        return model_based_env


    def _dreamer_make_world_model(self):
        nets = self.networks
        obs_in_key = self.keys["observation_in_key"]
        obs_out_key = self.keys["observation_out_key"]
        
        rssm_rollout = RSSMRollout(
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=[("next", "prior_logits"), "_", ("next", "belief")],
            ),
            SafeModule(
                nets["rssm_posterior"],
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )
        
        event_dim = 3 if obs_out_key == "reco_pixels" else 1  # 3 for RGB
        decoder = LocScaleDist(
            in_keys=[("next", "state"), ("next", "belief")],
            out_key=("next", obs_out_key),
            net=nets["decoder"],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": event_dim},
            loc_only=True,
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
        
        reward_model = LocScaleDist(
            in_keys=[("next", "state"), ("next", "belief")],
            out_key=("next", "reward"),
            net=nets["reward_model"],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
            loc_only=True,
        )

        world_model = WorldModelWrapper(
            transition_model,
            reward_model,
        )
        
        return world_model
