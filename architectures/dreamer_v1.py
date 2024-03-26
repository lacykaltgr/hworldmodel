# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torch.nn as nn
from tensordict.nn import InteractionType

from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.model_based.dreamer import DreamerEnv
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


class DreamerV1(ArchitectureConfig):
    
    def __init__(self, config, device):
        test_env = make_env(config, device="cpu")
        test_env = transform_env(config, test_env, parallel_envs=1, dummy=True)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = config.architecture.action_key,
            value_key = config.architecture.value_key,
        )
        
        self.networks = self._init_networks(config, test_env)
        self.modules = self._init_modules(config, test_env, device)
        self.losses = self._init_losses(config)
        
        self.policy = self.modules["actor_realworld"]
        
    
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_dim = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        
        return dict(
            encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=1024, depth=2, num_cells=hidden_dim, activation_class=activation)),
            decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
            rssm_prior = RSSMPrior(hidden_dim=hidden_dim,  rnn_hidden_dim=rssm_dim, state_dim=state_dim, action_spec=action_spec),
            rssm_posterior = RSSMPosterior(hidden_dim=rssm_dim, state_dim=state_dim),
            reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
            actor_model = DreamerActor(out_features=action_spec.shape[-1], depth=3, num_cells=hidden_dim, activation_class=activation),
            value_model = MLP(out_features=1, depth=3, num_cells=hidden_dim, activation_class=activation)
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model()
        actor_simulator = self._dreamer_make_actor_sim(proof_environment=proof_env)
        actor_realworld = self._dreamer_make_actor_real(proof_environment=proof_env, exploration_noise=config.networks.exploration_noise)
        value_model = self._dreamer_make_value_model()
        model_based_env = self._dreamer_make_mbenv(
            proof_env, use_decoder_in_env=config.architecture.use_decoder_in_env, state_dim=config.networks.state_dim, rssm_hidden_dim=config.networks.rssm_hidden_dim
        )
        
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            tensordict = actor_simulator(tensordict)
            value_model(tensordict)
            
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
        
        return dict(
            world_model = world_model.to(device),
            model_based_env = model_based_env.to(device),
            actor_simulator = actor_simulator.to(device),
            value_model = value_model.to(device),
            actor_realworld = actor_realworld.to(device)
        )
        
    def _init_losses(self, config):
        modules = self.modules
        losses = dict(
            world_model = DreamerModelLoss(
                modules["world_model"]
            ).optimize(modules=[modules["world_model"]], lr=config.optimization.world_model_lr),
            
            actor = DreamerActorLoss(
                modules["actor_simulator"],
                modules["value_model"],
                modules["model_based_env"],
                imagination_horizon=config.optimization.imagination_horizon,
                discount_loss = True
            ).optimize(modules=[modules["actor_simulator"]], lr=config.optimization.actor_lr),
            
            value = DreamerValueLoss(
                modules["value_model"],
                discount_loss=True
            ).optimize(modules=[modules["value_model"]], lr=config.optimization.value_lr),
        )
        if config.env.backend == "gym":
            losses.world_model_loss.set_keys(pixels="observation", reco_pixels="reco_observation")
        return losses



    def _dreamer_make_value_model(self):
        nets = self.networks
        value_key = self.keys["value_key"]
        
        value_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["value_model"],
                in_keys=["state", "belief"],
                out_keys=["loc"],
            ),
            SafeProbabilisticModule(
                in_keys=["loc"],
                out_keys=[value_key],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": 1},
            ),
        )

        return value_model



    def _dreamer_make_actor_sim(self, proof_environment):
        nets = self.networks
        action_key = self.keys["action_key"]
        
        actor_simulator = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["actor_model"],
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
                spec=CompositeSpec(
                    **{
                        "loc": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                            device=proof_environment.action_spec.device,
                        ),
                        "scale": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                            device=proof_environment.action_spec.device,
                        ),
                    }
                ),
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=[action_key],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
                distribution_kwargs={"tanh_loc": True},
                spec=CompositeSpec(**{action_key: proof_environment.action_spec}),
            ),
        )
        return actor_simulator


    def _dreamer_make_actor_real(self, proof_environment, exploration_noise):
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
                out_keys=[ "_", "_", "state",],
            ),
            SafeProbabilisticTensorDictSequential(
                SafeModule(
                    nets["actor_model"],
                    in_keys=["state", "belief"],
                    out_keys=["loc", "scale"],
                    spec=CompositeSpec(
                        **{
                            "loc": UnboundedContinuousTensorSpec(
                                proof_environment.action_spec.shape,
                            ),
                            "scale": UnboundedContinuousTensorSpec(
                                proof_environment.action_spec.shape,
                            ),
                        }
                    ),
                ),
                SafeProbabilisticModule(
                    in_keys=["loc", "scale"],
                    out_keys=[action_key],
                    default_interaction_type=InteractionType.MODE,
                    distribution_class=TanhNormal,
                    distribution_kwargs={"tanh_loc": True},
                    spec=CompositeSpec(
                        **{action_key: proof_environment.action_spec.to("cpu")}
                    ),
                ),
            ),
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", action_key],
                out_keys=["_", "_", "_", ("next", "belief")], # we don't need the prior state
            ),
        )
        
        actor_realworld = AdditiveGaussianWrapper(
            actor_realworld,
            sigma_init=1.0,
            sigma_end=1.0,
            annealing_num_steps=1,
            mean=0.0,
            std=exploration_noise,
        )
        
        # TODO: other exploration strategies
        
        return actor_realworld


    def _dreamer_make_mbenv(self, test_env, use_decoder_in_env: bool = False, state_dim: int = 30, rssm_hidden_dim: int = 200):
        nets = self.networks
        observation_out_key = self.keys["observation_out_key"]
        
        # MB environment
        if use_decoder_in_env:
            mb_env_obs_decoder = SafeModule(
                nets["decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", observation_out_key)],
            )
        else:
            mb_env_obs_decoder = None

        transition_model = SafeSequential(
            SafeModule(
                nets["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=["_", "_", "state", "belief"],
            ),
        )

        reward_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["reward_model"],
                in_keys=["state", "belief"],
                out_keys=["loc"],
            ),
            SafeProbabilisticModule(
                in_keys=["loc"],
                out_keys=["reward"],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": 1},
            ),
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