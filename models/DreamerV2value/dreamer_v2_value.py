# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from tensordict.nn import InteractionType

from torchrl.envs import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, check_env_specs
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
    WorldModelWrapper,
)
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from common.distributions import TruncNormalDist

from utils import _make_env, transform_env, get_activation
from ..DreamerV2.modules import (
    RSSMPriorV2, 
    RSSMPosteriorV2,
    ObsEncoder,
    ObsDecoder,
    DreamerActorV2,
    RSSMRollout,
)
import copy


class DreamerV2value:
    
    def __init__(self, config, device):
        test_env = _make_env(config, device="cpu")
        test_env = transform_env(config, test_env)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = "action",
            value_key = "state_value",
        )
        self.slow_critic_update = 50   # TODO: make this configurable
        
        self.networks = self._init_networks(config, test_env)
        self.modules = self._init_modules(config, test_env, device)
        
    def update(self, step):
        with torch.no_grad():
            if 10*step % self.slow_critic_update == 0:
                self.networks["value_target"].load_state_dict(self.networks["value_model"].state_dict())
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_vars = config.networks.state_vars
        state_classes = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        
        value_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation)
        value_target = copy.deepcopy(value_model)
        return nn.ModuleDict(modules = dict(
                encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=1024, depth=2, num_cells=hidden_dim, activation_class=activation)), 
                decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
                rssm_prior = RSSMPriorV2(hidden_dim=rssm_dim, rnn_hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes, action_spec=action_spec),
                rssm_posterior = RSSMPosteriorV2(hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes),
                actor_model = DreamerActorV2(out_features=action_spec.shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation, std_min_val=0.1),
                value_model = value_model,
                value_target = value_target
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._make_world_model().to(device)
        actor_simulator = self._make_actor_sim(proof_environment=proof_env).to(device)
        actor_realworld = self._make_actor_real(proof_environment=proof_env).to(device)
        value_model  = self._make_value_model().to(device)
        model_based_env = self._make_mbenv(proof_env, config.networks.state_dim, config.networks.rssm_hidden_dim).to(device)
        
        
        # Initialize world model
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
            
        # Initialize model-based environment, actor_simulator, value_model
        check_env_specs(model_based_env)
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            tensordict = actor_simulator(tensordict)
            value_model(tensordict)
        
        return nn.ModuleDict(modules=dict(
            world_model = world_model,
            model_based_env = model_based_env,
            actor_simulator = actor_simulator,
            value_model = value_model,
            actor_realworld = actor_realworld
        ))
        


    def _make_value_model(self):

        value_model = SafeModule(
            self.networks["value_model"],
            in_keys=["state", "belief"],
            out_keys=[self.keys["value_key"]]
        )

        return value_model


    def _make_actor_sim(self, proof_environment):
        
        actor_simulator = SafeProbabilisticTensorDictSequential(
            SafeModule(
                self.networks["actor_model"],
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
                out_keys=[self.keys["action_key"]],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TruncNormalDist,
                distribution_kwargs={"a": -1, "b": 1},
                spec=CompositeSpec(**{self.keys["action_key"]: proof_environment.action_spec}),
            )
        )
        return actor_simulator


    def _make_actor_real(self, proof_environment):
        # actor for real world: interacts with states ~ posterior
        # Out actor differs from the original paper where first they compute prior and posterior and then act on it
        # but we found that this approach worked better.
        actor_realworld = SafeSequential(
            SafeModule(
                self.networks["encoder"],
                in_keys=[self.keys["observation_in_key"]],
                out_keys=["encoded_latents"],
            ),
            SafeModule(
                self.networks["rssm_posterior"],
                in_keys=["belief", "encoded_latents"],
                out_keys=[ "_", "state",],
            ),
            SafeProbabilisticTensorDictSequential(
                SafeModule(
                    self.networks["actor_model"],
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
                    out_keys=[self.keys["action_key"]],
                    default_interaction_type=InteractionType.RANDOM,
                    distribution_class=TruncNormalDist,
                    distribution_kwargs={"a": -1, "b": 1},
                    spec=CompositeSpec(
                        **{self.keys["action_key"]: proof_environment.action_spec.to("cpu")}
                    ),
                ),
            ),
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", self.keys["action_key"]],
                out_keys=["_", "_", ("next", "belief")], # we don't need the prior state
            ),
        )
        
        return actor_realworld
    


    
    def _make_mbenv(self, test_env, state_dim: int = 30, rssm_hidden_dim: int = 200):
        
        mb_env_obs_decoder = SafeModule(
            self.networks["decoder"],
            in_keys=["state", "belief"],
            out_keys=[self.keys["observation_out_key"]],
        )

        transition_model = SafeSequential(
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=["_", "state", "belief"],
            ),
        )

        value_model = SafeModule(
                self.networks["value_target"],
                in_keys=["state", "belief"],
                out_keys=["reward"],
        )

        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(
                transition_model,
                value_model,
            ),
            prior_shape=torch.Size([state_dim]),
            belief_shape=torch.Size([rssm_hidden_dim]),
            obs_decoder=mb_env_obs_decoder,
        )

        model_based_env.set_specs_from_env(test_env)
        
        return model_based_env


    def _make_world_model(self):
        
        rssm_rollout = RSSMRollout(
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=[("next", "prior_logits"), "_", ("next", "belief")],
            ),
            SafeModule(
                self.networks["rssm_posterior"],
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )
        
        decoder = SafeModule(
            self.networks["decoder"],
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", self.keys["observation_out_key"])],
        )

        transition_model = SafeSequential(
            SafeModule(
                self.networks["encoder"],
                in_keys=[("next", self.keys["observation_in_key"])],
                out_keys=[("next", "encoded_latents")],
            ),
            rssm_rollout,
            decoder,
        )

        value_model = SafeModule(
            self.networks["value_model"],
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", self.keys["value_key"])],
        )
        
        world_model = WorldModelWrapper(
            transition_model,
            value_model,
        )
        
        return world_model

