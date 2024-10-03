# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from torchrl.envs import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, check_env_specs
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeSequential,
    WorldModelWrapper,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from common.distributions import TruncNormalDist

from utils import _make_env, transform_env, get_activation
from .modules import (
    RSSMPriorV2, 
    RSSMPosteriorV2,
    ObsEncoder,
    ObsDecoder,
    RSSMRollout,
    gradMPCPlanner
)

from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.objectives.utils import default_value_kwargs, ValueEstimators


class MPCDreamerV2:
    
    def __init__(self, config, device):
        test_env = _make_env(config, device="cpu")
        test_env = transform_env(config, test_env)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = "action",
            value_key = "state_value",
        )
        
        self.networks = self._init_networks(config, test_env)
        self.modules = self._init_modules(config, test_env, device)
        
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_vars = config.networks.state_vars
        state_classes = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        
        networks = nn.ModuleDict(modules = dict(
                encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=1024, depth=2, num_cells=hidden_dim, activation_class=activation)), 
                decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
                rssm_prior = RSSMPriorV2(hidden_dim=rssm_dim, rnn_hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes, action_spec=action_spec),
                rssm_posterior = RSSMPosteriorV2(hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes),  
            )
        )
        
        if config.networks.value_estimator == "reward":
            networks.update(dict(reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),))
        elif config.networks.value_estimator == "value":
            networks.update(dict(value_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation)))
        elif config.networks.value_estimator == "both":
            networks.update(dict(
                reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
                value_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation
            )))
        else:
            raise ValueError(f"Unknown value estimator description {config.networks.value_estimator}")
        
        return networks
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._make_world_model().to(device)
        model_based_env = self._make_mbenv(proof_env, config.networks.state_dim, config.networks.rssm_hidden_dim).to(device)
        value_model, value_estimator = self._make_value_model(config.networks.use_value_network).to(device)
        
        
        planner = gradMPCPlanner(
            env=model_based_env,
            advantage_module=value_estimator,
            temperature=config.planner.temperature,
            planning_horizon=config.planner.planning_horizon,
            optim_steps=config.planner.optim_steps,
            num_candidates=config.planner.n_candidates,
            top_k=config.planner.top_k,
            reward_key= ("next", "reward"),
            action_key= "action",
        ).to(device)
        policy = self._dreamer_make_actor_real(proof_env, planner)
        
        # Initialize world model
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
            
        # Initialize model-based environment, actor_simulator, value_model
        check_env_specs(model_based_env)
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            value_model(tensordict)
        
        modules = nn.ModuleDict(modules=dict(
            world_model = world_model,
            model_based_env = model_based_env,
            value_model = value_model,
            rl_policy = policy,
            planner = planner,
        ))
        


    def _make_value_model(self, use_value_network: bool = False):
        # if no separeta value network is used, 
        # the reward model is used as value model
        
        if use_value_network:
            value_model = SafeModule(
                self.networks["value_model"],
                in_keys=["state", "belief"],
                out_keys=[self.keys["value_key"]]
            )
            value_estimator = self.make_value_estimator(value_net=value_model)
        else:
            value_model = SafeModule(
                self.networks["reward_model"],
                in_keys=["state", "belief"],
                out_keys=[self.keys["value_key"]]
            )
            
            # this would use the reward as both reward and value
            # value_estimator = self.make_value_estimator(value_net=value_model)
            
            value_estimator = None
        return value_model, value_estimator



    def _make_actor_real(self, proof_environment, planner):
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
            planner,
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", self.keys["action_key"]],
                out_keys=["_", "_", ("next", "belief")], # we don't need the prior state
            ),
        )
        return actor_realworld
    


    
    def _dreamer_make_mbenv(self, test_env, state_dim: int = 30, rssm_hidden_dim: int = 200):
        

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

        reward_model = SafeModule(
                self.networks["reward_model"],
                in_keys=["state", "belief"],
                out_keys=["reward"],
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


    def _dreamer_make_world_model(self, config):
        
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
        
        if config.networks.value_estimator == "reward" or config.networks.value_estimator == "both":
            reward_model = SafeModule(
                self.networks["reward_model"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reward")],
            )
        elif config.networks.value_estimator == "value":
            reward_model = SafeModule(
                self.networks["value_model"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "value")],
            )
        
        world_model = WorldModelWrapper(
            transition_model,
            reward_model,
        )
        
        return world_model
    

    def make_value_estimator(self, value_net, **hyperparams):
        value_type = ValueEstimators.TDLambda
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            _value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.TD0:
            _value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.GAE:
            if hasattr(self, "lmbda"):
                hp["lmbda"] = self.lmbda
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            if hasattr(self, "lmbda"):
                hp["lmbda"] = self.lmbda
            _value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
                vectorized=True,  # TODO: vectorized version seems not to be similar to the non vectorised
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.keys["value_key"],
            "value_target": "value_target",
        }
        _value_estimator.set_keys(**tensor_keys)
        return _value_estimator

