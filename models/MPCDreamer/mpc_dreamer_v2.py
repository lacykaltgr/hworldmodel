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
    gradMPCPlanner,
    PolicyPrior,
    QFunction
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
        
        return nn.ModuleDict(modules = dict(
                encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=1024, depth=2, num_cells=hidden_dim, activation_class=activation)), 
                decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
                rssm_prior = RSSMPriorV2(hidden_dim=rssm_dim, rnn_hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes, action_spec=action_spec),
                rssm_posterior = RSSMPosteriorV2(hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes),
                reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
                actor_model = PolicyPrior(out_features=action_spec.shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation, std_min_val=0.1),
                
                # TODO: add q-function specific values to config
                q_function = QFunction(latent_dim=rssm_dim, action_dim=action_spec.shape[-1], mlp_dim=hidden_dim, ),
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._make_world_model().to(device)
        model_based_env = self._make_mbenv(proof_env, config.networks.state_dim, config.networks.rssm_hidden_dim).to(device)
        value_model = self._make_value_model(config.networks.use_value_network).to(device)
        policy = self._make_actor_real(config, mb_env=model_based_env).to(device)
        
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
        
        return nn.ModuleDict(modules=dict(
            world_model = world_model,
            model_based_env = model_based_env,
            value_model = value_model,
            rl_policy = policy,
        ))
        
        
        
    def _make_world_model(self, config):
        
        encoder = SafeModule(
            self.networks["encoder"],
            in_keys=[("next", self.keys["observation_in_key"])],
            out_keys=[("next", "encoded_latents")],
        )
        
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
        
        reward_model = SafeModule(
            self.networks["reward_model"],
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reward")],
        )

        transition_model = SafeSequential(
            encoder,
            rssm_rollout,
            decoder
        )
    
        world_model = WorldModelWrapper(
            transition_model,
            reward_model
        )
        
        return world_model
    


    
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
    
    
    
    
    def _make_value_model(self, use_value_network: bool = False):
        # if no separeta value network is used, 
        # the reward model is used as value model
        
        value_model = SafeModule(
            self.networks["q_function"],
            in_keys=["action", "state", "belief"],
            out_keys=[self.keys["value_key"]]
        )
            
        return value_model



    def _make_actor_real(self, config, mb_env):
        # actor for real world: interacts with states ~ posterior
        # Out actor differs from the original paper where first they compute prior and posterior and then act on it
        # but we found that this approach worked better.
        
        value_estimator = TD0Estimator(
            gamma=config.planner.gamma,
            value_network=self.modules["value_model"],
            differentiable=True,
            value_key=self.keys["value_key"],
        )
        
        planner = gradMPCPlanner(
            env=mb_env,
            actor_module=self.networks["actor_model"],
            value_estimator=value_estimator,
            temperature=config.planner.temperature,
            planning_horizon=config.planner.planning_horizon,
            optim_steps=config.planner.optim_steps,
            num_candidates=config.planner.n_candidates,
            reward_key= ("next", "reward"),
            action_key= "action",
        )
        
        
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
