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
from torchrl.modules.planners import MPPIPlanner

from utils import _make_env, transform_env, get_activation
from .modules import (
    Encoder,
    Dynamics,
    Reward,
    QFunction,
    Policy,
    LatentRollout
)

from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.objectives.utils import default_value_kwargs, ValueEstimators
from torchrl.collectors import RandomPolicy
from .modules.world_model import WorldModel
from common.scale import RunningScale


class TDMPC2:
    
    def __init__(self, config, device):
        test_env = _make_env(config, device="cpu")
        test_env = transform_env(config, test_env)
        
        self.device = device

        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = "action",
            value_key = "state_value",
        )
        
        self.networks = self._init_networks(config, test_env)
        self.modules = self._init_modules(config, test_env, device)
        
    def update(self, step):
        with torch.no_grad():
            if 10*step % self.slow_critic_update == 0:
                print("Updating value target")
                self.modules["value_target"].load_state_dict(self.modules["value_model"].state_dict())
        
    def _init_networks(self, config, proof_env):
        
        return nn.ModuleDict(modules = dict(
                encoder = Encoder(config),
                dynamics = Dynamics(config),
                reward = Reward(config),
                qs = QFunction(config),
                policy_prior = Policy(config)
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model().to(device)
        value_model  = self._dreamer_make_value_model().to(device)
        model_based_env = self._dreamer_make_mbenv(
            proof_env, state_dim=config.networks.state_dim, rssm_hidden_dim=config.networks.rssm_hidden_dim
        ).to(device)
        value_estimator = self.make_value_estimator(value_net=value_model)
        planner = MPPIPlanner(
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
        q, q_target = self._make_q()
        policy_updater = self._make_policy_updater()
        scale = RunningScale()
        
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
            policy = policy,
            value_estimator = value_estimator,
            planner = planner,
            q=q,
            q_target=q_target,
            policy_update=policy_updater,
            scale = scale
        ))
        


    def _make_target_encoder(self):
        target_encoder = SafeSequential(
            SafeModule(
                self.networks["encoder"],
                in_keys=[("next", self.keys["observation_in_key"])],
                out_keys=[("next", "target_state")]
            )
        )
        return target_encoder
    
    
    def _make_actor_sim(self, proof_environment, planner):
        # actor for real world: interacts with states ~ posterior
        # Out actor differs from the original paper where first they compute prior and posterior and then act on it
        # but we found that this approach worked better.
        actor_sim = SafeModule(
            self.networks["policy_prior"],
            in_keys=[("next", "embedding"), "reward"],
            out_keys=["_", "policy", "_", "_"]
        )
        return actor_sim
    
    
    def _make_q(self):
        q = SafeModule(
            self.networks["qs"],
            in_keys=[("next", "state"), "policy"],
            out_keys=["q_value"],
        )
        
        q_target = SafeModule(
            self.networks["qs"].q_target,
            in_keys=[("next", "state"), "policy"],
            out_keys=["q_target"],
        )
        return q, q_target
    
    
    def _make_world_model(self):
        
        latent_rollout = LatentRollout(
            SafeModule(
                self.networks["encoder"],
                in_keys=[self.keys["observation_in_key"]],
                out_keys=["state"],
            ),
            SafeModule(
                self.networks["dynamics"],
                in_keys=["state", "action"],
                out_keys=[("next", "state")],
            )
        )
        
        reward_predictor = SafeSequential(
            q_function = SafeModule(
                self.networks["qs"].return_with("all"),
                in_keys=["state", "policy"],
                out_keys=["q_value"],  
            ),

            reward_model = SafeModule(
                self.networks["reward"],
                in_keys=["state", "policy"],
                out_keys=["reward_preds"],
            ),
            
        )
        
        world_model = WorldModelWrapper(
            latent_rollout,
            reward_predictor,
        )
        
        return world_model
        
    
    def _make_policy_updater(self):
        
        updater = SafeSequential(
            actor = SafeModule(
                self.networks["policy_prior"],
                in_keys=["state"],
                out_keys=["_", "policy", "log_policy", "_"]
            ),
        
            q = SafeModule(
                self.networks["qs"].return_with("avg"),
                in_keys=["state", "policy"],
                out_keys=["qs"]
            )
        )
        return updater