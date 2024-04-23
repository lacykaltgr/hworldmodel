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
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper

from functional.utils import *
from losses.dreamer_losses import *
from . import ArchitectureConfig
from modules.discrete_wm import *
from losses.discrete_wm_loss import ModelLoss
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
        
    def update(self, step):
        if step % 100 == 0:
            self.networks["encoder"].update_target()
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        state_dim = config.networks.state_dim
    
        return dict(
            encoder = ObsEncoderWithTarget(1024, state_dim),
            decoder = ObsDecoder0(),
            transition = Predictor('gru', action_spec.shape[0], state_dim),
            reward_model = MLP(out_features=1, depth=3, num_cells=config.networks.state_dim, activation_class=activation)
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = GRURollout(
            transition_model = SafeModule(
                self.networks["transition"],
                in_keys=["state", "action"],
                out_keys=[("next", "belief")],
            ),
            encoder = SafeModule(
                self.networks["encoder"].encoder,
                in_keys=[("next", "pixels"), ("next", "belief")],
                out_keys=[("next", "state"), "_"],
            ),
        )
        
        decoder = SafeModule(
                self.networks["decoder"],
                in_keys=[("next", "state")],
                out_keys=[("next", "reco_pixels")]
        )

        target_encoder = SafeModule(
                self.networks["encoder"].target_encoder,
                in_keys=[("next", "pixels"), ("next", "belief")],
                out_keys=[("next", "state_target"), "_"]
        )

        reward_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                self.networks["reward_model"],
                in_keys=[("next", "state")],
                out_keys=[("next", "loc")],
            ),
            SafeProbabilisticModule(
                in_keys=[("next", "loc")],
                out_keys=[("next", "reward")],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": 1},
            ),
        )
            
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1)
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
        
        return dict(
            train_world_model = SafeSequential(world_model, decoder, target_encoder, reward_model).to(device),
            world_model = world_model.to(device),
            reward_model = reward_model.to(device),
            decoder = decoder.to(device),
            target_encoder = target_encoder.to(device),
            model_based_env = self._dreamer_make_mbenv(proof_env, config.networks.state_dim).to(device)
        )
        
    def _init_losses(self, config):
        
        losses = dict(
            world_model = ModelLoss(
                self.parts["train_world_model"],
                
            ).with_optimizer(params=self.parts["train_world_model"].parameters(), 
                             lr=config.optimization.world_model_lr, weight_decay=1e-5),
        )
        #if config.env.backend == "gym" or config.env.backend == "gymnasium":
        #    losses["world_model"].set_keys(pixels="observation", reco_pixels="reco_observation")
        return losses
    
    
    
    def _dreamer_make_mbenv(self, test_env, state_dim: int = 30):
        observation_out_key = self.keys["observation_out_key"]
    
        transition_model = SafeSequential(
            SafeModule(
                self.networks["transition"],
                in_keys=["state", "action"],
                out_keys=["state"],
            ),
        )

        reward_model = SafeModule(
                self.networks["reward_model"],
                in_keys=["state"],
                out_keys=["reward"],
        )
        
        mb_env_obs_decoder = SafeModule(
            self.networks["decoder"],
            in_keys=["state"],
            out_keys=[observation_out_key],
        )

        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(
                transition_model,
                reward_model,
            ),
            prior_shape=torch.Size([state_dim]),
            belief_shape=torch.Size([state_dim]),
            obs_decoder=mb_env_obs_decoder,
        )

        model_based_env.set_specs_from_env(test_env)
        return model_based_env
