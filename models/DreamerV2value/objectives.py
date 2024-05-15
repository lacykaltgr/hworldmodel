# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
import timeit

import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.objectives.common import LossModule

from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from common.bases.loss_base import loss
from torchrl.modules.distributions import OneHotCategorical, ReparamGradientStrategy as Repa
from torch.distributions import Independent
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    default_value_kwargs,
    distance_loss,
    # distance_loss,
    hold_out_net,
    ValueEstimators,
)
from torchrl.envs.utils import step_mdp
from torch.profiler import record_function


@loss
class DreamerModelLoss(LossModule):

    @dataclass
    class _AcceptedKeys:

        reward: NestedKey = "reward"
        prior: NestedKey = "prior_logits"
        posterior: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"
        value: NestedKey = "state_value"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TDLambda

    def __init__(
        self,
        world_model: TensorDictModule,
        *,
        lambda_kl: float = 2.0,
        lambda_reco: float = 1.0,
        lambda_value: float = 1.0,
        reco_loss: Optional[str] = None,
        value_loss: Optional[str] = None,
        stoch_classes: int = 32,
        stoch_dims: int = 32,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        global_average: bool = False,
        gamma: float = 0.99,
        lmda: float = 0.95,
        discount_loss: bool = True
    ):
        super().__init__()
        self.world_model = world_model
        
        self.reco_loss = reco_loss if reco_loss is not None else "l2"
        self.value_loss = value_loss if value_loss is not None else "l2"
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_value
        
        self.free_nats = free_nats
        self.global_average = global_average
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        
        self.stoch_classes = stoch_classes
        self.stoch_dims = stoch_dims
        
        self.gamma = gamma
        self.lmda = lmda
        self.discount_loss = discount_loss
        
        self.__dict__["decoder"] = self.world_model[0][-1]
        self.__dict__["value_model"] = self.world_model[1]


    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        
        tensordict = self.world_model(tensordict)
    
        # kl divergence loss
        kl_prior = self.kl_loss(
            tensordict.get(("next", self.tensor_keys.prior)),
            tensordict.get(("next", self.tensor_keys.posterior)).detach(),
        ).mean()
        kl_post = self.kl_loss(
            tensordict.get(("next", self.tensor_keys.prior)).detach(),
            tensordict.get(("next", self.tensor_keys.posterior)),
        ).mean()
        
        if self.free_nats > 0.0:
            kl_prior = kl_prior.clamp_min(min=self.free_nats)
            kl_post = kl_post.clamp_min(min=self.free_nats)

        kl_loss = (
            self.kl_balance * kl_prior + (1 - self.kl_balance) * kl_post
        ).unsqueeze(-1)
        
        reco_loss = 0.5 * distance_loss(
            tensordict.get(("next", self.tensor_keys.pixels)),
            tensordict.get(("next", self.tensor_keys.reco_pixels)),
            self.reco_loss,
        ) 
        if not self.global_average:
            reco_loss = reco_loss.sum((-3, -2, -1))
        reco_loss = reco_loss.mean().unsqueeze(-1)
        
        
        reward = tensordict.get(("next", self.tensor_keys.reward))
        value = tensordict.get(("next", self.tensor_keys.value))
        next_value = tensordict.get(("next", self.tensor_keys.value))
        
        with torch.no_grad():
            lambda_target = self.lambda_target(reward, next_value).detach()
        
        if self.discount_loss:
            discount = self.gamma * torch.ones_like(
                lambda_target, device=lambda_target.device
            )
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            
            value_loss = 0.5 * (
                discount * distance_loss(value, lambda_target, self.value_loss,)
                ).sum((-1, -2)).mean()
           
        else:
            value_loss = 0.5 * (
                distance_loss(value, lambda_target, self.value_loss)
                ).sum((-1, -2)).mean()

        return (
            TensorDict(
                {
                    "loss_model_kl": self.lambda_kl * kl_loss,
                    "loss_model_reco": self.lambda_reco * reco_loss,
                    "loss_model_value": self.lambda_reward * value_loss,
                },
                [],
            ),
            tensordict.detach(),
        )

    def kl_loss(
        self,
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        prior_logits = prior_logits.view(-1, self.stoch_dims, self.stoch_classes)
        posterior_logits = posterior_logits.view(-1, self.stoch_dims, self.stoch_classes)
        
        dist_prior = self.get_distribution(prior_logits)
        dist_post = self.get_distribution(posterior_logits)
        kl = torch.distributions.kl.kl_divergence(
            dist_post, dist_prior
        )
        return kl


    def get_distribution(self, logits):
        dist = Independent(OneHotCategorical(logits=logits, grad_method=Repa.PassThrough), 1)
        return dist
    
    
    def lambda_target(self, reward: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        done = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        terminated = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        input_tensordict = TensorDict(
            {
                ("next", self.tensor_keys.reward): reward,
                ("next", self.tensor_keys.value): value,
                ("next", self.tensor_keys.done): done,
                ("next", self.tensor_keys.terminated): terminated,
            },
            [],
        )
        return self.value_estimator.value_estimate(input_tensordict)

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        value_net = None
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
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
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
                vectorized=True,  # TODO: vectorized version seems not to be similar to the non vectorised
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
        }
        self._value_estimator.set_keys(**tensor_keys)
    
    
    

@loss
class DreamerActorLoss(LossModule):


    @dataclass
    class _AcceptedKeys:

        belief: NestedKey = "belief"
        value: NestedKey = "state_value"

    def __init__(
        self,
        actor_model: TensorDictModule,
        model_based_env: DreamerEnv,
        *,
        imagination_horizon: int = 15,
        discount_loss: bool = False,  # for consistency with paper
        gamma: int = 0.99,
        lmbda: int = 0.95,
        policy_ent_scale: float = 1e-4,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.policy_ent_scale = policy_ent_scale
        self.gamma = gamma
        self.lmbda = lmbda

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        tensordict = tensordict.select("state", self.tensor_keys.belief)
        tensordict = tensordict.reshape(-1)
        
        with hold_out_net(self.model_based_env):
            tensordict = self.model_based_env.reset(tensordict)
            fake_data = self.model_based_env.rollout(
                max_steps=self.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )

        lambda_target = fake_data.get(("next", "reward")) # mb_env uses value model to estimate "reward"

        entropy = self.actor_model.get_dist(fake_data).entropy.sum(-1).unsqueeze(-1)
        entropy_loss = self.policy_ent_scale * entropy
        
        if self.discount_loss:
            gamma = torch.tensor(self.gamma, device=tensordict.device)
            discount = gamma.expand(lambda_target.shape).clone()
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            actor_loss = -((lambda_target + entropy_loss) * discount).sum((-1, -2)).mean()
        else:
            actor_loss = -(lambda_target + entropy_loss).sum((-1, -2)).mean()
        
        loss_tensordict = TensorDict({"loss_actor": actor_loss}, [])
        return loss_tensordict, None
