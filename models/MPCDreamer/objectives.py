# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

import torch
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



@loss
class ModelLoss(LossModule):

    @dataclass
    class _AcceptedKeys:

        reward: NestedKey = "reward"
        true_reward: NestedKey = "true_reward"
        prior: NestedKey = "prior_logits"
        posterior: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        world_model: TensorDictModule,
        *,
        lambda_kl: float = 2.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        reco_loss: Optional[str] = None,
        reward_loss: Optional[str] = None,
        stoch_classes: int = 32,
        stoch_dims: int = 32,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        global_average: bool = False
    ):
        super().__init__()
        self.world_model = world_model
        self.reco_loss = reco_loss if reco_loss is not None else "l2"
        self.reward_loss = reward_loss if reward_loss is not None else "l2"
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        
        self.free_nats = free_nats
        self.global_average = global_average
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        
        self.stoch_classes = stoch_classes
        self.stoch_dims = stoch_dims
        
        self.__dict__["decoder"] = self.world_model[0][-1]
        self.__dict__["reward_model"] = self.world_model[1]

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )
        
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

        reward_loss = distance_loss(
            tensordict.get(("next", self.tensor_keys.true_reward)),
            tensordict.get(("next", self.tensor_keys.reward)),
            self.reward_loss,
        )
        if not self.global_average:
            reward_loss = reward_loss.squeeze(-1)
        reward_loss = reward_loss.mean().unsqueeze(-1)

        return (
            TensorDict(
                {
                    "loss_model_kl": self.lambda_kl * kl_loss,
                    "loss_model_reco": self.lambda_reco * reco_loss,
                    "loss_model_reward": self.lambda_reward * reward_loss,
                },
                [],
            ),
            tensordict,
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
    
    
  
@loss
class ActorLoss(LossModule):
    """Dreamer Actor Loss.

    Computes the loss of the dreamer actor. The actor loss is computed as the
    negative average lambda return.

    Reference: https://arxiv.org/abs/1912.01603.

    Args:
        actor_model (TensorDictModule): the actor model.
        value_model (TensorDictModule): the value model.
        model_based_env (DreamerEnv): the model based environment.
        imagination_horizon (int, optional): The number of steps to unroll the
            model. Defaults to ``15``.
        discount_loss (bool, optional): if ``True``, the loss is discounted with a
            gamma discount factor. Default to ``False``.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            belief (NestedKey): The input tensordict key where the belief is expected.
                Defaults to ``"belief"``.
            reward (NestedKey): The reward is expected to be in the tensordict key ("next", reward).
                Defaults to ``"reward"``.
            value (NestedKey): The reward is expected to be in the tensordict key ("next", value).
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            done (NestedKey): The input tensordict key where the flag if a
                trajectory is done is expected ("next", done). Defaults to ``"done"``.
            terminated (NestedKey): The input tensordict key where the flag if a
                trajectory is terminated is expected ("next", terminated). Defaults to ``"terminated"``.
        """

        belief: NestedKey = "belief"
        reward: NestedKey = "reward"
        value: NestedKey = "state_value"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        actor_model: TensorDictModule,
        value_model: TensorDictModule,
        model_based_env: DreamerEnv,
        *,
        imagination_horizon: int = 15,
        gamma: int = 0.99,
        policy_ent_scale: float = 1e-4
    ):
        super().__init__()
        self.actor_model = actor_model
        self.value_model = value_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.policy_ent_scale = policy_ent_scale
        self.gamma = gamma

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        tensordict = tensordict.select("state", self.tensor_keys.belief).detach()
        tensordict = tensordict.reshape(-1)
        
        with hold_out_net(self.model_based_env):
            tensordict = self.model_based_env.reset(tensordict)
            fake_data = self.model_based_env.rollout(
                max_steps=self.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )

        # add "value"
        self.value_model(fake_data)     # Q(z_t, a_t) ...
        
        # TODO: could use td value estimator here too
        td_target = fake_data.get("value")

        entropy = self.actor_model.get_dist(fake_data).entropy.sum(-1).unsqueeze(-1)
        entropy_loss = self.policy_ent_scale * entropy
        
        gamma = self.gamma.to(tensordict.device)
        discount = gamma.expand(td_target.shape).clone()
        discount[..., 0, :] = 1
        discount = discount.cumprod(dim=-2)
        actor_loss = -((td_target + entropy_loss) * discount).sum((-1, -2)).mean()
        
        loss_tensordict = TensorDict({"loss_actor": actor_loss}, [])
        return loss_tensordict, fake_data

        
        
@loss
class ValueLoss(LossModule):
    """Dreamer Value Loss.

    Computes the loss of the dreamer value model. The value loss is computed
    between the predicted value and the lambda target.

    Reference: https://arxiv.org/abs/1912.01603.

    Args:
        value_model (TensorDictModule): the value model.
        value_loss (str, optional): the loss to use for the value loss.
            Default: ``"l2"``.
        discount_loss (bool, optional): if ``True``, the loss is discounted with a
            gamma discount factor. Default: False.
        gamma (float, optional): the gamma discount factor. Default: ``0.99``.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            value (NestedKey): The input tensordict key where the state value is expected.
                Defaults to ``"state_value"``.
        """

        value: NestedKey = "state_value"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: Optional[str] = None,
        discount_loss: bool = True,  # for consistency with paper
        gamma: int = 0.99,
    ):
        super().__init__()
        self.value_model = value_model
        
        self.convert_to_functional(
            self.value_model,
            "value",
            create_target_params=True
        )
        
        self.value_loss = value_loss if value_loss is not None else "l2"
        self.gamma = gamma
        self.discount_loss = discount_loss

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def forward(self, fake_data) -> torch.Tensor:
        
        # add "value"
        self.value_model(fake_data)
        value = fake_data.get(self.tensor_keys.value)
        
        # add "value_target"
        # # r_t + gamma * V(z_{t+1}, a_{t+1})
        td_target = self.value_estimator.value_estimate(fake_data, target_params=self.target_value_params)  
        

        value_loss = 0.5 * (
            distance_loss(
                value,
                td_target,
                self.value_loss,
            )
            .sum((-1, -2))
            .mean()
        )
        
        loss_tensordict = TensorDict({"loss_value": value_loss}, [])
        return loss_tensordict, fake_data
    
    
    def make_value_estimator(self, value_type: ValueEstimators, **hyperparams):
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(value_network=self.value_model, **hp)
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(value_network=self.value_model, **hp)
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(value_network=self.value_model, **hp)
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
        self._value_estimator.set_keys({
            "value": self.tensor_keys.value,
            "value_target": "value_target",
        })
