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
from common.norm import RewardEMA


@loss
class DreamerModelLoss(LossModule):
    """Dreamer Model Loss.

    Computes the loss of the dreamer world model. The loss is composed of the
    kl divergence between the prior and posterior of the RSSM,
    the reconstruction loss over the reconstructed observation and the reward
    loss over the predicted reward.

    Reference: https://arxiv.org/abs/1912.01603.

    Args:
        world_model (TensorDictModule): the world model.
        lambda_kl (float, optional): the weight of the kl divergence loss. Default: 1.0.
        lambda_reco (float, optional): the weight of the reconstruction loss. Default: 1.0.
        lambda_reward (float, optional): the weight of the reward loss. Default: 1.0.
        reco_loss (str, optional): the reconstruction loss. Default: "l2".
        reward_loss (str, optional): the reward loss. Default: "l2".
        free_nats (int, optional): the free nats. Default: 3.
        delayed_clamp (bool, optional): if ``True``, the KL clamping occurs after
            averaging. If False (default), the kl divergence is clamped to the
            free nats value first and then averaged.
        global_average (bool, optional): if ``True``, the losses will be averaged
            over all dimensions. Otherwise, a sum will be performed over all
            non-batch/time dimensions and an average over batch and time.
            Default: False.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            reward (NestedKey): The reward is expected to be in the tensordict
                key ("next", reward). Defaults to ``"reward"``.
            true_reward (NestedKey): The `true_reward` will be stored in the
                tensordict key ("next", true_reward). Defaults to ``"true_reward"``.
            prior_mean (NestedKey): The prior mean is expected to be in the
                tensordict key ("next", prior_mean). Defaults to ``"prior_mean"``.
            prior_std (NestedKey): The prior mean is expected to be in the
                tensordict key ("next", prior_mean). Defaults to ``"prior_mean"``.
            posterior_mean (NestedKey): The posterior mean is expected to be in
                the tensordict key ("next", prior_mean). Defaults to ``"posterior_mean"``.
            posterior_std (NestedKey): The posterior std is expected to be in
                the tensordict key ("next", prior_mean). Defaults to ``"posterior_std"``.
            pixels (NestedKey): The pixels is expected to be in the tensordict key ("next", pixels).
                Defaults to ``"pixels"``.
            reco_pixels (NestedKey): The reconstruction pixels is expected to be
                in the tensordict key ("next", reco_pixels). Defaults to ``"reco_pixels"``.
        """

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
        tensordict = tensordict.clone(recurse=False)
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
        
        #reco_loss = distance_loss(
        #    tensordict.get(("next", self.tensor_keys.pixels)),
        #    tensordict.get(("next", self.tensor_keys.reco_pixels)),
        #    self.reco_loss,
        #)
        #if not self.global_average:
        #    reco_loss = reco_loss.sum((-3, -2, -1))
        reco_dist = torchrl.modules.IndependentNormal(
            loc=tensordict.get(("next", self.tensor_keys.reco_pixels)),
            scale=1,
            event_dim=3,
        )
        reco_loss = -reco_dist.log_prob(tensordict.get(("next", self.tensor_keys.pixels)))
        reco_loss = reco_loss.mean().unsqueeze(-1)

        reward_dist = self.reward_model.get_dist(tensordict)
        reward_loss = -reward_dist.log_prob(tensordict.get(("next", self.tensor_keys.true_reward)))
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
            tensordict.detach(),
        )

    def kl_loss(
        self,
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        #TODO: elvileg nem kell
        #prior_logits = prior_logits.view(-1, self.stoch_dims, self.stoch_classes)
        #posterior_logits = posterior_logits.view(-1, self.stoch_dims, self.stoch_classes)
        
        dist_prior = self.get_distribution(prior_logits)
        dist_post = self.get_distribution(posterior_logits)
        kl = torch.distributions.kl.kl_divergence(dist_post, dist_prior)
        return kl

    def get_distribution(self, logits):
        dist = Independent(OneHotCategorical(logits=logits, grad_method=Repa.PassThrough), 1)
        return dist
    
    
    

@loss
class DreamerActorLoss(LossModule):
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
    default_value_estimator = ValueEstimators.TDLambda

    def __init__(
        self,
        actor_model: TensorDictModule,
        value_model: TensorDictModule,
        model_based_env: DreamerEnv,
        *,
        imagination_horizon: int = 15,
        discount_loss: bool = False,  # for consistency with paper
        gamma: int = 0.99,
        lmbda: int = 0.95,
        policy_ent_scale: float = 1e-4,
        imag_gradient: str = "dynamics",
        imag_gradient_mix: float = 0.5, # only used if imag_gradient is "both"
    ):
        super().__init__()
        self.actor_model = actor_model
        self.__dict__["value_model"] = value_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.policy_ent_scale = policy_ent_scale
        self.imag_gradient = imag_gradient 
        self.imag_gradient_mix = imag_gradient_mix
        
        self.reward_ema = RewardEMA(actor_model.device)
        self.register_buffer("ema_vals", torch.zeros((2,), device=actor_model.device))
        self.gamma = gamma
        self.lmbda = lmbda

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        tensordict = tensordict.select("state", self.tensor_keys.belief).detach()
        tensordict = tensordict.reshape(-1)
        
        with hold_out_net(self.model_based_env):
            tensordict = self.model_based_env.reset(tensordict.copy())
            fake_data = self.model_based_env.rollout(
                max_steps=self.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )

        with hold_out_net(self.value_model):
            fake_data = self.value_model(fake_data)
            next_tensordict = step_mdp(fake_data, keep_other=True)
            next_tensordict = self.value_model(next_tensordict)

        # calculate target for lambda return
        reward = fake_data.get(("next", self.tensor_keys.reward))
        value = fake_data.get((self.tensor_keys.value))
        next_value = next_tensordict.get(self.tensor_keys.value)
        lambda_target = self.lambda_target(reward, next_value)
        fake_data.set("lambda_target", lambda_target)
        policy = self.actor_model.get_dist(fake_data)
        action = fake_data.get("action")

        # calculate entropy for actions
        entropy_loss = self.policy_ent_scale * policy.entropy.sum(-1).unsqueeze(-1)
        
        # calculate loss for actions
        offset, scale = self.reward_ema(lambda_target, self.ema_vals)
        normed_target = (lambda_target - offset) / scale
        normed_base = (value - offset) / scale
        adv = normed_target - normed_base
        
        if self.imag_gradient == "dynamics":
            actor_target = adv
        elif self.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(action)[:-1][:, :, None]
                * (lambda_target - value).detach()
            )
        elif self.imag_gradient == "both":
            actor_target = (
                policy.log_prob(action)[:-1][:, :, None]
                * (lambda_target - value).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * lambda_target + (1 - mix) * actor_target
        else:
            raise NotImplementedError(self._config.imag_gradient)
        
        if self.discount_loss:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            discount = gamma.expand(actor_target.shape).clone()
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            actor_loss = -(actor_target * discount)
        else:
            actor_loss = -actor_target
        
        actor_loss -= entropy_loss
        actor_loss = actor_loss.sum((-1, -2)).mean()
        
        loss_tensordict = TensorDict({"loss_actor": actor_loss}, [])
        return loss_tensordict, fake_data.detach()

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
class DreamerValueLoss(LossModule):
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

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: Optional[str] = None,
        discount_loss: bool = True,  # for consistency with paper
        gamma: int = 0.99,
    ):
        super().__init__()
        self.value_model = value_model
        self.value_loss = value_loss if value_loss is not None else "l2"
        self.gamma = gamma
        self.discount_loss = discount_loss

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, fake_data) -> torch.Tensor:
        lambda_target = fake_data.get("lambda_target").detach()
        tensordict_select = fake_data.select(*self.value_model.in_keys, strict=False)
        
        self.value_model(tensordict_select)
        if self.discount_loss:
            discount = self.gamma * torch.ones_like(
                lambda_target, device=lambda_target.device
            )
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            
            value_loss = (
                0.5 * (
                    discount
                    * distance_loss(
                        tensordict_select.get(self.tensor_keys.value),
                        lambda_target,
                        self.value_loss,
                    )
                )
                .sum((-1, -2))
                .mean()
            )
        else:
            value_loss = 0.5 * (
                distance_loss(
                    tensordict_select.get(self.tensor_keys.value),
                    lambda_target,
                    self.value_loss,
                )
                .sum((-1, -2))
                .mean()
            )
        
        loss_tensordict = TensorDict({"loss_value": value_loss}, [])
        return loss_tensordict, fake_data
