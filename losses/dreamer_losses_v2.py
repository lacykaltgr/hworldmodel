# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    default_value_kwargs,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from losses import loss
from torchrl.modules.distributions import OneHotCategorical, ReparamGradientStrategy as Repa
from torch.distributions import Independent


@loss(
    name="loss_world_model", 
    keys=["loss_model_kl", "loss_model_reco", "loss_model_reward"]
)
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
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        stoch_classes: int = 32,
        stoch_dims: int = 32,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
    ):
        super().__init__()
        self.world_model = world_model
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.stoch_classes = stoch_classes
        self.stoch_dims = stoch_dims

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.clone(recurse=False)
        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )
        tensordict = self.world_model(tensordict)
        
        # compute model loss
        
        # reconstruction loss
        decoder = self.world_model[0][-1]
        dist = decoder.get_dist(tensordict)
        true_pixels = tensordict.get(("next", self.tensor_keys.pixels))
        image_shape = true_pixels.shape[2:]
        reco_loss = (-dist.log_prob(
            tensordict.get(("next", self.tensor_keys.pixels))
        ) / np.prod(image_shape)).mean()
        
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
        
        kl_loss = self.kl_balance * kl_prior + (1 - self.kl_balance) * kl_post

        # reward loss
        reward_model = self.world_model[1]
        dist = reward_model.get_dist(tensordict)
        reward_loss = -dist.log_prob(
            tensordict.get(("next", self.tensor_keys.true_reward))
        ).mean()
        
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

@loss(name="loss_actor")
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
        discount_loss: bool = True,  # for consistency with paper
        entropy_regularization: float = 1e-5,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.value_model = value_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.entropy_regularization = entropy_regularization

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        with torch.no_grad():
            tensordict = tensordict.select("state", self.tensor_keys.belief)
            tensordict = tensordict.reshape(-1)

        # TODO: do we need exploration here?
        with hold_out_net(self.model_based_env), set_exploration_type(ExplorationType.MEAN):
            tensordict = self.model_based_env.reset(tensordict.copy())
            fake_data = self.model_based_env.rollout(
                max_steps=self.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )

            next_tensordict = step_mdp(fake_data, keep_other=True)
            
            with hold_out_net(self.value_model):
                next_tensordict = self.value_model(next_tensordict)

        reward = fake_data.get(("next", self.tensor_keys.reward))
        next_value = next_tensordict.get(self.tensor_keys.value)
        lambda_target = self.lambda_target(reward, next_value)
        fake_data.set("lambda_target", lambda_target)

        if self.discount_loss:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            discount = gamma.expand(lambda_target.shape).clone()
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            actor_loss = -(lambda_target * discount).mean()
        else:
            actor_loss = -lambda_target.mean()
        
        # entropy regularization  
        if self.entropy_regularization > 0:
            dist = self.actor_model.get_dist(tensordict)
            entropy = dist.entropy().mean()
            actor_loss -= self.entropy_regularization * entropy
              
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
                vectorized=False,  # TODO: vectorized version seems not to be similar to the non vectoried
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
        }
        self._value_estimator.set_keys(**tensor_keys)

@loss(name="loss_value")
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
        
        td = fake_data.select(("next", "state"), ("next", "belief"))
        td = td.rename_key_(("next", "state"), "state")
        tensordict_select = td.rename_key_(("next", "belief"), "belief")
        dist = self.value_model.get_dist(tensordict_select)
        
        if self.discount_loss:
            discount = self.gamma * torch.ones_like(
                lambda_target, device=lambda_target.device
            )
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
            value_loss = -(discount * dist.log_prob(lambda_target).unsqueeze(-1)).mean()
        else:
            value_loss = -dist.log_prob(lambda_target).mean()

        loss_tensordict = TensorDict({"loss_value": value_loss}, [])
        return loss_tensordict, fake_data
