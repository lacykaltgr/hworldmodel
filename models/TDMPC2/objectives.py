# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
import timeit

import torch
import torch.nn.functional as F
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
class tdmpcModelLoss(LossModule):
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
        global_average: bool = False,
        
        episode_length = None,
        discount_denom = None,
        discount_min = None,
        discount_max = None
    ):
        super().__init__()
        self.world_model = world_model
        self.reco_loss = reco_loss if reco_loss is not None else "l2"
        self.reward_loss = reward_loss if reward_loss is not None else "l2"
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        
        self.target_encoder = targer_encoder
        self.actor_sim = actor_sim
        self.Q = Q 
        self.Q_target = Q_target
        
        self.free_nats = free_nats
        self.global_average = global_average
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        
        self.stoch_classes = stoch_classes
        self.stoch_dims = stoch_dims
        
        
        self.discount_denom = discount_denom
        self.discount_min = discount_min
        self.discount_max = discount_max
        self.discount = torch.tensor(self._get_discount(episode_length), device='cuda')
        
        self.__dict__["decoder"] = self.world_model[0][-1]
        self.__dict__["reward_model"] = self.world_model[1]

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass
    
    def forward(self, tensordict: TensorDict):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        
        self.Q.track_q_grad(True)

        # Compute targets
        with torch.no_grad():
            #next_z = self.world_model.encode(obs[1:], task)
            #td_targets = self._td_target(next_z, reward, task)
            self.target_encoder(tensordict) # add "next_z" to tensordict
            self.actor_sim(tensordict) # add "policy" to tensordict
            self.Q_target(tensordict, return_type='min')
            td_target = tensordict.get("reward") + self.discount * tensordict.get("q_target")
            tensordict.update({
                "td_target" : td_target
            })

        # TODO: ittt tartok, érdemes objectives-el, felülről nyomni
        
        
        # Latent rollout
        #zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        #z = self.model.encode(obs[0], task)
        #zs[0] = z
        #consistency_loss = 0
        #for t in range(self.cfg.horizon):
        #    z = self.model.next(z, action[t], task)
        #    consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
        #    zs[t+1] = z
        
        # Predictions
        #_zs = zs[:-1]
        #qs = self.model.Q(_zs, action, task, return_type='all')
        #reward_preds = self.model.reward(_zs, action, task)
        
        self.world_model(tensordict)
        
        predicted_states = tensordict.get(("next", "state"))
        target_states = tensordict.get(("next", "target_state"))
        
        consistency_loss = F.mse_loss(predicted_states, target_states, reduce="none")
        # TODO: scale by decay factor """rho ** t""", then sum up
        

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
                
                
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))

        return (
            TensorDict(
                {
                    "loss_model_consistency": self.cfg.consistency_coef * consistency_loss,
                    "loss_model_reward": self.cfg.reward_coef * reward_loss,
                    "loss_model_value": self.cfg.value_coef * value_loss
                },
                [],
            ),
            tensordict
        )



    def get_distribution(self, logits):
        dist = Independent(OneHotCategorical(logits=logits, grad_method=Repa.PassThrough), 1)
        return dist
    
    
    def _get_discount(self, episode_length):
        """
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
        frac = episode_length/self.discount_denom
        return min(max((frac-1)/(frac), self.discount_min), self.discount_max)
    
    
    

@loss
class tdmpcPolicyLoss(LossModule):
    @dataclass
    class _AcceptedKeys:
        belief: NestedKey = "belief"
        reward: NestedKey = "reward"
        value: NestedKey = "state_value"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        
    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TDLambda

    def __init__(
        self,
        Q,
        policy_updater,
        scale
    ):
        super().__init__()
        self.Q = Q
        self.policy_updater = policy_updater
        self.scale = scale

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
            )

    def forward(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        
        self.Q.track_q_grad(False)
        
        tensordict = tensordict.select("state").detach()
        
        self.policy_updater(tensordict)
        
        qs = tensordict.get("qs")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        
        return (
            TensorDict(
                {
                    "loss_policy": (
                        (
                            self.cfg.entropy_coef * tensordict.get("log_policy") - qs
                        ).mean(dim=(1,2)
                        ) * rho
                        ).mean()
                },
                [],
            ),
            tensordict
        )
   