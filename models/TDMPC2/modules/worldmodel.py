from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule,
    TensorDictSequential
)
from torchrl.envs.utils import step_mdp


class Encoder(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = layers.enc(cfg)
		self.apply(init.weight_init)
  
	
	def forward(self, obs):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)


class TaskEmbedder(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
		self.apply(init.weight_init)
  
	
	def forward(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		return obs


	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)


class Dynamics(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))


	def forward(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)


class LatentRollout(nn.Module):
    
    def __init__(self):
        super().__init__()
        


class RSSMRollout(TensorDictModuleBase):
    """Rollout the RSSM network.

    Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
    states and beliefs.
    The previous posterior is used as the prior for the next time step.
    The forward method returns a stack of all intermediate states and beliefs.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (TensorDictModule): Prior network.
        rssm_posterior (TensorDictModule): Posterior network.


    """

    def __init__(self, encoder: TensorDictModule, latent_rollout: TensorDictModule):
        super().__init__()
        _module = TensorDictSequential(encoder, latent_rollout)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.encoder = encoder
        self.latent_rollout = latent_rollout
        self.step_keys = [("next", "state")]

    def forward(self, tensordict):
        """
        Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

        # input tensordict: collected observations, actions, rewards, target states
        # encode the first observation, then rollout from that state

        """
        tensordict_out = []
        *batch, time_steps = tensordict.shape
        _tensordict = tensordict[..., 0]

        update_values = tensordict.exclude(*self.out_keys)
        self.encoder(_tensordict)
        
        for t in range(time_steps):
            self.latent_rollout(_tensordict)
            tensordict_out.append(_tensordict)
            if t < time_steps - 1:
                _tensordict = step_mdp(
                    _tensordict.select(*self.step_keys, strict=False), keep_other=False
                )
                _tensordict = update_values[..., t + 1].update(_tensordict)

        return torch.stack(tensordict_out, tensordict.ndimension() - 1).contiguous()
    
    
	

