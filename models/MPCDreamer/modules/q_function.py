from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import init
from common import layers
from common import math

class QFunction(nn.Module):
	def __init__(self, latent_dim, action_dim, mlp_dim, num_q, tau, num_bins, v_min, v_max, dropout):	
		super().__init__()
  
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.mlp_dim = mlp_dim
		self.num_q = num_q
		self.tau = tau
		self.num_bins = num_bins
		self.v_min = v_min
		self.v_max = v_max
		self.dropout = dropout
  
		self._Qs = layers.Ensemble(
			[
				layers.mlp(latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), dropout=dropout)
				for _ in range(num_q)
			]
		)
		print("QFunction parameters before applying init.weight_init")
		for param in self.parameters():
			print(param)
		print("QFunction params before applying zero init")
		for param in self._Qs.params:
			print(param)
    
		self.apply(init.weight_init)
		init.zero_([self._Qs.params[-2]])

		self.q_target = self.QTarget(self._Qs)
		self.convert_to_two_hot = lambda x: math.two_hot_inv(x, num_bins, v_min, v_max)
		self.return_type = "all"
		

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self.q_target._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.tau)
    

	def return_with(self, return_type):
		assert self.return_type in {'min', 'avg', 'all'}
		self.return_type = return_type
		return self
	

	def forward(self, a, *z):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert self.return_type in {'min', 'avg', 'all'}

		if len(z) > 1:
			z = (torch.cat([*z], -1),)
			
		z = torch.cat([z, a], dim=-1)
		out = self._Qs(z)

		if self.return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = self.convert_to_two_hot(Q1), self.convert_to_two_hot(Q2)
		return torch.min(Q1, Q2) if self.return_type == 'min' else (Q1 + Q2) / 2