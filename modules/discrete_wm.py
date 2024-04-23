
import copy
from dataclasses import dataclass
import warnings

import torch
from torch import nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from torchrl.envs.utils import step_mdp

from torchrl.modules.models.model_based import ObsEncoder, ObsDecoder


class GRURollout(TensorDictModuleBase):
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

    def __init__(self, encoder: TensorDictModule, transition_model: TensorDictModule):
        super().__init__()
        _module = TensorDictSequential(encoder, transition_model)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.encoder = encoder
        self.transition_model = transition_model

    def forward(self, tensordict):
        """Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

        The rollout requires a belief and posterior state primer.

        At each step, two probability distributions are built and sampled from:
        - A prior distribution p(s_{t+1} | s_t, a_t, b_t) where b_t is a
            deterministic transform of the form b_t(s_{t-1}, a_{t-1}). The
            previous state s_t is sampled according to the posterior
            distribution (see below), creating a chain of posterior-to-priors
            that accumulates evidence to compute a prior distribution over
            the current event distribution:
            p(s_{t+1} s_t | o_t, a_t, s_{t-1}, a_{t-1}) = p(s_{t+1} | s_t, a_t, b_t) q(s_t | b_t, o_t)

        - A posterior distribution of the form q(s_{t+1} | b_{t+1}, o_{t+1})
            which amends to q(s_{t+1} | s_t, a_t, o_{t+1})

        """
        tensordict_out = []
        *batch, time_steps = tensordict.shape

        #update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        update_values = tensordict.unbind(-1)
        _tensordict = update_values[0]
        for t in range(time_steps):
            # samples according to p(s_{t+1} | s_t, a_t, b_t)
            # ["state", "belief", "action"] -> [("next", "prior_mean"), ("next", "prior_std"), "_", ("next", "belief")]
            #with timeit("rollout/time-encoder"):
            self.encoder(_tensordict)

            # samples according to p(s_{t+1} | s_t, a_t, o_{t+1}) = p(s_t | b_t, o_t)
            # [("next", "belief"), ("next", "encoded_latents")] -> [("next", "posterior_mean"), ("next", "posterior_std"), ("next", "state")]
            #with timeit("rollout/time-transition-model"):
            self.transition_model(_tensordict)

            tensordict_out.append(_tensordict)
            if t < time_steps - 1:
                _tensordict = step_mdp(
                    _tensordict.select(*self.out_keys, strict=False), keep_other=False
                )
                _tensordict = update_values[t + 1].update(_tensordict)

        return torch.stack(tensordict_out, tensordict.ndim - 1)
    
    
    
class ObsDecoder0(nn.Module):
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
        
    def __init__(self, channels=32, num_layers=4, kernel_sizes=None, depth=None):
        if depth is not None:
            warnings.warn(
                f"The depth argument in {type(self)} will soon be deprecated and "
                f"used for the depth of the network instead. Please use channels "
                f"for the layer size and num_layers for the depth until depth "
                f"replaces num_layers."
            )
            channels = depth
        if num_layers < 1:
            raise RuntimeError("num_layers cannot be smaller than 1.")

        super().__init__()
        self.state_to_latent = nn.Sequential(
            nn.LazyLinear(channels * 8 * 2 * 2),
            nn.ReLU(),
        )
        if kernel_sizes is None and num_layers == 4:
            kernel_sizes = [5, 5, 6, 6]
        elif kernel_sizes is None:
            kernel_sizes = 5
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        layers = [
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 1, kernel_sizes[-1], stride=2),
        ]
        kernel_sizes = kernel_sizes[:-1]
        k = 1
        for j in range(1, num_layers):
            if j != num_layers - 1:
                layers = [
                    nn.ConvTranspose2d(
                        channels * k * 2, channels * k, kernel_sizes[-1], stride=2
                    ),
                ] + layers
                kernel_sizes = kernel_sizes[:-1]
                k = k * 2
                layers = [nn.ReLU()] + layers
            else:
                layers = [
                    nn.LazyConvTranspose2d(channels * k, kernel_sizes[-1], stride=2)
                ] + layers

        self.decoder = nn.Sequential(*layers)
        self._depth = channels
        
    def forward(self, encoded):
        latent = self.state_to_latent(encoded)
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D, 1, 1)
        obs_decoded = self.decoder(latent)
        _, C, H, W = obs_decoded.shape
        obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        return obs_decoded
    
class Predictor(nn.Module):
  """
  Predictor class
  Takes encoded observation (state) as input and tries to predict the next

  cell   default='gru'
  action_dim
  state_dim
  """

  def __init__(self, cell, action_dim, state_dim):
    super(Predictor, self).__init__()
    if cell == 'gru':
      self.cell = nn.GRUCell(action_dim, state_dim)
    else:
      raise NotImplementedError()


  def forward(self, state, action):
    predicted_state = self.cell(action, state)
    return predicted_state
    

class StatefulObsEncoder(nn.Module):
  """
  ObsEncoder class of torchrl extended with GRU memory cell
  """
  def __init__(self, encoded_dim, state_dim):
      super(StatefulObsEncoder, self).__init__()
      self.encoder = ObsEncoder()
      self.memory_cell = nn.GRUCell(encoded_dim, state_dim)
      self.encoded_dim = encoded_dim
      self.state_dim = state_dim

  def forward(self, observation, state):
      encoded = self.encoder(observation)
      encoded_shape = encoded.shape

      if len(encoded_shape) == 3:
        encoded = encoded.reshape(-1, self.encoded_dim)
        state = state.reshape(-1, self.state_dim)
        new_state = self.memory_cell(encoded, state)
        new_state = new_state.reshape((*encoded_shape[:-1], self.state_dim))
        encoded = encoded.reshape(encoded_shape)
      else:
        new_state = self.memory_cell(encoded, state)

      return new_state, encoded

class ObsEncoderWithTarget(nn.Module):
    
  """
  Observation encoder class wrapper
  Adds a target encoder, which is updated slowly
  """

  def __init__(self, encoded_dim, state_dim):
    super(ObsEncoderWithTarget, self).__init__()
    self.encoder = StatefulObsEncoder(encoded_dim, state_dim)
    self.target_encoder = copy.deepcopy(self.encoder)

    for param in self.target_encoder.parameters():
      param.requires_grad = False

  def forward(self, observation, state):
    return self.encoder(observation, state)

  def forward_target(self, observation, state):
    with torch.no_grad():
      return self.target_encoder(observation, state)

  def update_target(self, momentum: float = 0.99):
      with torch.no_grad():
          # use momentum to update the EMA encoder
          for param_q, param_k in zip(
              self.encoder.parameters(), self.target_encoder.parameters()
          ):
              param_k.data.mul_(momentum).add_((1.-momentum) * param_q.detach().data)
