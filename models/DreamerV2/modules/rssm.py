import torch
from packaging import version
from torch import nn

from torchrl.modules.distributions import OneHotCategorical, ReparamGradientStrategy as Repa
from torch.distributions import Independent
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule,
    TensorDictSequential
)
from torchrl.envs.utils import step_mdp


class RSSMPriorV2(nn.Module):
    """The prior network of the RSSM.

    This network takes as input the previous state and belief and the current action.
    It returns the next prior state and belief, as well as the parameters of the prior state distribution.
    State is by construction stochastic and belief is deterministic. In "Dream to control", these are called "deterministic state " and "stochastic state", respectively.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        action_spec (TensorSpec): Action spec.
        hidden_dim (int, optional): Number of hidden units in the linear network. Input size of the recurrent network.
            Defaults to 200.
        rnn_hidden_dim (int, optional): Number of hidden units in the recurrent network. Also size of the belief.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (float, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.


    """

    def __init__(self, action_spec, hidden_dim=600, rnn_hidden_dim=600, state_vars=32, state_classes=32):
        super().__init__()

        # Prior
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.action_state_projector = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ELU())
        self.rnn_to_prior_projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, state_vars * state_classes)
        )

        self.state_vars = state_vars
        self.state_classes = state_classes
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_shape = action_spec.shape
        self._unsqueeze_rnn_input = version.parse(torch.__version__) < version.parse("1.11")

    def forward(self, state, belief, action):
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        unsqueeze = False
        if self._unsqueeze_rnn_input and action_state.ndimension() == 1:
            if belief is not None:
                belief = belief.unsqueeze(0)
            action_state = action_state.unsqueeze(0)
            unsqueeze = True
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            belief = self.rnn(action_state, belief)
        if unsqueeze:
            belief = belief.squeeze(0)

        logits = self.rnn_to_prior_projector(belief)
        reshaped_logits = logits.view(-1, self.state_vars, self.state_classes)
        dist = self.get_distribution(reshaped_logits)
        state = dist.rsample()
        state = state.view(logits.shape)
        return logits, state, belief
    
    def get_distribution(self, state):
        dist = Independent(OneHotCategorical(logits=state, grad_method=Repa.PassThrough), 1)
        return dist


class RSSMPosteriorV2(nn.Module):
    """The posterior network of the RSSM.

    This network takes as input the belief and the associated encoded observation.
    It returns the parameters of the posterior as well as a state sampled according to this distribution.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        hidden_dim (int, optional): Number of hidden units in the linear network.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (float, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.

    """

    def __init__(self, hidden_dim=600, state_vars=32, state_classes=32):
        super(RSSMPosteriorV2, self).__init__()
        self.obs_rnn_to_post_projector = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_vars * state_classes),
        )
        self.hidden_dim = hidden_dim
        self.state_vars = state_vars
        self.state_classes = state_classes

    def forward(self, belief, obs_embedding):
        logits = self.obs_rnn_to_post_projector.forward(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        reshaped_logits = logits.view(-1, self.state_vars, self.state_classes)
        dist = self.get_distribution(reshaped_logits)
        state = dist.rsample()
        state = state.view(logits.shape)
        return logits, state
    
    def get_distribution(self, state):
        dist = Independent(OneHotCategorical(logits=state, grad_method=Repa.PassThrough), 1)
        return dist



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

    def __init__(self, rssm_prior: TensorDictModule, rssm_posterior: TensorDictModule):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.step_keys = [("next", "state"), ("next", "belief")]

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
        _tensordict = tensordict[..., 0]

        update_values = tensordict.exclude(*self.out_keys)
        for t in range(time_steps):
            # samples according to p(s_{t+1} | s_t, a_t, b_t)
            # ["state", "belief", "action"] -> [("next", "prior_mean"), ("next", "prior_std"), "_", ("next", "belief")]
            self.rssm_prior(_tensordict)

            # samples according to p(s_{t+1} | s_t, a_t, o_{t+1}) = p(s_t | b_t, o_t)
            # [("next", "belief"), ("next", "encoded_latents")] -> [("next", "posterior_mean"), ("next", "posterior_std"), ("next", "state")]
            self.rssm_posterior(_tensordict)

            tensordict_out.append(_tensordict)
            if t < time_steps - 1:
                _tensordict = step_mdp(
                    _tensordict.select(*self.step_keys, strict=False), keep_other=False
                )
                _tensordict = update_values[..., t + 1].update(_tensordict)

        return torch.stack(tensordict_out, tensordict.ndimension() - 1).contiguous()