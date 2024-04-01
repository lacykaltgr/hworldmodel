import torch
from packaging import version
from torch import nn

from torchrl.modules.distributions import OneHotCategorical, ReparamGradientStrategy as Repa

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

    def __init__(
        self,
        action_spec,
        hidden_dim=200,
        rnn_hidden_dim=200,
        state_dim=30,
    ):
        super().__init__()

        # Prior
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.action_state_projector = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ELU())
        self.rnn_to_prior_projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, state_dim)
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_shape = action_spec.shape
        self._unsqueeze_rnn_input = version.parse(torch.__version__) < version.parse(
            "1.11"
        )

    def forward(self, state, belief, action):
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        unsqueeze = False
        if self._unsqueeze_rnn_input and action_state.ndimension() == 1:
            if belief is not None:
                belief = belief.unsqueeze(0)
            action_state = action_state.unsqueeze(0)
            unsqueeze = True
        belief = self.rnn(action_state, belief)
        if unsqueeze:
            belief = belief.squeeze(0)

        logits = self.rnn_to_prior_projector(belief)
        dist = self.get_distribution(logits)
        state = dist.rsample()
        return logits, state, belief
    
    def get_distribution(self, state):
        dist = OneHotCategorical(logits=state, grad_method=Repa.PassThrough)
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

    def __init__(self, hidden_dim=200, state_dim=30, scale_lb=0.1):
        super().__init__()
        self.obs_rnn_to_post_projector = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * state_dim),
        ),
        self.hidden_dim = hidden_dim

    def forward(self, belief, obs_embedding):
        logits = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        dist = self.get_distribution(logits)
        state = dist.rsample()
        return logits, state
