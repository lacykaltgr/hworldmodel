from torchrl.modules.distributions import NormalParamWrapper
from torchrl.modules.models.models import MLP
from torch import nn
import torch

class DreamerActorV2(nn.Module):
    """Dreamer actor network.

    This network is used to predict the action distribution given the
    the stochastic state and the deterministic belief at the current
    time step.
    It outputs the mean and the scale of the action distribution.

    Reference: https://arxiv.org/abs/1912.01603

    Args:
        out_features (int): Number of output features.
        depth (int, optional): Number of hidden layers.
            Defaults to 4.
        num_cells (int, optional): Number of hidden units per layer.
            Defaults to 200.
        activation_class (nn.Module, optional): Activation class.
            Defaults to nn.ELU.
        std_bias (float, optional): Bias of the softplus transform.
            Defaults to 5.0.
        std_min_val (float, optional): Minimum value of the standard deviation.
            Defaults to 1e-4.
    """

    def __init__(
        self,
        out_features,
        depth=4,
        num_cells=200,
        activation_class=nn.ELU,
        std_min_val=0.1,
    ):
        super().__init__()
        self.std_min_val = std_min_val
        self.backbone = MLP(
            out_features=2 * out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
        )

    def forward(self, state, belief):
        tensors = (state.detach(), belief.detach())
        net_output = self.backbone(*tensors)
        loc, scale = net_output.chunk(2, -1)
        loc = torch.tanh(loc)
        scale = 2 * nn.functional.sigmoid(scale / 2) + self.std_min_val
        return loc, scale