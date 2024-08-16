import warnings
import torch
from torch import nn

class DepthDecoder(nn.Module):
    """Observation decoder network.

    Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        channels (int, optional): Number of hidden units in the last layer.
            Defaults to 32.
        num_layers (int, optional): Depth of the network. Defaults to 4.
        kernel_sizes (int or list of int, optional): the kernel_size of each layer.
            Defaults to ``[5, 5, 6, 6]`` if num_layers if 4, else ``[5] * num_layers``.
    """

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

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D, 1, 1)
        obs_decoded = self.decoder(latent)
        _, C, H, W = obs_decoded.shape
        obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        return obs_decoded
