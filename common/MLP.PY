from typing import Callable, Type, List, Sequence

import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.modules.distributions import TanhNormal

from jit_functions import symlog


class SymlogMLP(MLP):
    def __init__(
        self,
        in_features = None,
        out_features = None,
        depth = None,
        num_cells = None,
        activation_class = nn.SiLU,
        activation_kwargs = None,
        norm_class = None,
        norm_kwargs = None,
        dropout= None,
        bias_last_layer = True,
        single_bias_last_layer = False,
        layer_class = nn.Linear,
        layer_kwargs = None,
        activate_last_layer = False,
    ):
        super(MLP, self).__init__(
            in_features=in_features,
            out_features=out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            bias_last_layer=bias_last_layer,
            single_bias_last_layer=single_bias_last_layer,
            layer_class=layer_class,
            layer_kwargs=layer_kwargs,
            activate_last_layer=activate_last_layer,
        )
        self.layers.apply(tools.weight_init)

    def forward(self, features):
        x = features
        x = symlog(x)
        out = super().forward(x)
        return out

def get_dist_class(self, dist, mean, std, shape):
    if self._dist == "tanh_normal":
        return TanhNormal
    elif self._dist == "normal":
        std = (self._max_std - self._min_std) * torch.sigmoid(
            std + 2.0
        ) + self._min_std
        dist = torchd.normal.Normal(torch.tanh(mean), std)
        dist = tools.ContDist(
            torchd.independent.Independent(dist, 1), absmax=self._absmax
        )
    elif self._dist == "normal_std_fixed":
        dist = torchd.normal.Normal(mean, self._std)
        dist = tools.ContDist(
            torchd.independent.Independent(dist, 1), absmax=self._absmax
        )
    elif self._dist == "trunc_normal":
        mean = torch.tanh(mean)
        std = 2 * torch.sigmoid(std / 2) + self._min_std
        dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
        dist = tools.ContDist(
            torchd.independent.Independent(dist, 1), absmax=self._absmax
        )
    elif self._dist == "onehot":
        dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
    elif self._dist == "onehot_gumble":
        dist = tools.ContDist(
            torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
        )
    elif dist == "huber":
        dist = tools.ContDist(
            torchd.independent.Independent(
                tools.UnnormalizedHuber(mean, std, 1.0),
                len(shape),
                absmax=self._absmax,
            )
        )
    elif dist == "binary":
        dist = tools.Bernoulli(
            torchd.independent.Independent(
                torchd.bernoulli.Bernoulli(logits=mean), len(shape)
            )
        )
    elif dist == "symlog_disc":
        dist = tools.DiscDist(logits=mean, device=self._device)
    elif dist == "symlog_mse":
        dist = tools.SymlogDist(mean)
    else:
        raise NotImplementedError(dist)
    return dist
