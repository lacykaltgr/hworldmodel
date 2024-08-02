from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np


class Transform(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def undo(self, x):
        return x
    
class ProposalTransform(Transform):
    def __init__(self, proposal):
        self.proposal = proposal

    @abstractmethod
    def mean(self):
        pass

class JitterTransform(nn.Module):
    def __init__(self, jitter_size):
        super(JitterTransform, self).__init__()
        self.jitter_size = jitter_size
        self.ox = 0
        self.oy = 0

    def forward(self, x):

        from .utils import roll
        
        # TODO: jittering for non-image data
        
        ox, oy = np.random.randint(-self.jitter_size, self.jitter_size + 1, 2)  # use uniform distribution
        ox, oy = int(ox), int(oy)
        self.ox, self.oy = ox, oy
        x.data = roll(roll(x.data, ox, -1), oy, -2)
        return x
    
    def undo(self, x):
        from .utils import roll

        x.data = roll(roll(x.data, -self.ox, -1), -self.oy, -2)
        return x
    
        
class NormalizeTransform(nn.Module):
    def __init__(self, norm, scale, eps=1e-8):
        super(NormalizeTransform, self).__init__()
        self.norm = norm
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        from .utils import batch_std

        data_idx = batch_std(x.data) + self.eps > self.norm / self.scale
        x.data[data_idx] = (x.data / (batch_std(x.data, keepdim=True) + self.eps) * self.norm / self.scale)[
            data_idx]
        return x
    
class BackpropNormTransform(nn.Module):
    """

    """
    def __init__(self, train_norm, scale=1.0, eps=1e-8):
        super(BackpropNormTransform, self).__init__()
        self.norm = train_norm
        self.scale = scale
        self.eps = eps
        
    def forward(self, x):
        from .utils import batch_std

        x_ = x  # img = inputs
        if self.train_norm and self.train_norm > 0.0:
            img_idx = batch_std(x.data) + self.eps > self.train_norm / self.scale  # images to update
            if img_idx.any():
                x_ = x.clone()  # avoids overwriting original image but lets gradient through
                x_[img_idx] = ((x_[img_idx] / (batch_std(x_[img_idx], keepdim=True) +
                                                self.eps)) * (self.train_norm / self.scale))
        return x_
    
class ClipTransform(nn.Module):
    def __init__(self, min_val, max_val):
        super(ClipTransform, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x.data = torch.clamp(x.data, self.min_val, self.max_val)
        return x
    
    
class LocScaleDistributionTransform(nn.Module):
    def __init__(self, distribution, n_samples, variance_trainable, variance):
        super(LocScaleDistributionTransform, self).__init__()
        if distribution == "normal":
            self.distribution_cls = torch.distributions.Normal
        elif distribution == "laplace":
            self.distribution_cls = torch.distributions.Laplace
        else:
            raise NotImplementedError(f"Distribution {distribution} not supported")

        self.variance_trainable = variance_trainable
        self.variance = variance
        self.n_samples = n_samples
        
        
    def forward(self, x):
        if self.variance_trainable:
            loc, scale = torch.chunk(x, 2, dim=-1)  # TODO : which dimension to split
        else:
            loc = x
            scale = self.variance
        distribution = self.distribution_cls(loc, scale)
        samples = distribution.rsample((self.n_samples,))
        return samples
    