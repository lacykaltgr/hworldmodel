
from tensordict.nn.distributions import TruncatedNormal
import torch.distributions as dist
import torch
import torch.nn.functional as F

from .jit_functions import symlog, symexp, hyperbolic_forward, hyperbolic_backward 


def unimix_logits(logits, unimix_ratio=0.0):
    if unimix_ratio > 0.0:
        probs = F.softmax(logits, dim=-1)
        probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
        logits = torch.log(probs)
    return logits



class TruncNormalDist(TruncatedNormal):
  
  """
  Truncated Normal Distribution
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  @property
  def mode(self):
    return self.mean


      
class UnnormalizedHuber(dist.Normal):
    """
    Unnormalized Huber Distribution
    """
  
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2)
            - self._threshold
        )

    def mode(self):
        return self.mean



class MSEDist(dist.Distribution):

  def __init__(self, mode, event_dims, fwd_transform, bwd_transform, tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, event_dims + 1)])
    self._fwd = fwd_transform
    self._bwd = bwd_transform
    self._tol = tol
    batch_shape = mode.shape[:len(mode.shape) - event_dims]
    event_shape = mode.shape[len(mode.shape) - event_dims:]
    super().__init__(batch_shape, event_shape)

  def mode(self):
    return self._bwd(self._mode)

  def mean(self):
    return self._bwd(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = (self._mode - self._fwd(value)) ** 2
    distance = torch.where(distance < self._tol, 0, distance)
    loss = distance.sum(self._dims)
    return -loss


class SymlogMSEDist(MSEDist):
  
    def __init__(mode, dims, agg='sum', tol=1e-8):
      fwd, bwd = symlog, symexp
      super().__init__(mode, dims, fwd, bwd, agg, tol)
      
class HyperbolicMSEDist(MSEDist):
    
      def __init__(mode, dims, agg='sum', tol=1e-8):
        fwd, bwd = hyperbolic_forward, hyperbolic_backward
        super().__init__(mode, dims, fwd, bwd, agg, tol)

  
class TwoHotDist(dist.Distribution):
    def __init__(
        self,
        logits,
        buckets,
        fwd_transform=symlog,
        bwd_transform=symexp,
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = buckets
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = fwd_transform
        self.transbwd = bwd_transform

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)
      
      
class SymlogTwoHotDist(TwoHotDist):
  def __init__(self, logits, low=-20.0, high=20.0):
    steps = logits.shape[-1] # 255?
    buckets= torch.linspace(low, high, steps=steps).to(logits.device)
    super().__init__(logits, buckets, symlog, symexp)
    

class SymexpTwoHotDist(TwoHotDist):
  def __init__(self, logits, amplitude=20.0):
    steps = logits.shape[-1] # 255?
    if steps % 2 == 1:
        half = torch.linspace(-amplitude, 0, steps=(steps - 1) // 2 + 1)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1][::-1]], 0)
    else:
        half = torch.linspace(-amplitude, 0, steps // 2)
        half = symexp(half)
        bins = torch.concatenate([half, -half[::-1]], 0)
    super().__init__(logits, bins, symexp, symlog)
    

class HyperbolicTwoHotDist(TwoHotDist):
  def __init__(self, logits, amplitude = 300, eps=1e-3):
    steps = logits.shape[-1] # 255?
    bins = hyperbolic_backward(torch.linspace(-amplitude, amplitude, steps))
    super().__init__(logits, bins)