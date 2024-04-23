
from torchrl.modules import TruncatedNormal
import torch.distributions as dist

class TruncNormalDist(TruncatedNormal):

  def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
    super().__init__(loc, scale, low, high)
    self._clip = clip
    self._mult = mult

  def sample(self, *args, **kwargs):
    event = super().sample(*args, **kwargs)
    if self._clip:
      clipped = tf.clip_by_value(
          event, self.low + self._clip, self.high - self._clip)
      event = event - tf.stop_gradient(event) + tf.stop_gradient(clipped)
    if self._mult:
      event *= self._mult
    return event