
import torch

def absmax(cls):
    class AbsMaxDist(cls):
        def __init__(self,  absmax=None, **dist_kwargs):
            super().__init__(**dist_kwargs)
            self.absmax = absmax

        def mode(self):
            out = self._dist.mean
            if self.absmax is not None:
                out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            return out

        def sample(self, sample_shape=()):
            out = self._dist.rsample(sample_shape)
            if self.absmax is not None:
                out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            return out
    
    return AbsMaxDist